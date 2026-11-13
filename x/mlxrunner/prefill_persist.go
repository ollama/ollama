package mlxrunner

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const prefillPersistVersion = 1

var errPrefillCacheEmpty = errors.New("prefill cache empty")

type prefillPersistManifest struct {
	Version        int                      `json:"version"`
	Offset         int                      `json:"offset"`
	Tokens         []int32                  `json:"tokens"`
	DraftLookahead int                      `json:"draft_lookahead"`
	Layers         []cache.LayerPersistMeta `json:"layers"`
}

func (c *prefixCache) saveToPath(path string) error {
	offset := c.minCacheOffset()
	if offset <= 0 {
		return errPrefillCacheEmpty
	}

	tokens, err := c.activeTokenPrefix(offset)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(path, 0o700); err != nil {
		return err
	}

	manifest := prefillPersistManifest{
		Version:        prefillPersistVersion,
		Offset:         offset,
		Tokens:         tokens,
		DraftLookahead: c.draftLookahead,
	}

	for i, layer := range c.caches {
		if layer == nil {
			manifest.Layers = append(manifest.Layers, cache.LayerPersistMeta{})
			continue
		}
		arrays, meta, err := cache.ExportLayer(layer)
		if err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
		manifest.Layers = append(manifest.Layers, meta)

		layerPath := filepath.Join(path, fmt.Sprintf("layer_%d.safetensors", i))
		named := make(map[string]*mlx.Array, len(arrays))
		for k, v := range arrays {
			named[k] = v
		}
		mlx.AsyncEval(mapsValues(named)...)
		if err := mlx.SaveSafetensors(layerPath, named); err != nil {
			return fmt.Errorf("layer %d save: %w", i, err)
		}
	}

	manifestBytes, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(path, "manifest.json"), manifestBytes, 0o600)
}

func mapsValues(m map[string]*mlx.Array) []*mlx.Array {
	out := make([]*mlx.Array, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	return out
}

func (c *prefixCache) restoreFromPath(path string) error {
	manifestBytes, err := os.ReadFile(filepath.Join(path, "manifest.json"))
	if err != nil {
		return err
	}
	var manifest prefillPersistManifest
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		return err
	}
	if manifest.Version != prefillPersistVersion {
		return fmt.Errorf("unsupported prefill cache version %d", manifest.Version)
	}
	if manifest.Offset <= 0 || len(manifest.Tokens) < manifest.Offset {
		return fmt.Errorf("invalid prefill cache manifest")
	}
	if len(manifest.Layers) != len(c.caches) {
		return fmt.Errorf("layer count mismatch")
	}

	c.draftLookahead = manifest.DraftLookahead
	c.freeAll()
	c.root = nil
	c.activePath = nil
	c.pagedOutBytes = 0
	c.ensureRoot()

	for i, layer := range c.caches {
		if layer == nil {
			continue
		}
		layerPath := filepath.Join(path, fmt.Sprintf("layer_%d.safetensors", i))
		loaded := make(map[string]*mlx.Array)
		for name, arr := range mlx.Load(layerPath) {
			loaded[name] = arr
		}
		if len(loaded) == 0 {
			return fmt.Errorf("layer %d: empty safetensors", i)
		}
		if err := cache.ImportLayer(layer, loaded, manifest.Layers[i]); err != nil {
			return fmt.Errorf("layer %d: %w", i, err)
		}
	}

	if err := c.installActivePrefix(manifest.Tokens[:manifest.Offset], manifest.Offset); err != nil {
		return err
	}
	return nil
}

func (c *prefixCache) activeTokenPrefix(offset int) ([]int32, error) {
	if len(c.activePath) == 0 {
		return nil, fmt.Errorf("no active cache path")
	}
	keys := make([]trieKey, 0)
	for _, node := range c.activePath[1:] {
		keys = append(keys, node.tokens...)
	}
	tokens := trieKeysToTokens(keys, c.draftLookahead)
	if len(tokens) < offset {
		return nil, fmt.Errorf("active path shorter than cache offset")
	}
	return slices.Clone(tokens[:offset]), nil
}

func trieKeysToTokens(keys []trieKey, draftLookahead int) []int32 {
	switch draftLookahead {
	case 0:
		out := make([]int32, len(keys))
		for i, k := range keys {
			out[i] = int32(k)
		}
		return out
	case 1:
		if len(keys) == 0 {
			return nil
		}
		out := make([]int32, len(keys)+1)
		out[0] = int32(keys[0] >> 32)
		for i, k := range keys {
			out[i+1] = int32(k)
		}
		return out
	default:
		panic(fmt.Sprintf("prefixCache: unsupported draft look-ahead %d", draftLookahead))
	}
}

func tokensToTrieKeys(tokens []int32, draftLookahead int) []trieKey {
	switch draftLookahead {
	case 0:
		keys := make([]trieKey, len(tokens))
		for i, t := range tokens {
			keys[i] = trieKey(t)
		}
		return keys
	case 1:
		if len(tokens) <= 1 {
			return nil
		}
		keys := make([]trieKey, len(tokens)-1)
		for i := range keys {
			keys[i] = trieKey(uint32(tokens[i]))<<32 | trieKey(uint32(tokens[i+1]))
		}
		return keys
	default:
		panic(fmt.Sprintf("prefixCache: unsupported draft look-ahead %d", draftLookahead))
	}
}

func (c *prefixCache) installActivePrefix(tokens []int32, offset int) error {
	keys := tokensToTrieKeys(tokens, c.draftLookahead)
	if len(keys) == 0 {
		return nil
	}

	child := &trieNode{
		tokens:    slices.Clone(keys),
		endOffset: offset,
		parent:    c.root,
		lastUsed:  c.root.lastUsed,
	}
	snaps := make([]cache.Snapshot, len(c.caches))
	for j, kv := range c.caches {
		if kv == nil {
			continue
		}
		snaps[j] = kv.Snapshot(0)
	}
	child.setSnapshots(snaps, &c.pagedOutBytes)
	c.root.children = []*trieNode{child}
	c.activePath = []*trieNode{c.root, child}
	return nil
}

func (r *Runner) savePrefillCache(path string) error {
	if r.cache == nil {
		return errPrefillCacheEmpty
	}
	return r.cache.saveToPath(path)
}

func (r *Runner) restorePrefillCache(path string) error {
	if r.cache == nil {
		return fmt.Errorf("no prefix cache")
	}
	return r.cache.restoreFromPath(path)
}
