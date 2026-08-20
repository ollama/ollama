package mlxrunner

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/mlxrunner/cache"
)

// KVConnector loads and stores restorable MLX prefix-cache snapshots.
type KVConnector interface {
	Lookup(keys []trieKey) (*KVConnectorMatch, error)
	SnapshotOffsets(inputs []int32, draftLookahead int) []int
	Store(entry *KVConnectorEntry) error
}

// KVConnectorMatch is a reusable external prefix match for the current request.
type KVConnectorMatch struct {
	Offset    int
	Snapshots []cache.Snapshot
}

// KVConnectorEntry is a restorable prefix-cache snapshot that can be stored.
type KVConnectorEntry struct {
	Offset    int
	Keys      []trieKey
	Snapshots []cache.Snapshot
}

// ExampleConnector stores paged-out prefix-cache snapshots as local files so
// the framework can be wired end to end before remote backends are added.
type ExampleConnector struct {
	modelDir string
}

func newKVConnector(modelName string) (KVConnector, error) {
	switch kind := strings.TrimSpace(envconfig.MLXKVConnector()); kind {
	case "", "off", "none":
		return nil, nil
	case "example", "file":
		return newExampleConnector(modelName)
	default:
		return nil, fmt.Errorf("unsupported OLLAMA_MLX_KV_CONNECTOR %q", kind)
	}
}

func newExampleConnector(modelName string) (*ExampleConnector, error) {
	root := strings.TrimSpace(envconfig.MLXKVConnectorDir())
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, err
		}
		root = filepath.Join(home, ".ollama", "kvconnector", "mlx")
	}

	modelDir := filepath.Join(root, modelCacheID(modelName))
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return nil, err
	}

	return &ExampleConnector{modelDir: modelDir}, nil
}

func (c *ExampleConnector) Lookup(keys []trieKey) (*KVConnectorMatch, error) {
	if len(keys) == 0 {
		return nil, nil
	}

	digests := keyPrefixDigests(keys)
	for offset := len(keys); offset > 0; offset-- {
		path := c.snapshotPath(offset, digests[offset-1])
		if _, err := os.Stat(path); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, err
		}
		persisted, err := cache.LoadSnapshots(path)
		if err == nil {
			if persisted.Offset != offset {
				closeSnapshots(persisted.Snapshots)
				return nil, fmt.Errorf("kvconnector snapshot %s stored offset %d, want %d", path, persisted.Offset, offset)
			}
			return &KVConnectorMatch{Offset: persisted.Offset, Snapshots: persisted.Snapshots}, nil
		}
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		return nil, err
	}

	return nil, nil
}

func (c *ExampleConnector) SnapshotOffsets(inputs []int32, draftLookahead int) []int {
	if len(inputs) <= 1 {
		return nil
	}

	var offsets []int
	for offset := 8192; offset < len(inputs); offset += 8192 {
		offsets = append(offsets, offset)
	}
	offsets = append(offsets, len(inputs)-1+draftLookahead)
	slices.Sort(offsets)
	return slices.Compact(offsets)
}

func (c *ExampleConnector) Store(entry *KVConnectorEntry) error {
	if entry == nil || entry.Offset <= 0 || len(entry.Keys) == 0 {
		return nil
	}
	if entry.Offset != len(entry.Keys) {
		return fmt.Errorf("kvconnector entry offset %d does not match %d keys", entry.Offset, len(entry.Keys))
	}

	digest := keyPrefixDigests(entry.Keys)[len(entry.Keys)-1]
	path := c.snapshotPath(entry.Offset, digest)
	tmp := filepath.Join(c.modelDir, fmt.Sprintf(".%s.%d.tmp", filepath.Base(path), time.Now().UnixNano()))

	if err := cache.SaveSnapshots(tmp, entry.Offset, entry.Snapshots); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func (c *ExampleConnector) snapshotPath(offset int, digest string) string {
	return filepath.Join(c.modelDir, fmt.Sprintf("%06d-%s.safetensors", offset, digest[:16]))
}

func modelCacheID(modelName string) string {
	sum := sha256.Sum256([]byte(modelName))
	label := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		case r >= '0' && r <= '9':
			return r
		case r == '-', r == '_':
			return r
		default:
			return '-'
		}
	}, filepath.Base(modelName))
	label = strings.Trim(label, "-")
	if label == "" {
		label = "model"
	}
	return fmt.Sprintf("%s-%s", label, hex.EncodeToString(sum[:6]))
}

func keyPrefixDigests(keys []trieKey) []string {
	h := sha256.New()
	digests := make([]string, len(keys))
	var buf [8]byte
	for i, key := range keys {
		binary.LittleEndian.PutUint64(buf[:], uint64(key))
		_, _ = h.Write(buf[:])
		digests[i] = hex.EncodeToString(h.Sum(nil))
	}
	return digests
}

func closeSnapshots(snaps []cache.Snapshot) {
	for _, snap := range snaps {
		if snap != nil {
			snap.Close()
		}
	}
}
