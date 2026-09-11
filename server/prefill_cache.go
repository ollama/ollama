// prefill_cache.go owns the lifecycle of the prefill caches that runners save
// and restore: where a snapshot lives, which runner may reuse it, when it is
// written and read, and how much disk the set of them may hold. Runners are
// reached only through llm.PrefillCachePersistor, so nothing here knows how a
// snapshot is produced.
//
// Key properties:
//   - The feature is off unless OLLAMA_PREFILL_CACHE is set. Without it the
//     cache root is empty and every entry point returns having done nothing.
//   - The root is a fresh temporary directory per daemon, removed when the
//     daemon's context ends, so a snapshot never outlives the process that
//     wrote it.
//   - A snapshot's directory is named by a hash of the same model, adapter and
//     option set that needsReload compares, so a runner only ever finds a
//     snapshot it would not have reloaded for.
//   - Saving talks to the runner over HTTP, so it runs before loadedMu is
//     taken; pruning takes loadedMu, so it runs after the runner's refMu is
//     released. Neither holds one scheduler lock across the other.
package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

const (
	maxPrefillCacheDiskBytes   int64 = 8 << 30
	prefillCachePersistTimeout       = time.Minute
	prefillCacheRootPrefix           = "ollama-prefill-cache-"
)

func initPrefillCacheRoot(ctx context.Context) string {
	if !envconfig.PrefillCache() {
		return ""
	}
	root, err := os.MkdirTemp("", prefillCacheRootPrefix)
	if err != nil {
		slog.Warn("prefill cache persistence disabled", "error", err)
		return ""
	}
	go func() {
		<-ctx.Done()
		if err := os.RemoveAll(root); err != nil {
			slog.Debug("failed to remove prefill cache directory", "path", root, "error", err)
		}
	}()

	return root
}

type prefillCacheIdentity struct {
	Runner       string     `json:"runner"`
	Model        string     `json:"model"`
	Draft        string     `json:"draft,omitempty"`
	Adapters     []string   `json:"adapters,omitempty"`
	Projectors   []string   `json:"projectors,omitempty"`
	Options      api.Runner `json:"options"`
	NumParallel  int        `json:"num_parallel"`
	KVCacheType  string     `json:"kv_cache_type,omitempty"`
	ContextShift bool       `json:"context_shift,omitempty"`
}

func (s *Scheduler) prefillCachePath(identity prefillCacheIdentity) string {
	if s.prefillCacheRoot == "" {
		return ""
	}
	data, err := json.Marshal(identity)
	if err != nil {
		slog.Debug("failed to identify prefill cache", "error", err)
		return ""
	}
	sum := sha256.Sum256(data)
	return filepath.Join(s.prefillCacheRoot, hex.EncodeToString(sum[:]))
}

func (s *Scheduler) llamaPrefillCachePath(req *LlmRequest, opts api.Runner, numParallel int) string {
	return s.prefillCachePath(prefillCacheIdentity{
		Runner:       "llama.cpp",
		Model:        schedulerModelKey(req.model),
		Draft:        req.model.DraftPath,
		Adapters:     req.model.AdapterPaths,
		Projectors:   req.model.ProjectorPaths,
		Options:      opts,
		NumParallel:  numParallel,
		KVCacheType:  envconfig.KvCacheType(),
		ContextShift: req.contextShift,
	})
}

// restorePrefillCache runs with refMu held, while the runner is still loading
// and before any request can reach it.
func (runner *runnerRef) restorePrefillCache(ctx context.Context) {
	cache, ok := runner.llama.(llm.PrefillCachePersistor)
	if !ok {
		return
	}
	if runner.prefillCacheDir != "" {
		// Prefer other entries if another unload prunes while restore reads this one.
		now := time.Now()
		_ = os.Chtimes(runner.prefillCacheDir, now, now)
	}
	restoreCtx, cancel := context.WithTimeout(ctx, prefillCachePersistTimeout)
	defer cancel()
	if err := cache.RestorePrefillCache(restoreCtx); err != nil {
		slog.Warn(
			"failed to restore prefill cache; continuing with cold cache",
			"model", runner.modelKey,
			"error", err,
		)
	}
}

// savePrefillCache runs with refMu held. A runner that is still loading has
// no state worth saving.
func (runner *runnerRef) savePrefillCache(ctx context.Context) {
	if runner.llama == nil || runner.loading {
		return
	}
	if cache, ok := runner.llama.(llm.PrefillCachePersistor); ok && !runner.llama.HasExited() {
		saveCtx, cancel := context.WithTimeout(ctx, prefillCachePersistTimeout)
		defer cancel()
		if err := cache.SavePrefillCache(saveCtx); err != nil {
			slog.Warn("failed to save prefill cache; runner will unload without a snapshot", "model", runner.modelKey, "error", err)
		}
	}
}

// prunePrefillCache evicts the oldest entries until the cache is under
// maxPrefillCacheDiskBytes. Entries belonging to loaded runners are kept: a
// live runner writes into its own directory and holds it open.
func (s *Scheduler) prunePrefillCache() {
	if s.prefillCacheRoot == "" {
		return
	}
	prunePrefillCache(s.prefillCacheRoot, maxPrefillCacheDiskBytes, s.prefillCacheDirsInUse())
}

func (s *Scheduler) prefillCacheDirsInUse() map[string]struct{} {
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	dirs := make(map[string]struct{}, len(s.loaded))
	for _, runner := range s.loaded {
		if runner.prefillCacheDir != "" {
			dirs[runner.prefillCacheDir] = struct{}{}
		}
	}
	return dirs
}

func prunePrefillCache(root string, maxBytes int64, keep map[string]struct{}) {
	if root == "" || maxBytes < 0 {
		return
	}
	type cacheDir struct {
		path    string
		size    int64
		modTime time.Time
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Debug("failed to scan prefill cache", "path", root, "error", err)
		}
		return
	}
	var dirs []cacheDir
	var total int64
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		dir := cacheDir{
			path:    filepath.Join(root, entry.Name()),
			size:    prefillCacheDiskUsage(filepath.Join(root, entry.Name())),
			modTime: info.ModTime(),
		}
		dirs = append(dirs, dir)
		total += dir.size
	}
	if total <= maxBytes {
		return
	}
	slices.SortFunc(dirs, func(a, b cacheDir) int { return a.modTime.Compare(b.modTime) })
	for _, dir := range dirs {
		if total <= maxBytes {
			break
		}
		if _, ok := keep[dir.path]; ok {
			continue
		}
		if err := os.RemoveAll(dir.path); err != nil {
			// RemoveAll may have removed part of the directory before failing.
			total = prefillCacheDiskUsage(root)
			slog.Debug("failed to evict prefill cache", "path", dir.path, "error", err)
			continue
		}
		total -= dir.size
		slog.Debug("evicted prefill cache", "path", dir.path, "size", dir.size)
	}
}

func prefillCacheDiskUsage(path string) int64 {
	var size int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info.Mode().IsRegular() {
			size += info.Size()
		}
		return nil
	})
	return size
}
