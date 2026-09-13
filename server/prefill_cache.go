package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
)

const prefillCacheMaxBytes int64 = 8 << 30 // 8 GiB

type prefillCacheFingerprint struct {
	ModelKey     string   `json:"model_key"`
	Digest       string   `json:"digest,omitempty"`
	Adapters     []string `json:"adapters,omitempty"`
	Projectors   []string `json:"projectors,omitempty"`
	RunnerKind   string   `json:"runner_kind"`
	NumParallel  int      `json:"num_parallel"`
	ContextShift bool     `json:"context_shift"`
	NumCtx       int      `json:"num_ctx"`
	NumGPU       int      `json:"num_gpu"`
	NumBatch     int      `json:"num_batch"`
	UseMMap      *bool    `json:"use_mmap,omitempty"`
	KvCacheType  string   `json:"kv_cache_type,omitempty"`
}

type prefillCacheEntry struct {
	fingerprint string
	path        string
	bytes       int64
	lastUsed    time.Time
}

type prefillCacheStore struct {
	mu      sync.Mutex
	dir     string
	entries map[string]*prefillCacheEntry
}

func newPrefillCacheStore() (*prefillCacheStore, error) {
	dir := envconfig.PrefillCacheDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	return &prefillCacheStore{
		dir:     dir,
		entries: make(map[string]*prefillCacheEntry),
	}, nil
}

func (s *prefillCacheStore) Dir() string {
	if s == nil {
		return ""
	}
	return s.dir
}

func (s *prefillCacheStore) Cleanup() {
	if s == nil {
		return
	}
	_ = os.RemoveAll(s.dir)
}

func runnerKind(runner *runnerRef) string {
	if runner.isImagegen {
		return "imagegen"
	}
	if runner.model != nil && runner.model.IsMLX() {
		return "mlx"
	}
	return "llama"
}

func (s *prefillCacheStore) eligible(runner *runnerRef) bool {
	if runner == nil || runner.model == nil || runner.Options == nil {
		return false
	}
	if runner.isImagegen {
		return false
	}
	if runner.numParallel != 1 {
		return false
	}
	if len(runner.model.ProjectorPaths) > 0 {
		return false
	}
	return true
}

func fingerprintRunner(runner *runnerRef) string {
	fp := prefillCacheFingerprint{
		ModelKey:     runner.modelKey,
		Digest:       runner.model.Digest,
		Adapters:     slices.Clone(runner.model.AdapterPaths),
		Projectors:   slices.Clone(runner.model.ProjectorPaths),
		RunnerKind:   runnerKind(runner),
		NumParallel:  runner.numParallel,
		ContextShift: runner.contextShift,
		NumCtx:       runner.Options.NumCtx,
		NumGPU:       runner.Options.NumGPU,
		NumBatch:     runner.Options.NumBatch,
		UseMMap:      runner.Options.UseMMap,
		KvCacheType:  envconfig.KvCacheType(),
	}
	sort.Strings(fp.Adapters)
	sort.Strings(fp.Projectors)

	data, err := json.Marshal(fp)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func (s *prefillCacheStore) pathFor(runner *runnerRef) (string, bool) {
	if !s.eligible(runner) {
		return "", false
	}
	fp := fingerprintRunner(runner)
	if fp == "" {
		return "", false
	}
	switch runnerKind(runner) {
	case "mlx":
		return filepath.Join(s.dir, fp), true
	default:
		return filepath.Join(s.dir, fp+".bin"), true
	}
}

func (s *prefillCacheStore) lookup(runner *runnerRef) (string, bool) {
	path, ok := s.pathFor(runner)
	if !ok {
		return "", false
	}
	if _, err := os.Stat(path); err != nil {
		return "", false
	}
	return path, true
}

func entrySize(path string, kind string) int64 {
	switch kind {
	case "mlx":
		var total int64
		_ = filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if info, err := d.Info(); err == nil {
				total += info.Size()
			}
			return nil
		})
		return total
	default:
		info, err := os.Stat(path)
		if err != nil {
			return 0
		}
		return info.Size()
	}
}

func (s *prefillCacheStore) record(runner *runnerRef, path string) {
	if s == nil || path == "" {
		return
	}
	fp := fingerprintRunner(runner)
	if fp == "" {
		return
	}

	bytes := entrySize(path, runnerKind(runner))
	s.mu.Lock()
	defer s.mu.Unlock()

	if old, ok := s.entries[fp]; ok && old.path != path {
		_ = os.RemoveAll(old.path)
	}
	s.entries[fp] = &prefillCacheEntry{
		fingerprint: fp,
		path:        path,
		bytes:       bytes,
		lastUsed:    time.Now(),
	}
	s.evictLocked()
}

func (s *prefillCacheStore) evictLocked() {
	var total int64
	for _, e := range s.entries {
		total += e.bytes
	}
	if total <= prefillCacheMaxBytes {
		return
	}

	type candidate struct {
		fp string
		e  *prefillCacheEntry
	}
	candidates := make([]candidate, 0, len(s.entries))
	for fp, e := range s.entries {
		candidates = append(candidates, candidate{fp: fp, e: e})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].e.lastUsed.Before(candidates[j].e.lastUsed)
	})

	for _, c := range candidates {
		if total <= prefillCacheMaxBytes {
			break
		}
		slog.Debug("evicting prefill cache entry", "path", c.e.path, "bytes", c.e.bytes)
		_ = os.RemoveAll(c.e.path)
		delete(s.entries, c.fp)
		total -= c.e.bytes
	}
}
