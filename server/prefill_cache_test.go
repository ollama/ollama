package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestPrefillCacheFingerprintChangesWithOptions(t *testing.T) {
	base := &runnerRef{
		modelKey:    "/models/a.gguf",
		numParallel: 1,
		model: &Model{
			ModelPath: "/models/a.gguf",
		},
		Options: &api.Options{Runner: api.Runner{NumCtx: 4096, NumGPU: 1}},
	}
	fp1 := fingerprintRunner(base)

	changed := *base
	changed.Options = &api.Options{Runner: api.Runner{NumCtx: 8192, NumGPU: 1}}
	fp2 := fingerprintRunner(&changed)

	if fp1 == fp2 {
		t.Fatal("expected fingerprint to change when NumCtx changes")
	}
}

func TestPrefillCacheStoreEligible(t *testing.T) {
	dir := t.TempDir()
	store := &prefillCacheStore{dir: dir, entries: map[string]*prefillCacheEntry{}}

	runner := &runnerRef{
		numParallel: 1,
		modelKey:    "model",
		model:       &Model{},
		Options:     &api.Options{Runner: api.Runner{NumCtx: 2048}},
	}
	if !store.eligible(runner) {
		t.Fatal("expected text runner to be eligible")
	}

	runner.model.ProjectorPaths = []string{"proj.gguf"}
	if store.eligible(runner) {
		t.Fatal("expected multimodal runner to be ineligible")
	}

	runner.model.ProjectorPaths = nil
	runner.numParallel = 2
	if store.eligible(runner) {
		t.Fatal("expected numParallel>1 to be ineligible")
	}

	runner.numParallel = 1
	runner.isImagegen = true
	if store.eligible(runner) {
		t.Fatal("expected imagegen runner to be ineligible")
	}
}

func TestPrefillCacheStoreRecordAndEvict(t *testing.T) {
	dir := t.TempDir()
	store := &prefillCacheStore{dir: dir, entries: map[string]*prefillCacheEntry{}}

	runner := &runnerRef{
		numParallel: 1,
		modelKey:    "model-a",
		model:       &Model{ModelPath: "model-a"},
		Options:     &api.Options{Runner: api.Runner{NumCtx: 2048}},
	}
	path, ok := store.pathFor(runner)
	if !ok {
		t.Fatal("expected path")
	}
	if err := os.WriteFile(path, make([]byte, 1024), 0o600); err != nil {
		t.Fatal(err)
	}
	store.record(runner, path)

	if _, ok := store.lookup(runner); !ok {
		t.Fatal("expected saved cache to be found")
	}

	store.mu.Lock()
	store.entries[fingerprintRunner(runner)].bytes = prefillCacheMaxBytes + 1
	store.evictLocked()
	store.mu.Unlock()

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("expected LRU eviction to remove cache file")
	}
}

func TestPrefillCacheStorePaths(t *testing.T) {
	dir := t.TempDir()
	store := &prefillCacheStore{dir: dir, entries: map[string]*prefillCacheEntry{}}

	llamaRunner := &runnerRef{
		numParallel: 1,
		modelKey:    "gguf",
		model:       &Model{ModelPath: "gguf"},
		Options:     &api.Options{Runner: api.Runner{NumCtx: 2048}},
	}
	llamaPath, ok := store.pathFor(llamaRunner)
	if !ok {
		t.Fatal("expected llama path")
	}
	if filepath.Ext(llamaPath) != ".bin" {
		t.Fatalf("expected .bin file for llama runner, got %q", llamaPath)
	}

	mlxRunner := &runnerRef{
		numParallel: 1,
		modelKey:    "digest:abc",
		model: &Model{
			Digest: "abc",
			Config: model.ConfigV2{ModelFormat: "safetensors"},
		},
		Options: &api.Options{Runner: api.Runner{NumCtx: 2048}},
	}
	mlxPath, ok := store.pathFor(mlxRunner)
	if !ok {
		t.Fatal("expected mlx path")
	}
	if filepath.Base(mlxPath) == "" || filepath.Ext(mlxPath) == ".bin" {
		t.Fatalf("expected mlx directory path, got %q", mlxPath)
	}
}
