package server

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestModelListCacheHydratesSummary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-cache", map[string]any{
		"test.context_length":   uint32(4096),
		"test.embedding_length": uint32(384),
	}, "{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")

	cache := newModelListCache()
	if err := cache.hydrate(context.Background()); err != nil {
		t.Fatalf("hydrate failed: %v", err)
	}

	summary, ok := cache.Get(model.ParseName("list-cache"))
	if !ok {
		t.Fatal("list summary missing")
	}

	if summary.Model != "list-cache:latest" || summary.Name != "list-cache:latest" {
		t.Fatalf("summary model/name = %q/%q, want list-cache:latest", summary.Model, summary.Name)
	}
	if summary.Digest == "" {
		t.Fatal("summary digest is empty")
	}
	if summary.Size == 0 {
		t.Fatal("summary size is zero")
	}
	if summary.Details.Family != "test" || summary.Details.Format != "gguf" {
		t.Fatalf("summary details = %+v, want gguf/test", summary.Details)
	}
	if summary.Details.ContextLength != 4096 {
		t.Fatalf("context length = %d, want 4096", summary.Details.ContextLength)
	}
	if summary.Details.EmbeddingLength != 384 {
		t.Fatalf("embedding length = %d, want 384", summary.Details.EmbeddingLength)
	}

	for _, capability := range []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert} {
		if !slices.Contains(summary.Capabilities, capability) {
			t.Fatalf("capabilities = %v, want %s", summary.Capabilities, capability)
		}
	}

	listModel := summary.ListModelResponse()
	if !slices.Contains(listModel.Capabilities, model.CapabilityTools) ||
		listModel.Details.ContextLength != 4096 ||
		listModel.Details.EmbeddingLength != 384 {
		t.Fatalf("list response = %+v, want capabilities/context/embedding", listModel)
	}
}

func TestModelListCacheRefreshUpdatesEntry(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-refresh", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	if err := cache.hydrate(context.Background()); err != nil {
		t.Fatalf("hydrate failed: %v", err)
	}

	name := model.ParseName("list-refresh")
	first, ok := cache.Get(name)
	if !ok {
		t.Fatal("list summary missing")
	}

	changeShowCacheManifest(t, "list-refresh")
	if err := cache.RefreshModel(name); err != nil {
		t.Fatalf("refresh failed: %v", err)
	}

	refreshed, ok := cache.Get(name)
	if !ok {
		t.Fatal("refreshed list summary missing")
	}
	if refreshed.Digest == first.Digest {
		t.Fatalf("digest did not change after refresh: %s", refreshed.Digest)
	}
	if cache.Len() != 1 {
		t.Fatalf("cache entries = %d, want 1", cache.Len())
	}
}

func TestModelListCacheMutationHooks(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	cache := newModelListCache()
	s := Server{modelCaches: &modelCaches{modelList: cache}}

	_, digest := createBinFile(t, map[string]any{"test.context_length": uint32(2048)}, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:  "list-hooks",
		Files:  map[string]string{"model.gguf": digest},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create model status = %d, want 200: %s", w.Code, w.Body.String())
	}

	if _, ok := cache.Get(model.ParseName("list-hooks")); !ok {
		t.Fatal("create did not refresh model list cache")
	}

	w = createRequest(t, s.CopyHandler, api.CopyRequest{
		Source:      "list-hooks",
		Destination: "list-hooks-copy",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("copy model status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := cache.Get(model.ParseName("list-hooks-copy")); !ok {
		t.Fatal("copy did not refresh model list cache")
	}

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Model: "list-hooks-copy"})
	if w.Code != http.StatusOK {
		t.Fatalf("delete model status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := cache.Get(model.ParseName("list-hooks-copy")); ok {
		t.Fatal("delete did not remove model list cache entry")
	}
}

func TestModelListCacheSyncsManifestChanges(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-sync-a", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	cache.Start(context.Background())
	if err := cache.Wait(context.Background()); err != nil {
		t.Fatalf("wait failed: %v", err)
	}

	createListCacheModel(t, "list-sync-b", map[string]any{"test.context_length": uint32(2048)}, "")
	models, err := cache.List(context.Background())
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}

	names := make([]string, 0, len(models))
	for _, m := range models {
		names = append(names, m.Name)
	}
	for _, want := range []string{"list-sync-a:latest", "list-sync-b:latest"} {
		if !slices.Contains(names, want) {
			t.Fatalf("names = %v, want %s", names, want)
		}
	}

	var other Server
	w := createRequest(t, other.DeleteHandler, api.DeleteRequest{Model: "list-sync-a"})
	if w.Code != http.StatusOK {
		t.Fatalf("delete model status = %d, want 200: %s", w.Code, w.Body.String())
	}

	models, err = cache.List(context.Background())
	if err != nil {
		t.Fatalf("list after delete failed: %v", err)
	}
	names = names[:0]
	for _, m := range models {
		names = append(names, m.Name)
	}
	if slices.Contains(names, "list-sync-a:latest") || !slices.Contains(names, "list-sync-b:latest") {
		t.Fatalf("names after delete = %v, want only list-sync-b", names)
	}
}

func TestModelListCacheSyncDropsStaleEntryOnRefreshFailure(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-stale", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	cache.Start(context.Background())
	if err := cache.Wait(context.Background()); err != nil {
		t.Fatalf("wait failed: %v", err)
	}

	name := model.ParseName("list-stale")
	if _, ok := cache.Get(name); !ok {
		t.Fatal("list summary missing")
	}

	changeShowCacheManifest(t, "list-stale")
	cache.build = func(model.Name, *manifest.Manifest) (modelListSummary, error) {
		return modelListSummary{}, errors.New("refresh failed")
	}

	models, err := cache.List(context.Background())
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("models = %+v, want stale entry removed", models)
	}
	if _, ok := cache.Get(name); ok {
		t.Fatal("stale entry remained in cache after refresh failure")
	}
}

func createListCacheModel(t *testing.T, name string, kv map[string]any, tmpl string) {
	t.Helper()
	_, digest := createBinFile(t, kv, nil)

	req := api.CreateRequest{
		Model:  name,
		Files:  map[string]string{"model.gguf": digest},
		Stream: &stream,
	}
	if tmpl != "" {
		req.Template = tmpl
	}

	var s Server
	w := createRequest(t, s.CreateHandler, req)
	if w.Code != http.StatusOK {
		t.Fatalf("create model status = %d, want 200: %s", w.Code, w.Body.String())
	}
}
