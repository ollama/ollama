package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/manifest"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestModelShowCacheLocalHitUsesManifestDigest(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "show-cache-local", map[string]any{"test.context_length": uint32(1024)})

	cache := newModelShowCache()
	calls := 0
	cache.getModelInfo = func(req api.ShowRequest) (*api.ShowResponse, error) {
		calls++
		return showCacheTestResponse(calls, req.Verbose), nil
	}

	first, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-local"})
	if err != nil {
		t.Fatalf("first GetLocal failed: %v", err)
	}
	second, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-local"})
	if err != nil {
		t.Fatalf("second GetLocal failed: %v", err)
	}

	if calls != 1 {
		t.Fatalf("getModelInfo calls = %d, want 1", calls)
	}
	if first.ModelInfo["call"] != 1 || second.ModelInfo["call"] != 1 {
		t.Fatalf("cached call markers = %v / %v, want both 1", first.ModelInfo["call"], second.ModelInfo["call"])
	}
}

func TestModelShowCacheLocalManifestDigestChangeRefreshes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "show-cache-refresh", map[string]any{"test.context_length": uint32(1024)})

	cache := newModelShowCache()
	calls := 0
	cache.getModelInfo = func(req api.ShowRequest) (*api.ShowResponse, error) {
		calls++
		return showCacheTestResponse(calls, req.Verbose), nil
	}

	if _, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-refresh"}); err != nil {
		t.Fatalf("first GetLocal failed: %v", err)
	}
	changeShowCacheManifest(t, "show-cache-refresh")
	refreshed, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-refresh"})
	if err != nil {
		t.Fatalf("refreshed GetLocal failed: %v", err)
	}

	if calls != 2 {
		t.Fatalf("getModelInfo calls = %d, want 2", calls)
	}
	if refreshed.ModelInfo["call"] != 2 {
		t.Fatalf("refreshed call marker = %v, want 2", refreshed.ModelInfo["call"])
	}
	if len(cache.local) != 1 {
		t.Fatalf("local cache entries = %d, want 1", len(cache.local))
	}
}

func TestModelShowCacheLocalVerboseVariantsAreSeparate(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "show-cache-verbose", map[string]any{"test.context_length": uint32(1024)})

	cache := newModelShowCache()
	calls := 0
	cache.getModelInfo = func(req api.ShowRequest) (*api.ShowResponse, error) {
		calls++
		return showCacheTestResponse(calls, req.Verbose), nil
	}

	plain, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-verbose"})
	if err != nil {
		t.Fatalf("plain GetLocal failed: %v", err)
	}
	verbose, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-verbose", Verbose: true})
	if err != nil {
		t.Fatalf("verbose GetLocal failed: %v", err)
	}
	plainAgain, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-verbose"})
	if err != nil {
		t.Fatalf("plain repeat GetLocal failed: %v", err)
	}

	if calls != 2 {
		t.Fatalf("getModelInfo calls = %d, want 2", calls)
	}
	if plain.ModelInfo["verbose"] != false || verbose.ModelInfo["verbose"] != true || plainAgain.ModelInfo["call"] != 1 {
		t.Fatalf("unexpected verbose cache markers: plain=%v verbose=%v plainAgainCall=%v", plain.ModelInfo, verbose.ModelInfo, plainAgain.ModelInfo["call"])
	}
}

func TestModelShowCacheLocalHydrationSkipsUnchangedInMemory(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "show-cache-hydrate", map[string]any{"test.context_length": uint32(1024)})

	cache := newModelShowCache()
	calls := 0
	cache.getModelInfo = func(req api.ShowRequest) (*api.ShowResponse, error) {
		calls++
		return showCacheTestResponse(calls, req.Verbose), nil
	}

	if err := cache.hydrateLocal(context.Background()); err != nil {
		t.Fatalf("first hydrateLocal failed: %v", err)
	}
	if err := cache.hydrateLocal(context.Background()); err != nil {
		t.Fatalf("second hydrateLocal failed: %v", err)
	}
	resp, err := cache.GetLocal(api.ShowRequest{Model: "show-cache-hydrate"})
	if err != nil {
		t.Fatalf("GetLocal after hydration failed: %v", err)
	}

	if calls != 1 {
		t.Fatalf("getModelInfo calls after unchanged in-memory hydration = %d, want 1", calls)
	}
	if resp.ModelInfo["call"] != 1 {
		t.Fatalf("hydrated call marker = %v, want 1", resp.ModelInfo["call"])
	}
}

func TestModelShowCacheBypassesSystemAndOptionsOverlays(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "show-cache-overlay", map[string]any{"test.context_length": uint32(1024)})

	cache := newModelShowCache()
	key, digest, err := modelShowLocalKeyForRequest(api.ShowRequest{Model: "show-cache-overlay"})
	if err != nil {
		t.Fatalf("local key failed: %v", err)
	}
	cache.setLocal(key, digest, &api.ShowResponse{System: "cached", ModelInfo: map[string]any{}})

	s := Server{modelCaches: &modelCaches{show: cache}}
	w := createRequest(t, s.ShowHandler, api.ShowRequest{
		Model:  "show-cache-overlay",
		System: "overlay-system",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}

	var resp api.ShowResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response failed: %v", err)
	}
	if resp.System != "overlay-system" {
		t.Fatalf("system = %q, want overlay-system", resp.System)
	}

	w = createRequest(t, s.ShowHandler, api.ShowRequest{
		Model:   "show-cache-overlay",
		Options: map[string]any{"num_ctx": float64(8192)},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("options overlay status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode options response failed: %v", err)
	}
	if resp.System == "cached" {
		t.Fatalf("options overlay unexpectedly returned cached response")
	}
}

func TestModelShowCacheLocalAndCloudSameBaseDoNotCollide(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "show-cache-dual", map[string]any{"test.context_length": uint32(1024)})

	cache := newModelShowCache()
	cache.getModelInfo = func(req api.ShowRequest) (*api.ShowResponse, error) {
		return &api.ShowResponse{
			Details:   api.ModelDetails{Format: "local"},
			ModelInfo: map[string]any{},
		}, nil
	}
	cloudKey := modelShowCloudKeyForModel("show-cache-dual:cloud", false)
	cache.setCloud(cloudKey, &api.ShowResponse{
		Details:   api.ModelDetails{Format: "cloud"},
		ModelInfo: map[string]any{},
	})
	cache.mu.Lock()
	cache.cloudNextReadRefreshAfter[cloudKey] = time.Now().Add(time.Hour)
	cache.mu.Unlock()

	s := Server{modelCaches: &modelCaches{show: cache}}

	w := createRequest(t, s.ShowHandler, api.ShowRequest{Model: "show-cache-dual:cloud"})
	if w.Code != http.StatusOK {
		t.Fatalf("cloud status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var cloudResp api.ShowResponse
	if err := json.NewDecoder(w.Body).Decode(&cloudResp); err != nil {
		t.Fatalf("decode cloud response failed: %v", err)
	}
	if cloudResp.Details.Format != "cloud" {
		t.Fatalf("cloud format = %q, want cloud", cloudResp.Details.Format)
	}

	w = createRequest(t, s.ShowHandler, api.ShowRequest{Model: "show-cache-dual"})
	if w.Code != http.StatusOK {
		t.Fatalf("local status = %d, want 200: %s", w.Code, w.Body.String())
	}
	var localResp api.ShowResponse
	if err := json.NewDecoder(w.Body).Decode(&localResp); err != nil {
		t.Fatalf("decode local response failed: %v", err)
	}
	if localResp.Details.Format != "local" {
		t.Fatalf("local format = %q, want local", localResp.Details.Format)
	}
}

func TestModelShowCacheCloudWarmHitReturnsStaleAndRefreshes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	refreshDone := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/show" {
			t.Fatalf("unexpected upstream path %q", r.URL.Path)
		}
		defer close(refreshDone)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"details":{"format":"updated"},"model_info":{"source":"fresh"}}`))
	}))
	defer upstream.Close()
	withCloudProxyBaseURL(t, upstream.URL)

	cache := newModelShowCache()
	cache.client = upstream.Client()
	cache.setCloud(modelShowCloudKeyForModel("kimi-k2.5:cloud", false), &api.ShowResponse{
		Details:   api.ModelDetails{Format: "cached"},
		ModelInfo: map[string]any{"source": "stale"},
	})

	s := Server{modelCaches: &modelCaches{show: cache}}
	w := createRequest(t, s.ShowHandler, api.ShowRequest{Model: "kimi-k2.5:cloud"})
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", w.Code, w.Body.String())
	}

	var resp api.ShowResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response failed: %v", err)
	}
	if resp.Details.Format != "cached" {
		t.Fatalf("format = %q, want cached", resp.Details.Format)
	}

	select {
	case <-refreshDone:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for cloud refresh")
	}
	waitForCondition(t, 2*time.Second, func() bool {
		resp, ok := cache.getCloud(modelShowCloudKeyForModel("kimi-k2.5:cloud", false))
		return ok && resp.Details.Format == "updated"
	})
}

func TestModelShowCacheCloudColdMissFallsBackToProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	var capturedPath, capturedBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		body, _ := io.ReadAll(r.Body)
		capturedBody = string(body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"details":{"format":"cold"},"model_info":{}}`))
	}))
	defer upstream.Close()
	withCloudProxyBaseURL(t, upstream.URL)

	s := &Server{modelCaches: &modelCaches{show: newModelShowCache()}}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatalf("GenerateRoutes failed: %v", err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/show", bytes.NewBufferString(`{"model":"kimi-k2.5:cloud"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, string(body))
	}
	if capturedPath != "/api/show" {
		t.Fatalf("upstream path = %q, want /api/show", capturedPath)
	}
	if !strings.Contains(capturedBody, `"model":"kimi-k2.5"`) {
		t.Fatalf("expected normalized model in upstream body, got %q", capturedBody)
	}
}

func TestModelShowCacheCloudHydrationUsesTagsAndShow(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	var mu sync.Mutex
	var showModels []string
	var tagsCalled bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/api/tags":
			if r.Method != http.MethodGet {
				t.Fatalf("tags method = %s, want GET", r.Method)
			}
			mu.Lock()
			tagsCalled = true
			mu.Unlock()
			_, _ = w.Write([]byte(`{"models":[{"name":"alpha:cloud"},{"model":"beta"}]}`))
		case "/api/show":
			var req api.ShowRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode show request failed: %v", err)
			}
			mu.Lock()
			showModels = append(showModels, req.Model)
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(api.ShowResponse{
				Details:   api.ModelDetails{Format: req.Model},
				ModelInfo: map[string]any{"model": req.Model},
			})
		default:
			t.Fatalf("unexpected upstream path %q", r.URL.Path)
		}
	}))
	defer upstream.Close()
	withCloudProxyBaseURL(t, upstream.URL)

	cache := newModelShowCache()
	cache.client = upstream.Client()
	if err := cache.hydrateCloud(context.Background()); err != nil {
		t.Fatalf("hydrateCloud failed: %v", err)
	}

	mu.Lock()
	gotTagsCalled := tagsCalled
	gotShowModels := slices.Clone(showModels)
	mu.Unlock()
	slices.Sort(gotShowModels)

	if !gotTagsCalled {
		t.Fatal("expected /api/tags to be called")
	}
	if !slices.Equal(gotShowModels, []string{"alpha", "beta"}) {
		t.Fatalf("show models = %v, want [alpha beta]", gotShowModels)
	}
	for _, modelName := range gotShowModels {
		resp, ok := cache.getCloud(modelShowCloudKeyForModel(modelName, false))
		if !ok {
			t.Fatalf("missing cached cloud show response for %s", modelName)
		}
		if resp.Details.Format != modelName {
			t.Fatalf("cached format for %s = %q", modelName, resp.Details.Format)
		}
	}
}

func TestModelShowCacheCloudKeyNormalizesSourceTags(t *testing.T) {
	tests := map[string]string{
		" kimi-k2.5:cloud ": "kimi-k2.5",
		"gpt-oss:20b-cloud": "gpt-oss:20b",
		"qwen3":             "qwen3",
	}

	for input, want := range tests {
		if got := modelShowCloudKeyForModel(input, false).Model; got != want {
			t.Fatalf("cloud key model for %q = %q, want %q", input, got, want)
		}
	}
}

func TestModelShowCacheCloudDisabledDoesNotServeStale(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	t.Cleanup(envconfig.ReloadServerConfig)
	t.Setenv("OLLAMA_NO_CLOUD", "1")
	envconfig.ReloadServerConfig()

	cache := newModelShowCache()
	cache.setCloud(modelShowCloudKeyForModel("kimi-k2.5:cloud", false), &api.ShowResponse{
		Details:   api.ModelDetails{Format: "cached"},
		ModelInfo: map[string]any{},
	})

	if err := cache.hydrateCloud(context.Background()); !errors.Is(err, errModelShowNoCloud) {
		t.Fatalf("hydrateCloud error = %v, want %v", err, errModelShowNoCloud)
	}

	s := Server{modelCaches: &modelCaches{show: cache}}
	w := createRequest(t, s.ShowHandler, api.ShowRequest{Model: "kimi-k2.5:cloud"})
	if w.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), internalcloud.DisabledError(cloudErrRemoteModelDetailsUnavailable)) {
		t.Fatalf("unexpected disabled response: %s", w.Body.String())
	}
}

func createShowCacheModel(t *testing.T, name string, kv map[string]any) {
	t.Helper()
	_, digest := createBinFile(t, kv, nil)
	var s Server
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:  name,
		Files:  map[string]string{"model.gguf": digest},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create model status = %d, want 200: %s", w.Code, w.Body.String())
	}
}

func changeShowCacheManifest(t *testing.T, name string) {
	t.Helper()
	n, err := getExistingName(modelpkg.ParseName(name))
	if err != nil {
		t.Fatalf("get existing name: %v", err)
	}
	mf, err := manifest.ParseNamedManifest(n)
	if err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	layer, err := manifest.NewLayer(strings.NewReader("changed"), "application/vnd.ollama.image.system")
	if err != nil {
		t.Fatalf("new layer: %v", err)
	}
	layers := append([]manifest.Layer(nil), mf.Layers...)
	layers = append(layers, layer)
	if err := manifest.WriteManifest(n, mf.Config, layers); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
}

func showCacheTestResponse(call int, verbose bool) *api.ShowResponse {
	return &api.ShowResponse{
		Details: api.ModelDetails{
			Format: "gguf",
			Family: "test",
		},
		Capabilities: []modelpkg.Capability{modelpkg.CapabilityCompletion},
		ModelInfo: map[string]any{
			"call":    call,
			"verbose": verbose,
		},
	}
}

func withCloudProxyBaseURL(t *testing.T, url string) {
	t.Helper()
	original := cloudProxyBaseURL
	cloudProxyBaseURL = url
	t.Cleanup(func() {
		cloudProxyBaseURL = original
	})
}
