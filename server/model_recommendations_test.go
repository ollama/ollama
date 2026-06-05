package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
)

func TestModelRecommendationsDefaultOrder(t *testing.T) {
	want := []string{
		"kimi-k2.6:cloud",
		"glm-5.1:cloud",
		"qwen3.5:cloud",
		"minimax-m2.7:cloud",
		"gemma4",
		"qwen3.5",
	}

	if got := modelRecommendationNames(defaultModelRecommendations); !slices.Equal(got, want) {
		t.Fatalf("recommendations = %v, want %v", got, want)
	}
}

func TestModelRecommendationsCacheRefreshAppliesServerSideChanges(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "")

	first := []api.ModelRecommendation{
		{Model: " first-cloud:cloud ", Description: " first ", ContextLength: 2048, MaxOutputTokens: 512},
		{Model: " first-local ", Description: " first local ", VRAMBytes: 3 * format.GigaByte},
	}
	second := []api.ModelRecommendation{
		{Model: "second-cloud:cloud", Description: "second", ContextLength: 4096, MaxOutputTokens: 1024},
		{Model: "second-local", Description: "second local", VRAMBytes: 6 * format.GigaByte},
	}

	calls := 0
	cache := newModelRecommendationsCache()
	cache.client = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.Method != http.MethodGet {
			t.Fatalf("method = %q, want GET", req.Method)
		}
		if req.URL.String() != modelRecommendationsURL {
			t.Fatalf("url = %q, want %q", req.URL.String(), modelRecommendationsURL)
		}

		calls++
		payload := api.ModelRecommendationsResponse{Recommendations: first}
		if calls > 1 {
			payload.Recommendations = second
		}

		data, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal payload failed: %v", err)
		}
		return jsonHTTPResponse(http.StatusOK, string(data)), nil
	})}

	if err := cache.refresh(context.Background()); err != nil {
		t.Fatalf("first refresh failed: %v", err)
	}
	if got, want := cache.Get(), []api.ModelRecommendation{
		{Model: "first-cloud:cloud", Description: "first", ContextLength: 2048, MaxOutputTokens: 512},
		{Model: "first-local", Description: "first local", VRAMBytes: 3 * format.GigaByte},
	}; !slices.Equal(got, want) {
		t.Fatalf("after first refresh recommendations = %#v, want %#v", got, want)
	}

	if err := cache.refresh(context.Background()); err != nil {
		t.Fatalf("second refresh failed: %v", err)
	}
	if got, want := cache.Get(), second; !slices.Equal(got, want) {
		t.Fatalf("after second refresh recommendations = %#v, want %#v", got, want)
	}

	path, err := modelRecommendationsSnapshotPath()
	if err != nil {
		t.Fatalf("snapshot path failed: %v", err)
	}
	snapshotData, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read snapshot failed: %v", err)
	}
	var snapshot api.ModelRecommendationsResponse
	if err := json.Unmarshal(snapshotData, &snapshot); err != nil {
		t.Fatalf("unmarshal snapshot failed: %v", err)
	}
	if !slices.Equal(snapshot.Recommendations, second) {
		t.Fatalf("snapshot recommendations = %#v, want %#v", snapshot.Recommendations, second)
	}
}

func TestModelRecommendationsCacheRefreshErrorCasesPreserveCurrentData(t *testing.T) {
	cases := []struct {
		name      string
		transport roundTripFunc
		errSubstr string
	}{
		{
			name: "transport error",
			transport: func(*http.Request) (*http.Response, error) {
				return nil, errors.New("network down")
			},
			errSubstr: "network down",
		},
		{
			name: "remote status error",
			transport: func(*http.Request) (*http.Response, error) {
				return jsonHTTPResponse(http.StatusInternalServerError, "upstream broken"), nil
			},
			errSubstr: "status 500: upstream broken",
		},
		{
			name: "invalid json payload",
			transport: func(*http.Request) (*http.Response, error) {
				return jsonHTTPResponse(http.StatusOK, "{"), nil
			},
			errSubstr: "unexpected EOF",
		},
		{
			name: "duplicate recommendations",
			transport: func(*http.Request) (*http.Response, error) {
				return jsonHTTPResponse(http.StatusOK, `{"recommendations":[{"model":"dup","description":"a"},{"model":"dup","description":"b"}]}`), nil
			},
			errSubstr: `duplicate recommendation "dup"`,
		},
		{
			name: "empty recommendations",
			transport: func(*http.Request) (*http.Response, error) {
				return jsonHTTPResponse(http.StatusOK, `{"recommendations":[]}`), nil
			},
			errSubstr: "empty recommendations",
		},
		{
			name: "only invalid cloud recommendations",
			transport: func(*http.Request) (*http.Response, error) {
				return jsonHTTPResponse(http.StatusOK, `{"recommendations":[{"model":"bad:cloud","description":"missing limits"}]}`), nil
			},
			errSubstr: "no valid recommendations",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setupModelRecommendationsTestEnv(t, "")

			cache := newModelRecommendationsCache()
			stable := []api.ModelRecommendation{{Model: "stable-local", Description: "stable desc", VRAMBytes: 2 * format.GigaByte}}
			cache.set(stable)
			cache.client = &http.Client{Transport: tc.transport}

			err := cache.refresh(context.Background())
			if err == nil {
				t.Fatalf("refresh returned nil error")
			}
			if !strings.Contains(err.Error(), tc.errSubstr) {
				t.Fatalf("error = %q, want substring %q", err.Error(), tc.errSubstr)
			}

			if got := cache.Get(); !slices.Equal(got, stable) {
				t.Fatalf("recommendations changed on error: got %#v, want %#v", got, stable)
			}

			path, pathErr := modelRecommendationsSnapshotPath()
			if pathErr != nil {
				t.Fatalf("snapshot path failed: %v", pathErr)
			}
			if _, statErr := os.Stat(path); !errors.Is(statErr, os.ErrNotExist) {
				t.Fatalf("snapshot file should not be written on error, stat err = %v", statErr)
			}
		})
	}
}

func TestModelRecommendationsCacheRefreshNoCloudShortCircuits(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "1")

	called := false
	cache := newModelRecommendationsCache()
	cache.client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		called = true
		return jsonHTTPResponse(http.StatusOK, `{"recommendations":[{"model":"should-not-be-used","description":"n/a"}]}`), nil
	})}

	err := cache.refresh(context.Background())
	if !errors.Is(err, errModelRecommendationsNoCloud) {
		t.Fatalf("refresh error = %v, want %v", err, errModelRecommendationsNoCloud)
	}
	if called {
		t.Fatalf("remote endpoint should not be called when cloud is disabled")
	}
}

func TestModelRecommendationsSnapshotPersistAndLoad(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "")

	want := []api.ModelRecommendation{
		{Model: "persist-cloud:cloud", Description: "persisted", ContextLength: 8192, MaxOutputTokens: 2048},
		{Model: "persist-local", Description: "persisted local", VRAMBytes: 5 * format.GigaByte},
	}

	writer := newModelRecommendationsCache()
	if err := writer.persistSnapshot(want); err != nil {
		t.Fatalf("persistSnapshot failed: %v", err)
	}

	loader := newModelRecommendationsCache()
	loader.set([]api.ModelRecommendation{{Model: "old", Description: "old"}})
	loader.loadSnapshot()

	if got := loader.Get(); !slices.Equal(got, want) {
		t.Fatalf("loaded recommendations = %#v, want %#v", got, want)
	}
}

func TestModelRecommendationsLoadSnapshotInvalidDoesNotOverwrite(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "")

	path, err := modelRecommendationsSnapshotPath()
	if err != nil {
		t.Fatalf("snapshot path failed: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	if err := os.WriteFile(path, []byte("{invalid"), 0o644); err != nil {
		t.Fatalf("write invalid snapshot failed: %v", err)
	}

	cache := newModelRecommendationsCache()
	existing := []api.ModelRecommendation{{Model: "existing", Description: "existing description"}}
	cache.set(existing)
	cache.loadSnapshot()

	if got := cache.Get(); !slices.Equal(got, existing) {
		t.Fatalf("recommendations overwritten by invalid snapshot: got %#v, want %#v", got, existing)
	}
}

func TestValidateModelRecommendationsTrimsAndDropsInvalidCloudEntries(t *testing.T) {
	input := []api.ModelRecommendation{
		{Model: " good-cloud:cloud ", Description: " good cloud ", ContextLength: 1024, MaxOutputTokens: 256, RequiredPlan: " pro "},
		{Model: "bad-cloud:cloud", Description: "missing limits"},
		{Model: " good-local ", Description: " good local ", VRAMBytes: 2 * format.GigaByte},
	}

	got, err := validateModelRecommendations(input)
	if err != nil {
		t.Fatalf("validateModelRecommendations failed: %v", err)
	}

	want := []api.ModelRecommendation{
		{Model: "good-cloud:cloud", Description: "good cloud", ContextLength: 1024, MaxOutputTokens: 256, RequiredPlan: "pro"},
		{Model: "good-local", Description: "good local", VRAMBytes: 2 * format.GigaByte},
	}
	if !slices.Equal(got, want) {
		t.Fatalf("validated recommendations = %#v, want %#v", got, want)
	}
}

func TestValidateModelRecommendationsDoesNotSynthesizeRequiredPlans(t *testing.T) {
	input := []api.ModelRecommendation{
		{Model: "kimi-k2.6:cloud", Description: "coding", ContextLength: 262_144, MaxOutputTokens: 262_144},
		{Model: "qwen3.5:cloud", Description: "reasoning", ContextLength: 262_144, MaxOutputTokens: 32_768},
		{Model: "custom:cloud", Description: "custom", ContextLength: 4096, MaxOutputTokens: 1024},
		{Model: "minimax-m2.7:cloud", Description: "custom", ContextLength: 204_800, MaxOutputTokens: 128_000, RequiredPlan: "team"},
	}

	got, err := validateModelRecommendations(input)
	if err != nil {
		t.Fatalf("validateModelRecommendations failed: %v", err)
	}

	byName := make(map[string]api.ModelRecommendation, len(got))
	for _, rec := range got {
		byName[rec.Model] = rec
	}

	if rec := byName["kimi-k2.6:cloud"]; rec.RequiredPlan != "" {
		t.Fatalf("kimi required plan should not be synthesized: %#v", rec)
	}
	if rec := byName["qwen3.5:cloud"]; rec.RequiredPlan != "" {
		t.Fatalf("qwen required plan should not be synthesized: %#v", rec)
	}
	if rec := byName["custom:cloud"]; rec.RequiredPlan != "" {
		t.Fatalf("custom required plan should not be synthesized: %#v", rec)
	}
	if rec := byName["minimax-m2.7:cloud"]; rec.RequiredPlan != "team" {
		t.Fatalf("explicit required plan should not be overwritten: %#v", rec)
	}
}

func TestModelRecommendationsHandlerReturnsDefaults(t *testing.T) {
	gin.SetMode(gin.TestMode)

	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/api/experimental/model-recommendations", nil)

	s := &Server{}
	s.ModelRecommendationsExperimentalHandler(ctx)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	got := decodeRecommendationNames(t, w)
	want := modelRecommendationNames(defaultModelRecommendations)
	if !slices.Equal(got, want) {
		t.Fatalf("models = %v, want %v", got, want)
	}
}

func TestModelRecommendationsHandlerUsesCache(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setupModelRecommendationsTestEnv(t, "1")

	cache := newModelRecommendationsCache()
	cache.set([]api.ModelRecommendation{{Model: "test-model", Description: "test description"}})

	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/api/experimental/model-recommendations", nil)

	s := &Server{modelCaches: &modelCaches{recommendations: cache}}
	s.ModelRecommendationsExperimentalHandler(ctx)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	got := decodeRecommendationNames(t, w)
	if !slices.Equal(got, []string{"test-model"}) {
		t.Fatalf("models = %v, want %v", got, []string{"test-model"})
	}
	waitForCacheIdle(t, cache)
}

func TestModelRecommendationsRouteRegistration(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setupModelRecommendationsTestEnv(t, "1")

	cache := newModelRecommendationsCache()
	cache.set([]api.ModelRecommendation{{Model: "route-model", Description: "route description"}})
	s := &Server{modelCaches: &modelCaches{recommendations: cache}}

	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatalf("GenerateRoutes failed: %v", err)
	}

	getReq := httptest.NewRequest(http.MethodGet, "/api/experimental/model-recommendations", nil)
	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)
	if getResp.Code != http.StatusOK {
		t.Fatalf("GET status = %d, want %d", getResp.Code, http.StatusOK)
	}
	if got := decodeRecommendationNames(t, getResp); !slices.Equal(got, []string{"route-model"}) {
		t.Fatalf("GET models = %v, want %v", got, []string{"route-model"})
	}

	postReq := httptest.NewRequest(http.MethodPost, "/api/experimental/model-recommendations", nil)
	postResp := httptest.NewRecorder()
	router.ServeHTTP(postResp, postReq)
	if postResp.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST status = %d, want %d", postResp.Code, http.StatusMethodNotAllowed)
	}
	waitForCacheIdle(t, cache)
}

func TestModelRecommendationsGetSWRTriggersRefreshOnRead(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "")

	cache := newModelRecommendationsCache()
	old := []api.ModelRecommendation{{Model: "old", Description: "old"}}
	newRecs := []api.ModelRecommendation{{Model: "new-cloud:cloud", Description: "new", ContextLength: 1024, MaxOutputTokens: 256}}
	cache.set(old)

	refreshDone := make(chan struct{})
	cache.client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		defer close(refreshDone)
		return jsonHTTPResponse(http.StatusOK, `{"recommendations":[{"model":"new-cloud:cloud","description":"new","context_length":1024,"max_output_tokens":256}]}`), nil
	})}

	gotImmediate := cache.GetSWR(context.Background())
	if !slices.Equal(gotImmediate, old) {
		t.Fatalf("GetSWR should return current cache immediately: got %#v, want %#v", gotImmediate, old)
	}

	select {
	case <-refreshDone:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for async refresh")
	}

	waitForCondition(t, 2*time.Second, func() bool {
		return slices.Equal(cache.Get(), newRecs)
	})
	waitForCacheIdle(t, cache)
}

func TestModelRecommendationsGetSWRSkipsWhenRefreshAlreadyInFlight(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "")

	cache := newModelRecommendationsCache()
	cache.set([]api.ModelRecommendation{{Model: "old", Description: "old"}})

	started := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int32

	cache.client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		n := calls.Add(1)
		if n == 1 {
			close(started)
		}
		<-release
		return jsonHTTPResponse(http.StatusOK, `{"recommendations":[{"model":"updated","description":"ok"}]}`), nil
	})}

	cache.GetSWR(context.Background())
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first refresh call")
	}

	for range 5 {
		cache.GetSWR(context.Background())
	}
	time.Sleep(50 * time.Millisecond)
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls during in-flight refresh = %d, want 1", got)
	}

	close(release)
	waitForCacheIdle(t, cache)
}

func TestModelRecommendationsGetSWRThrottlesRefreshAfterCompletion(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "")
	withModelRecommendationsReadRefreshCooldown(t, 100*time.Millisecond)

	cache := newModelRecommendationsCache()
	cache.set([]api.ModelRecommendation{{Model: "old", Description: "old"}})

	started := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int32
	cache.client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		if calls.Add(1) == 1 {
			close(started)
			<-release
		}
		return jsonHTTPResponse(http.StatusOK, `{"recommendations":[{"model":"updated","description":"ok"}]}`), nil
	})}

	cache.GetSWR(context.Background())
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first refresh call")
	}

	time.Sleep(2 * modelRecommendationsReadRefreshCooldown)
	close(release)
	waitForCacheIdle(t, cache)

	cache.GetSWR(context.Background())
	time.Sleep(25 * time.Millisecond)
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls during read refresh cooldown = %d, want 1", got)
	}
}

func TestModelRecommendationsGetSWRRetriesAfterReadRefreshCooldown(t *testing.T) {
	setupModelRecommendationsTestEnv(t, "")
	withModelRecommendationsReadRefreshCooldown(t, 100*time.Millisecond)

	cache := newModelRecommendationsCache()
	old := []api.ModelRecommendation{{Model: "old", Description: "old"}}
	cache.set(old)

	var calls atomic.Int32
	cache.client = &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		if calls.Add(1) == 1 {
			return nil, errors.New("temporary upstream failure")
		}
		return jsonHTTPResponse(http.StatusOK, `{"recommendations":[{"model":"recovered","description":"ok"}]}`), nil
	})}

	cache.GetSWR(context.Background())
	waitForCondition(t, 2*time.Second, func() bool { return calls.Load() >= 1 })
	waitForCacheIdle(t, cache)

	if !slices.Equal(cache.Get(), old) {
		t.Fatalf("cache should remain unchanged after failed refresh, got %#v", cache.Get())
	}

	cache.GetSWR(context.Background())
	time.Sleep(25 * time.Millisecond)
	if got := calls.Load(); got != 1 {
		t.Fatalf("calls during read refresh cooldown after failure = %d, want 1", got)
	}

	waitForCondition(t, 2*time.Second, func() bool {
		cache.GetSWR(context.Background())
		return calls.Load() >= 2
	})
	waitForCondition(t, 2*time.Second, func() bool {
		return slices.Equal(cache.Get(), []api.ModelRecommendation{{Model: "recovered", Description: "ok"}})
	})
	waitForCacheIdle(t, cache)
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func jsonHTTPResponse(statusCode int, body string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func setupModelRecommendationsTestEnv(t *testing.T, noCloudEnv string) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("HOMEDRIVE", filepath.VolumeName(home))
	t.Setenv("HOMEPATH", strings.TrimPrefix(home, filepath.VolumeName(home)))

	// Use explicit false rather than empty to avoid platform/env ambiguity.
	if noCloudEnv == "" {
		noCloudEnv = "false"
	}
	t.Setenv("OLLAMA_NO_CLOUD", noCloudEnv)
	envconfig.ReloadServerConfig()
	t.Cleanup(envconfig.ReloadServerConfig)
}

func withModelRecommendationsReadRefreshCooldown(t *testing.T, d time.Duration) {
	t.Helper()
	old := modelRecommendationsReadRefreshCooldown
	modelRecommendationsReadRefreshCooldown = d
	t.Cleanup(func() {
		modelRecommendationsReadRefreshCooldown = old
	})
}

func waitForCondition(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("timed out waiting for condition")
}

func waitForCacheIdle(t *testing.T, cache *modelRecommendationsCache) {
	t.Helper()
	waitForCondition(t, 2*time.Second, func() bool {
		cache.mu.RLock()
		refreshing := cache.refreshing
		cache.mu.RUnlock()
		return !refreshing
	})
}

func decodeRecommendationNames(t *testing.T, w *httptest.ResponseRecorder) []string {
	t.Helper()

	var resp struct {
		Recommendations []struct {
			Model string `json:"model"`
		} `json:"recommendations"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	names := make([]string, 0, len(resp.Recommendations))
	for _, rec := range resp.Recommendations {
		names = append(names, rec.Model)
	}
	return names
}

func modelRecommendationNames(recs []api.ModelRecommendation) []string {
	names := make([]string, len(recs))
	for i, rec := range recs {
		names[i] = rec.Model
	}
	return names
}
