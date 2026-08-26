//go:build darwin

package main

import (
	"context"
	"errors"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/proxy"
)

func TestMain(m *testing.M) {
	previousLoader := claudeModelsLoader
	previousAccessResolver := claudeAccessStateResolver
	previousLocalResolver := claudeLocalModelsResolver
	previousCloudResolver := claudeCloudModelsResolver
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		return proxy.DefaultClaudeDesktopModels(), "fallback"
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{
			Cloud:   proxy.ClaudeDesktopCloudOn,
			Account: proxy.ClaudeDesktopAccountSignedIn,
			Plan:    "pro",
		}, nil
	}
	claudeLocalModelsResolver = func(context.Context) ([]string, error) { return nil, nil }
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) { return nil, nil }
	code := m.Run()
	claudeModelsLoader = previousLoader
	claudeAccessStateResolver = previousAccessResolver
	claudeLocalModelsResolver = previousLocalResolver
	claudeCloudModelsResolver = previousCloudResolver
	os.Exit(code)
}

func TestLoadClaudeDesktopModelsUsesAppEndpoint(t *testing.T) {
	useTestOllamaRequestSigner(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/experimental/model-recommendations" || r.URL.Query().Get("app") != "claude-desktop" {
			t.Fatalf("request URL = %q", r.URL.String())
		}
		if r.URL.Query().Get("ts") == "" {
			t.Fatal("model recommendations request is missing its signed timestamp")
		}
		if r.Header.Get("Authorization") != "test-public-key:test-signature" {
			t.Fatal("model recommendations request is missing public-key identity")
		}
		_, _ = w.Write([]byte(`{"recommendations":[{"model":"glm-5.2:cloud","description":"Cloud model","max_output_tokens":262144,"required_plan":"pro"},{"model":"gemma4:26b","description":"Local model","max_output_tokens":262144}]}`))
	}))
	defer server.Close()

	previousClient := claudeRecommendationsClient
	previousEndpoint := claudeRecommendationsEndpoint
	claudeRecommendationsClient = server.Client()
	claudeRecommendationsEndpoint = func() string {
		return server.URL + "/api/experimental/model-recommendations?app=claude-desktop"
	}
	t.Cleanup(func() {
		claudeRecommendationsClient = previousClient
		claudeRecommendationsEndpoint = previousEndpoint
	})

	models, source := loadClaudeDesktopModels(context.Background())
	if source != "endpoint" || len(models) != 1 || models[0].Name != "glm-5.2:cloud" {
		t.Fatalf("models/source = %+v/%q", models, source)
	}
	if !models[0].Recommended {
		t.Fatal("endpoint recommendation was not marked as recommended")
	}
}

func TestCurrentClaudeDesktopCloudModelsUsesAccountEndpoint(t *testing.T) {
	useTestOllamaRequestSigner(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" || r.URL.Query().Get("ts") == "" {
			t.Fatalf("request URL = %q", r.URL.String())
		}
		if r.Header.Get("Authorization") != "test-public-key:test-signature" {
			t.Fatal("cloud model request is missing public-key identity")
		}
		_, _ = w.Write([]byte(`{"models":[{"name":"glm-5.2:cloud"},{"name":"qwen3:8b"}]}`))
	}))
	defer server.Close()

	previousClient := claudeCloudModelsClient
	previousEndpoint := claudeCloudModelsEndpoint
	claudeCloudModelsClient = server.Client()
	claudeCloudModelsEndpoint = func() string { return server.URL + "/api/tags" }
	t.Cleanup(func() {
		claudeCloudModelsClient = previousClient
		claudeCloudModelsEndpoint = previousEndpoint
	})

	models, err := currentClaudeDesktopCloudModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].Name != "glm-5.2:cloud" || models[1].Name != "qwen3:8b:cloud" || models[0].Recommended {
		t.Fatalf("cloud models = %+v, want normalized non-recommended account models", models)
	}
	access := proxy.EvaluateClaudeDesktopModelAccess(models[0], proxy.ClaudeDesktopAccessState{
		Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn,
	}, false, true)
	if access.Availability != proxy.ClaudeDesktopAvailabilityAvailable {
		t.Fatalf("cloud model access = %+v, want available", access)
	}
}

func TestClaudeDesktopDownloadEndpointUsesTypedZipContract(t *testing.T) {
	useTestOllamaRequestSigner(t)
	req, err := newSignedOllamaRequest(context.Background(), http.MethodGet, claudeDesktopDownloadEndpoint("http://127.0.0.1:18080/"))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := req.URL.Path, "/download-app"; got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	if query := req.URL.Query(); query.Get("app") != "claude-desktop" || query.Get("type") != "mac-zip" || query.Get("ts") == "" {
		t.Fatalf("query = %q", req.URL.RawQuery)
	}
	if req.Header.Get("Authorization") != "test-public-key:test-signature" {
		t.Fatal("Claude Desktop download request is missing public-key identity")
	}
}

func TestClaudeEndpointRequestsSignCompleteRequestURI(t *testing.T) {
	previousSigner := signOllamaData
	t.Cleanup(func() {
		signOllamaData = previousSigner
	})

	for _, endpoint := range []string{
		"https://ollama.com/api/experimental/model-recommendations?app=claude-desktop",
		"https://ollama.com/api/tags",
		"https://ollama.com/download-app?app=claude-desktop&type=mac-zip",
	} {
		var challenge string
		signOllamaData = func(_ context.Context, data []byte) (string, error) {
			challenge = string(data)
			return "test-public-key:test-signature", nil
		}
		req, err := newSignedOllamaRequest(context.Background(), http.MethodGet, endpoint)
		if err != nil {
			t.Fatal(err)
		}
		if want := http.MethodGet + "," + req.URL.RequestURI(); challenge != want {
			t.Fatalf("challenge = %q, want %q", challenge, want)
		}
	}
}

func TestClaudeEndpointRequestsAreNotSentUnsigned(t *testing.T) {
	previousSigner := signOllamaData
	signOllamaData = func(context.Context, []byte) (string, error) {
		return "", errors.New("signing unavailable")
	}
	t.Cleanup(func() {
		signOllamaData = previousSigner
	})

	called := false
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		called = true
	}))
	defer server.Close()
	previousClient := claudeRecommendationsClient
	previousEndpoint := claudeRecommendationsEndpoint
	claudeRecommendationsClient = server.Client()
	claudeRecommendationsEndpoint = func() string {
		return server.URL + "/api/experimental/model-recommendations?app=claude-desktop"
	}
	t.Cleanup(func() {
		claudeRecommendationsClient = previousClient
		claudeRecommendationsEndpoint = previousEndpoint
	})

	models, source := loadClaudeDesktopModels(context.Background())
	if source != "fallback" {
		t.Fatalf("model source = %q, want fallback", source)
	}
	for _, model := range models {
		if model.Recommended {
			t.Fatalf("offline fallback model %q must not enable Auto mode", model.Name)
		}
	}
	if called {
		t.Fatal("model recommendations request was sent without identity")
	}
	if _, err := newSignedOllamaRequest(context.Background(), http.MethodGet, claudeDesktopDownloadEndpoint(server.URL)); err == nil {
		t.Fatal("Claude Desktop download request succeeded without identity")
	}
}

func useTestOllamaRequestSigner(t *testing.T) {
	t.Helper()
	previous := signOllamaData
	signOllamaData = func(context.Context, []byte) (string, error) {
		return "test-public-key:test-signature", nil
	}
	t.Cleanup(func() {
		signOllamaData = previous
	})
}

func TestResolveClaudeDesktopCatalogUsesPersistedSelection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	available, selected, source := resolveClaudeDesktopCatalog(context.Background())
	if source != "fallback" || len(selected) != len(available) {
		t.Fatalf("default catalog = %d/%d source %q, want all fallback models selected", len(selected), len(available), source)
	}

	if err := launch.SaveClaudeDesktopModels([]string{"qwen3:8b"}); err != nil {
		t.Fatal(err)
	}
	available, selected, source = resolveClaudeDesktopCatalog(context.Background())
	if source != "user" {
		t.Fatalf("source = %q, want user", source)
	}
	if len(selected) != 1 || selected[0].Name != "qwen3:8b" {
		t.Fatalf("selected models = %+v, want persisted qwen3:8b", selected)
	}
	if got := available[len(available)-1].Name; got != "qwen3:8b" {
		t.Fatalf("last available model = %q, want persisted qwen3:8b", got)
	}
}

func TestResolveClaudeDesktopStartupCatalogVerifiesPersistedAccountCloudModel(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopModels([]string{"custom:cloud"}); err != nil {
		t.Fatal(err)
	}

	previousResolver := claudeCloudModelsResolver
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"custom:cloud"}), nil
	}
	t.Cleanup(func() { claudeCloudModelsResolver = previousResolver })

	_, selected, source := resolveClaudeDesktopStartupCatalog(context.Background())
	if source != "user" || len(selected) != 1 || selected[0].OllamaModel != "custom:cloud" {
		t.Fatalf("selected/source = %+v/%q", selected, source)
	}
	access := proxy.EvaluateClaudeDesktopModelAccess(selected[0], proxy.ClaudeDesktopAccessState{
		Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn,
	}, false, true)
	if access.Availability != proxy.ClaudeDesktopAvailabilityAvailable {
		t.Fatalf("persisted cloud model access = %+v, want available", access)
	}
}

func TestResolveClaudeDesktopStartupCatalogPreservesPersistedRouteMappings(t *testing.T) {
	tests := []struct {
		name     string
		mappings map[string]string
		state    proxy.ClaudeDesktopAccessState
		local    []string
		cloud    []string
	}{
		{
			name:     "free sparse mapping",
			mappings: map[string]string{"claude-sonnet-5": "gemma4:31b-cloud"},
			state:    proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
		},
		{
			name:     "one model shared by every route",
			mappings: sharedClaudeDesktopMappings("gemma4:31b-cloud"),
			state:    proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
		},
		{
			name: "mixed local and cloud mapping",
			mappings: map[string]string{
				"claude-fable-5": "qwen3:8b",
				"claude-opus-5":  "glm-5.2:cloud",
			},
			state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"},
			local: []string{"qwen3:8b"},
			cloud: []string{"glm-5.2:cloud"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOME", t.TempDir())
			if err := launch.SaveClaudeDesktopModelMappings(tt.mappings); err != nil {
				t.Fatal(err)
			}

			previousAccess := claudeAccessStateResolver
			previousLocal := claudeLocalModelsResolver
			previousCloud := claudeCloudModelsResolver
			claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
				return tt.state, nil
			}
			claudeLocalModelsResolver = func(context.Context) ([]string, error) {
				return tt.local, nil
			}
			claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
				return proxy.ClaudeDesktopModelsFromCloudInventory(tt.cloud), nil
			}
			t.Cleanup(func() {
				claudeAccessStateResolver = previousAccess
				claudeLocalModelsResolver = previousLocal
				claudeCloudModelsResolver = previousCloud
			})

			_, selected, source := resolveClaudeDesktopStartupCatalog(context.Background())
			if source != "user" {
				t.Fatalf("source = %q, want user", source)
			}
			if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, tt.mappings) {
				t.Fatalf("startup mappings = %v, want persisted routes %v", got, tt.mappings)
			}
		})
	}
}

func TestResolveClaudeDesktopStartupCatalogMarksDefaultAccountModelsAutoEligible(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousLoader := claudeModelsLoader
	previousResolver := claudeCloudModelsResolver
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		return proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "glm-5.2:cloud"}}), "endpoint"
	}
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"glm-5.2"}), nil
	}
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeCloudModelsResolver = previousResolver
	})

	available, selected, source := resolveClaudeDesktopStartupCatalog(context.Background())
	if source != "endpoint" || len(available) != 1 || len(selected) != 1 {
		t.Fatalf("catalog = %+v selected = %+v source = %q", available, selected, source)
	}
	if !claudeDesktopModelsSupportAutoMode(selected) {
		t.Fatal("default account cloud model was not Auto-eligible")
	}
}

func TestResolveClaudeDesktopStartupCatalogUsesAccountDefaults(t *testing.T) {
	states := []struct {
		name  string
		state proxy.ClaudeDesktopAccessState
	}{
		{name: "signed out", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}},
		{name: "free", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}},
		{name: "pro", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"}},
		{name: "team", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"}},
	}
	for _, tt := range states {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOME", t.TempDir())
			previousAccess := claudeAccessStateResolver
			claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
				return tt.state, nil
			}
			t.Cleanup(func() { claudeAccessStateResolver = previousAccess })

			_, selected, source := resolveClaudeDesktopStartupCatalog(context.Background())
			want := proxy.DefaultClaudeDesktopMappings(
				claudeDesktopHasFullDefaultAccess(tt.state),
			)
			if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, want) {
				t.Fatalf("startup mappings = %v, want %v (source %q)", got, want, source)
			}
		})
	}
}

func TestResolveClaudeDesktopStartupCatalogVerifiesFallbackFromAccountInventory(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopModels([]string{"glm-5.2:cloud"}); err != nil {
		t.Fatal(err)
	}

	previousLoader := claudeModelsLoader
	previousResolver := claudeCloudModelsResolver
	previousAccess := claudeAccessStateResolver
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		return fallbackClaudeDesktopModels(), "fallback"
	}
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"glm-5.2"}), nil
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn}, nil
	}
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeCloudModelsResolver = previousResolver
		claudeAccessStateResolver = previousAccess
	})

	available, selected, source := resolveClaudeDesktopStartupCatalog(context.Background())
	if source != "user" || len(available) == 0 || len(selected) != 1 {
		t.Fatalf("catalog = %+v selected = %+v source = %q", available, selected, source)
	}
	if selected[0].OllamaModel != "glm-5.2:cloud" || !selected[0].AccountCloud {
		t.Fatalf("selected model = %+v, want verified GLM route", selected[0])
	}
	access := proxy.EvaluateClaudeDesktopModelAccess(selected[0], proxy.ClaudeDesktopAccessState{
		Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro",
	}, false, true)
	if access.Availability != proxy.ClaudeDesktopAvailabilityAvailable {
		t.Fatalf("fallback account model access = %+v, want available", access)
	}
	if !claudeDesktopModelsSupportAutoMode(selected) {
		t.Fatal("verified fallback account model was not Auto-eligible")
	}
}

func TestResolveClaudeDesktopStartupCatalogDoesNotListCloudWhenOff(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopModels([]string{"custom:cloud"}); err != nil {
		t.Fatal(err)
	}

	previousAccess := claudeAccessStateResolver
	previousResolver := claudeCloudModelsResolver
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, nil
	}
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		t.Fatal("account cloud list called while Cloud was off")
		return nil, nil
	}
	t.Cleanup(func() {
		claudeAccessStateResolver = previousAccess
		claudeCloudModelsResolver = previousResolver
	})

	_, selected, _ := resolveClaudeDesktopStartupCatalog(context.Background())
	if len(selected) != 1 || selected[0].OllamaModel != "custom:cloud" {
		t.Fatalf("selected = %+v", selected)
	}
	access := proxy.EvaluateClaudeDesktopModelAccess(selected[0], proxy.ClaudeDesktopAccessState{
		Cloud: proxy.ClaudeDesktopCloudOff,
	}, false, true)
	if access.Reason != proxy.ClaudeDesktopAccessCloudOff {
		t.Fatalf("cloud-off access = %+v", access)
	}
}

func TestRefreshClaudeDesktopCatalogUpdatesPolicyAndPreservesSlots(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousLoader := claudeModelsLoader
	previousInterval := claudeCatalogRefreshInterval
	previousNow := claudeCatalogNow
	claudeProxyMu.Lock()
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAvailableModels = nil
	claudeModelSource = ""
	claudeCatalogUpdated = time.Time{}
	claudeProxyMu.Unlock()
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeCatalogRefreshInterval = previousInterval
		claudeCatalogNow = previousNow
		claudeProxyMu.Lock()
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	now := time.Unix(1_000, 0)
	claudeCatalogNow = func() time.Time { return now }
	claudeCatalogRefreshInterval = time.Minute
	load := 0
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		load++
		requiredPlan := "free"
		if load > 1 {
			requiredPlan = "pro"
		}
		return proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "changing-model:cloud", RequiredPlan: requiredPlan}}), "endpoint"
	}

	_, initial, _ := refreshClaudeDesktopCatalog(context.Background(), nil, false)
	if len(initial) != 1 {
		t.Fatalf("initial selection = %+v", initial)
	}
	initialID := initial[0].GatewayID()
	now = now.Add(2 * time.Minute)
	_, updated, _ := refreshClaudeDesktopCatalog(context.Background(), initial, false)
	if load != 2 {
		t.Fatalf("catalog loads = %d, want 2", load)
	}
	if len(updated) != 1 || updated[0].RequiredPlan != "pro" {
		t.Fatalf("updated selection = %+v, want Pro metadata", updated)
	}
	if updated[0].GatewayID() != initialID {
		t.Fatalf("gateway ID changed from %q to %q", initialID, updated[0].GatewayID())
	}
}

func TestRefreshClaudeDesktopCatalogFailsClosedForRemovedSelection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousLoader := claudeModelsLoader
	previousInterval := claudeCatalogRefreshInterval
	previousNow := claudeCatalogNow
	claudeProxyMu.Lock()
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAvailableModels = nil
	claudeModelSource = ""
	claudeCatalogUpdated = time.Time{}
	claudeProxyMu.Unlock()
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeCatalogRefreshInterval = previousInterval
		claudeCatalogNow = previousNow
		claudeProxyMu.Lock()
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	now := time.Unix(2_000, 0)
	claudeCatalogNow = func() time.Time { return now }
	claudeCatalogRefreshInterval = time.Minute
	load := 0
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		load++
		name := "retired-model:cloud"
		if load > 1 {
			name = "replacement-model:cloud"
		}
		return proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: name, RequiredPlan: "free"}}), "endpoint"
	}

	_, initial, _ := refreshClaudeDesktopCatalog(context.Background(), nil, false)
	now = now.Add(2 * time.Minute)
	_, updated, _ := refreshClaudeDesktopCatalog(context.Background(), initial, false)
	if len(updated) != 1 || updated[0].Name != "retired-model:cloud" {
		t.Fatalf("updated selection = %+v, want preserved retired selection", updated)
	}
	state := proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}
	access := proxy.EvaluateClaudeDesktopModelAccess(updated[0], state, false, true)
	if access.Reason != proxy.ClaudeDesktopAccessVerificationUnavailable {
		t.Fatalf("removed-model access = %+v, want verification unavailable", access)
	}
}

func TestRefreshClaudeDesktopCatalogDoesNotDowngradeEntitlementOnFallback(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousLoader := claudeModelsLoader
	previousInterval := claudeCatalogRefreshInterval
	previousNow := claudeCatalogNow
	claudeProxyMu.Lock()
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAvailableModels = nil
	claudeModelSource = ""
	claudeCatalogUpdated = time.Time{}
	claudeProxyMu.Unlock()
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeCatalogRefreshInterval = previousInterval
		claudeCatalogNow = previousNow
		claudeProxyMu.Lock()
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	now := time.Unix(3_000, 0)
	claudeCatalogNow = func() time.Time { return now }
	claudeCatalogRefreshInterval = time.Minute
	load := 0
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		load++
		if load == 1 {
			return proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "gemma4:31b-cloud", RequiredPlan: "pro"}}), "endpoint"
		}
		return proxy.DefaultClaudeDesktopModels(), "fallback"
	}

	_, initial, _ := refreshClaudeDesktopCatalog(context.Background(), nil, false)
	now = now.Add(2 * time.Minute)
	_, updated, _ := refreshClaudeDesktopCatalog(context.Background(), initial, false)
	if len(updated) != 1 || updated[0].RequiredPlan != "pro" {
		t.Fatalf("fallback selection = %+v, want last-known Pro requirement", updated)
	}
	state := proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}
	access := proxy.EvaluateClaudeDesktopModelAccess(updated[0], state, false, true)
	if access.Reason != proxy.ClaudeDesktopAccessUpgradeRequired {
		t.Fatalf("fallback access = %+v, want upgrade required", access)
	}
}

func TestIncludeSelectedClaudeDesktopModelsKeepsCustomModels(t *testing.T) {
	available := proxy.DefaultClaudeDesktopModels()
	selected := proxy.SelectClaudeDesktopModels(available, []string{"qwen3:8b"})
	models := includeSelectedClaudeDesktopModels(available, selected)
	if got := models[len(models)-1].Name; got != "qwen3:8b" {
		t.Fatalf("last available model = %q, want qwen3:8b", got)
	}
}

func TestSelectKnownClaudeDesktopModelsAllowsInstalledModelsOnly(t *testing.T) {
	available := proxy.DefaultClaudeDesktopModels()
	selected, err := selectKnownClaudeDesktopModels(available, nil, []string{"qwen3:8b"}, []string{"qwen3:8b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 1 || selected[0].Name != "qwen3:8b" {
		t.Fatalf("selected models = %+v, want installed qwen3:8b", selected)
	}

	selected, err = selectKnownClaudeDesktopModels(available, nil, nil, []string{"deepseek-v4-flash:0731:cloud"})
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 1 || selected[0].Name != "deepseek-v4-flash" || selected[0].OllamaModel != "deepseek-v4-flash:0731:cloud" {
		t.Fatalf("selected cloud model = %+v", selected)
	}

	if _, err := selectKnownClaudeDesktopModels(available, nil, []string{"qwen3:8b"}, []string{"made-up-model"}); err == nil {
		t.Fatal("expected an arbitrary model name to be rejected")
	}

	accountCloud := proxy.ClaudeDesktopModelsFromCloudInventory([]string{"custom:cloud"})
	selected, err = selectKnownClaudeDesktopModels(append(available, accountCloud...), nil, nil, []string{"custom:cloud"})
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 1 || selected[0].OllamaModel != "custom:cloud" || selected[0].Recommended {
		t.Fatalf("account cloud selection = %+v", selected)
	}

	// The five-model selection cap must not make later installed models
	// unselectable. The cap applies to the final selection, not the inventory.
	localNames := []string{"local-1", "local-2", "local-3", "local-4", "local-5", "local-6"}
	selected, err = selectKnownClaudeDesktopModels(available, nil, localNames, []string{"local-6"})
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 1 || selected[0].Name != "local-6" {
		t.Fatalf("later installed model selection = %+v, want local-6", selected)
	}
}

func sharedClaudeDesktopMappings(model string) map[string]string {
	mappings := make(map[string]string, proxy.MaxClaudeDesktopModels)
	for _, route := range proxy.ClaudeDesktopRoutes() {
		mappings[route.ID] = model
	}
	return mappings
}

func TestMapKnownClaudeDesktopModelsAllowsSharedModels(t *testing.T) {
	selected, err := mapKnownClaudeDesktopModels(
		proxy.DefaultClaudeDesktopModels(),
		nil,
		[]string{"qwen3:8b"},
		sharedClaudeDesktopMappings("qwen3:8b"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != proxy.MaxClaudeDesktopModels {
		t.Fatalf("selected models = %+v", selected)
	}
	for _, model := range selected {
		if model.OllamaModel != "qwen3:8b" {
			t.Fatalf("selected model = %+v", model)
		}
	}
	sparse, err := mapKnownClaudeDesktopModels(proxy.DefaultClaudeDesktopModels(), nil, nil, map[string]string{"claude-fable-5": "glm-5.2:cloud"})
	if err != nil {
		t.Fatal(err)
	}
	if len(sparse) != 1 || sparse[0].GatewayID() != "claude-fable-5" {
		t.Fatalf("sparse mapping = %+v", sparse)
	}
}

func TestClaudeDesktopDefaultsFollowAccountPlan(t *testing.T) {
	paidDefaults := proxy.DefaultClaudeDesktopMappings(true)
	restrictedDefaults := proxy.DefaultClaudeDesktopMappings(false)
	tests := []struct {
		name         string
		state        proxy.ClaudeDesktopAccessState
		wantMappings map[string]string
	}{
		{name: "signed out", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedOut}, wantMappings: restrictedDefaults},
		{name: "free", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}, wantMappings: restrictedDefaults},
		{name: "Pro", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"}, wantMappings: paidDefaults},
		{name: "Team", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"}, wantMappings: paidDefaults},
		{name: "future paid plan", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "enterprise"}, wantMappings: paidDefaults},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := proxy.DefaultClaudeDesktopMappingsForModels(
				proxy.DefaultClaudeDesktopModels(),
				claudeDesktopHasFullDefaultAccess(tt.state),
			)
			if !maps.Equal(got, tt.wantMappings) {
				t.Fatalf("default mappings = %v, want %v", got, tt.wantMappings)
			}
		})
	}
}

type fakeClaudeDesktopController struct {
	configured       bool
	profileCurrent   bool
	configureCalls   int
	configureErr     error
	running          bool
	opened           bool
	installed        bool
	restart          bool
	setErr           error
	openErr          error
	modelsAtSet      []string
	configureOnSet   bool
	requireRestart   bool
	autoMode         bool
	restoreCalls     int
	restoreConfirmed bool
}

func (f *fakeClaudeDesktopController) UsesOllamaGateway() bool { return f.configured }
func (f *fakeClaudeDesktopController) Running() bool           { return f.running }
func (f *fakeClaudeDesktopController) Open() error {
	f.opened = true
	f.running = true
	if f.openErr != nil {
		return f.openErr
	}
	return f.setErr
}

func (f *fakeClaudeDesktopController) AutodiscoveryConfiguredWithAutoMode(autoMode bool) bool {
	return f.configured && f.profileCurrent && f.autoMode == autoMode
}

func (f *fakeClaudeDesktopController) ConfigureAutodiscoveryWithAutoMode(autoMode bool) error {
	f.configureCalls++
	if f.configureErr != nil {
		f.profileCurrent = false
		return f.configureErr
	}
	f.configured = true
	f.profileCurrent = true
	f.autoMode = autoMode
	return nil
}

func (f *fakeClaudeDesktopController) SetInstalledFromDesktopWithAutoMode(installed, restart, autoMode bool) error {
	if f.requireRestart && !restart {
		return errors.New("Claude Desktop restart confirmation is required before changing its profile")
	}
	f.installed = installed
	f.restart = restart
	f.autoMode = autoMode
	f.modelsAtSet = launch.ClaudeDesktopModels()
	if f.configureOnSet {
		f.configured = installed
	}
	return f.setErr
}

func TestSetClaudeDesktopConnectionForwardsRestartConfirmation(t *testing.T) {
	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	claudeDesktopInstalled = func() bool { return true }
	fake := &fakeClaudeDesktopController{configured: true, requireRestart: true}
	claudeDesktop = fake
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
	})

	if err := setClaudeDesktopConnection(false, false); err == nil || !strings.Contains(err.Error(), "restart confirmation is required") {
		t.Fatalf("unconfirmed connection change error = %v, want restart confirmation error", err)
	}
	if err := setClaudeDesktopConnection(false, true); err != nil {
		t.Fatalf("confirmed connection change error = %v", err)
	}
	if !fake.restart {
		t.Fatal("confirmed restart was not forwarded to the Claude controller")
	}
}

func (f *fakeClaudeDesktopController) ApplyProfileChange(change func() error, restartConfirmed bool) error {
	if f.running && !restartConfirmed {
		return launch.ErrClaudeDesktopRestartConfirmationRequired
	}
	if err := change(); err != nil {
		return err
	}
	f.installed = true
	f.restart = f.running
	f.modelsAtSet = launch.ClaudeDesktopModels()
	return f.setErr
}

func (f *fakeClaudeDesktopController) RestoreForShutdownWithConfirmation(_ context.Context, quitConfirmed bool) error {
	f.restoreCalls++
	f.restoreConfirmed = quitConfirmed
	if f.running && !quitConfirmed {
		return launch.ErrClaudeDesktopQuitConfirmationRequired
	}
	f.configured = false
	f.profileCurrent = false
	f.installed = false
	return nil
}

func TestUpdateClaudeDesktopProfileRepairsExistingConnectionOnce(t *testing.T) {
	previousDesktop := claudeDesktop
	previousRunning := claudeDesktopRunning
	fake := &fakeClaudeDesktopController{configured: true}
	claudeDesktop = fake
	claudeDesktopRunning = func() bool { return false }
	t.Cleanup(func() {
		claudeDesktop = previousDesktop
		claudeDesktopRunning = previousRunning
	})

	if err := updateClaudeDesktopProfile(); err != nil {
		t.Fatal(err)
	}
	if !fake.configured || !fake.profileCurrent || fake.configureCalls != 1 {
		t.Fatalf("updated profile = %+v, want one repair preserving the connection", fake)
	}

	if err := updateClaudeDesktopProfile(); err != nil {
		t.Fatal(err)
	}
	if fake.configureCalls != 1 {
		t.Fatalf("configure calls = %d, want idempotent repair", fake.configureCalls)
	}
}

func TestUpdateClaudeDesktopProfileDefersRepairWhileClaudeRuns(t *testing.T) {
	previousDesktop := claudeDesktop
	previousRunning := claudeDesktopRunning
	fake := &fakeClaudeDesktopController{configured: true}
	claudeDesktop = fake
	claudeDesktopRunning = func() bool { return true }
	t.Cleanup(func() {
		claudeDesktop = previousDesktop
		claudeDesktopRunning = previousRunning
	})

	if err := updateClaudeDesktopProfile(); err != nil {
		t.Fatal(err)
	}
	if fake.configureCalls != 0 || fake.profileCurrent {
		t.Fatalf("startup reconciliation changed a live Claude profile: %+v", fake)
	}
}

func TestSetClaudeDesktopAutoModePersistsUntilConnection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousDesktop := claudeDesktop
	previousRunning := claudeDesktopRunning
	fake := &fakeClaudeDesktopController{}
	claudeDesktop = fake
	claudeDesktopRunning = func() bool {
		t.Fatal("disconnected preference change should not inspect the Claude process")
		return false
	}
	t.Cleanup(func() {
		claudeDesktop = previousDesktop
		claudeDesktopRunning = previousRunning
	})

	if err := setClaudeDesktopAutoMode(false, true); err != nil {
		t.Fatal(err)
	}
	enabled, err := launch.ClaudeDesktopAutoModeEnabled()
	if err != nil {
		t.Fatal(err)
	}
	if enabled || fake.configureCalls != 0 {
		t.Fatalf("enabled = %v, configure calls = %d, want saved preference without profile write", enabled, fake.configureCalls)
	}
}

func TestSetClaudeDesktopAutoModeAvoidsUnnecessaryRestart(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopAutoMode(true); err != nil {
		t.Fatal(err)
	}
	previousAvailable := claudeAvailableModels
	claudeAvailableModels = mergeClaudeDesktopCloudInventory(
		proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "glm-5.2:cloud"}}),
		proxy.ClaudeDesktopModelsFromCloudInventory([]string{"glm-5.2:cloud"}),
		false,
	)
	previousDesktop := claudeDesktop
	previousRunning := claudeDesktopRunning
	fake := &fakeClaudeDesktopController{configured: true, profileCurrent: true, autoMode: true}
	claudeDesktop = fake
	claudeDesktopRunning = func() bool {
		t.Fatal("unchanged current preference should not inspect the Claude process")
		return false
	}
	t.Cleanup(func() {
		claudeAvailableModels = previousAvailable
		claudeDesktop = previousDesktop
		claudeDesktopRunning = previousRunning
	})

	if err := setClaudeDesktopAutoMode(true, true); err != nil {
		t.Fatal(err)
	}
	if fake.configureCalls != 0 || fake.restart {
		t.Fatalf("unchanged preference triggered profile lifecycle: %+v", fake)
	}
}

func TestSetClaudeDesktopAutoModeRewritesProfileBeforeRestart(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousDesktop := claudeDesktop
	previousRunning := claudeDesktopRunning
	fake := &fakeClaudeDesktopController{configured: true, profileCurrent: true, running: true}
	claudeDesktop = fake
	claudeDesktopRunning = func() bool { return true }
	t.Cleanup(func() {
		claudeDesktop = previousDesktop
		claudeDesktopRunning = previousRunning
	})

	if err := setClaudeDesktopAutoMode(false, true); err != nil {
		t.Fatal(err)
	}
	if fake.configureCalls != 1 || !fake.profileCurrent || !fake.restart {
		t.Fatalf("profile restart lifecycle = %+v, want one profile write followed by restart", fake)
	}
}

func TestSetClaudeDesktopAutoModeCancelDoesNotSavePreference(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopAutoMode(true); err != nil {
		t.Fatal(err)
	}
	previousDesktop := claudeDesktop
	fake := &fakeClaudeDesktopController{
		configured: true, profileCurrent: true, running: true, autoMode: true,
	}
	claudeDesktop = fake
	t.Cleanup(func() { claudeDesktop = previousDesktop })

	err := setClaudeDesktopAutoMode(false, false)
	if !errors.Is(err, launch.ErrClaudeDesktopRestartConfirmationRequired) {
		t.Fatalf("error = %v, want restart confirmation", err)
	}
	enabled, loadErr := launch.ClaudeDesktopAutoModeEnabled()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if !enabled || fake.configureCalls != 0 || fake.restart {
		t.Fatalf("canceled Auto mode changed preference/profile: enabled=%v fake=%+v", enabled, fake)
	}
}

func TestSetClaudeDesktopAutoModeKeepsDesiredPreferenceAfterProfileFailure(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousDesktop := claudeDesktop
	previousRunning := claudeDesktopRunning
	fake := &fakeClaudeDesktopController{configured: true, profileCurrent: true, configureErr: errors.New("profile write failed")}
	claudeDesktop = fake
	claudeDesktopRunning = func() bool { return false }
	t.Cleanup(func() {
		claudeDesktop = previousDesktop
		claudeDesktopRunning = previousRunning
	})

	err := setClaudeDesktopAutoMode(false, true)
	if err == nil || !strings.Contains(err.Error(), "profile write failed") {
		t.Fatalf("setClaudeDesktopAutoMode error = %v, want profile write failure", err)
	}
	enabled, loadErr := launch.ClaudeDesktopAutoModeEnabled()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if enabled {
		t.Fatal("desired preference should remain disabled so reconciliation can retry")
	}
	if fake.profileCurrent {
		t.Fatal("failed profile write should remain visibly out of date")
	}
}

func TestClaudeDesktopAutoModeModelEligibility(t *testing.T) {
	recommended := proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "glm-5.2:cloud"},
		{Model: "kimi-k3:cloud"},
		{Model: "gemma4:31b-cloud"},
	})
	accountCloud := proxy.ClaudeDesktopModelsFromCloudInventory([]string{
		"glm-5.2:cloud",
		"gemma4:31b-cloud",
	})
	recommended = mergeClaudeDesktopCloudInventory(recommended, accountCloud, false)
	custom := proxy.SelectClaudeDesktopModels(nil, []string{"qwen3:8b"})
	tagOnly := proxy.SelectClaudeDesktopModels(nil, []string{"made-up:cloud"})

	tests := []struct {
		name   string
		models []proxy.ClaudeDesktopModel
		want   bool
	}{
		{name: "account cloud model", models: recommended[:1], want: true},
		{name: "recommendation absent from account list", models: recommended[1:2]},
		{name: "account gemma4 model", models: recommended[2:], want: true},
		{name: "custom model excluded", models: custom},
		{name: "cloud suffix alone excluded", models: tagOnly},
		{name: "account cloud and custom", models: []proxy.ClaudeDesktopModel{recommended[0], custom[0]}, want: true},
		{name: "empty selection excluded"},
		{name: "offline fallback excluded", models: fallbackClaudeDesktopModels()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := claudeDesktopModelsSupportAutoMode(tt.models); got != tt.want {
				t.Fatalf("claudeDesktopModelsSupportAutoMode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSetClaudeDesktopAutoModeRejectsUnsupportedSelection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopAutoMode(false); err != nil {
		t.Fatal(err)
	}

	previousAvailable := claudeAvailableModels
	claudeAvailableModels = proxy.SelectClaudeDesktopModels(nil, []string{"qwen3:8b"})
	t.Cleanup(func() { claudeAvailableModels = previousAvailable })

	err := setClaudeDesktopAutoMode(true, true)
	if err == nil || !strings.Contains(err.Error(), "cloud model available to your Ollama.com account") {
		t.Fatalf("setClaudeDesktopAutoMode() error = %v", err)
	}
	enabled, loadErr := launch.ClaudeDesktopAutoModeEnabled()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if enabled {
		t.Fatal("rejected Auto mode change modified the saved preference")
	}
}

func TestApplyClaudeDesktopMappingsPersistsSelection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	previousInstalled := claudeDesktopInstalled
	previousAddr := claudeProxyListenAddr
	previousDesktop := claudeDesktop
	previousCloudResolver := claudeCloudModelsResolver
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = "127.0.0.1:0"
	fake := &fakeClaudeDesktopController{configured: true}
	claudeDesktop = fake
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"kimi-k3:cloud"}), nil
	}
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
		claudeDesktop = previousDesktop
		claudeCloudModelsResolver = previousCloudResolver
	})

	if _, err := applyClaudeDesktopMappings(sharedClaudeDesktopMappings("kimi-k3:cloud"), true); err != nil {
		t.Fatal(err)
	}
	if !fake.installed {
		t.Fatal("expected the Claude profile to be installed")
	}
	if !fake.autoMode {
		t.Fatal("expected the account cloud model to keep Auto mode enabled")
	}
	if got, want := launch.ClaudeDesktopModels(), []string{"kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud"}; !slices.Equal(got, want) {
		t.Fatalf("persisted models = %v, want Ollama routes %v", got, want)
	}
}

func TestApplyClaudeDesktopMappingsRequiresLiveRestartConfirmation(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	previousCloudResolver := claudeCloudModelsResolver
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	claudeAppProxy = nil
	claudeProxyMu.Unlock()
	claudeDesktopInstalled = func() bool { return true }
	fake := &fakeClaudeDesktopController{configured: true, running: true}
	claudeDesktop = fake
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"kimi-k3:cloud"}), nil
	}
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeCloudModelsResolver = previousCloudResolver
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeProxyMu.Unlock()
	})

	applied, err := applyClaudeDesktopMappings(sharedClaudeDesktopMappings("kimi-k3:cloud"), false)
	if applied || !errors.Is(err, launch.ErrClaudeDesktopRestartConfirmationRequired) {
		t.Fatalf("applied/error = %v/%v, want live restart confirmation", applied, err)
	}
	if len(launch.ClaudeDesktopModelMappings()) != 0 || fake.restart || fake.opened {
		t.Fatalf("unconfirmed apply changed state: mappings=%v fake=%+v", launch.ClaudeDesktopModelMappings(), fake)
	}
}

func TestApplyClaudeDesktopMappingsDoesNotRestartWithoutChanges(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	want := sharedClaudeDesktopMappings("kimi-k3:cloud")
	if err := launch.SaveClaudeDesktopModelMappings(want); err != nil {
		t.Fatal(err)
	}

	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	previousCloudResolver := claudeCloudModelsResolver
	claudeDesktopInstalled = func() bool { return true }
	fake := &fakeClaudeDesktopController{configured: true, running: true}
	claudeDesktop = fake
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"kimi-k3:cloud"}), nil
	}
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeCloudModelsResolver = previousCloudResolver
	})

	applied, err := applyClaudeDesktopMappings(want, false)
	if err != nil || applied {
		t.Fatalf("unchanged apply = %v/%v, want no-op", applied, err)
	}
	if fake.restart || fake.opened || fake.configureCalls != 0 {
		t.Fatalf("unchanged mappings affected Claude: %+v", fake)
	}
}

func TestApplyClaudeDesktopMappingsRollsBackWhenRestartFails(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopModels([]string{"glm-5.2:cloud"}); err != nil {
		t.Fatal(err)
	}
	models := proxy.SelectClaudeDesktopModels(proxy.DefaultClaudeDesktopModels(), []string{"glm-5.2:cloud"})
	gateway, err := proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  "http://127.0.0.1:11434",
		Model:      models[0].OllamaModel,
		Models:     models,
	})
	if err != nil {
		t.Fatal(err)
	}

	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAppProxy = gateway
	claudeAvailableModels = proxy.DefaultClaudeDesktopModels()
	claudeModelSource = "endpoint"
	claudeCatalogUpdated = claudeCatalogNow()
	claudeProxyMu.Unlock()
	claudeDesktopInstalled = func() bool { return true }
	fake := &fakeClaudeDesktopController{configured: true, setErr: errors.New("restart failed")}
	claudeDesktop = fake
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	_, err = applyClaudeDesktopMappings(sharedClaudeDesktopMappings("kimi-k3:cloud"), true)
	if err == nil || !strings.Contains(err.Error(), "restart failed") {
		t.Fatalf("restart error = %v", err)
	}
	if got := launch.ClaudeDesktopModels(); !slices.Equal(got, []string{"glm-5.2:cloud"}) {
		t.Fatalf("persisted models after failure = %v", got)
	}
	if !slices.Equal(fake.modelsAtSet, []string{"kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud"}) {
		t.Fatalf("models visible before restart = %v, want new selection", fake.modelsAtSet)
	}
	gotModels := gateway.Models()
	if len(gotModels) != 1 || gotModels[0].OllamaModel != "glm-5.2:cloud" {
		t.Fatalf("live gateway models after failure = %+v", gotModels)
	}
}

func TestApplyClaudeDesktopMappingsStartsClaudeWhenStopped(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if err := launch.SaveClaudeDesktopModels([]string{"glm-5.2:cloud"}); err != nil {
		t.Fatal(err)
	}
	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	previousAddr := claudeProxyListenAddr
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAppProxy = nil
	claudeAvailableModels = nil
	claudeModelSource = ""
	claudeCatalogUpdated = time.Time{}
	claudeProxyMu.Unlock()
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = "127.0.0.1:0"
	fake := &fakeClaudeDesktopController{}
	claudeDesktop = fake
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeProxyListenAddr = previousAddr
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	_, err := applyClaudeDesktopMappings(sharedClaudeDesktopMappings("kimi-k3:cloud"), true)
	if err != nil {
		t.Fatal(err)
	}
	if !fake.configured || !fake.installed || !fake.opened || fake.restart {
		t.Fatalf("stopped Claude action = %+v, want configured and opened without restart", fake)
	}
	if got := launch.ClaudeDesktopModels(); !slices.Equal(got, []string{"kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud", "kimi-k3:cloud"}) {
		t.Fatalf("persisted models after open failure = %v", got)
	}
	claudeProxyMu.Lock()
	gateway := claudeAppProxy
	claudeProxyMu.Unlock()
	if gateway == nil {
		t.Fatal("starting Claude must start the gateway")
	}
}

func TestResetClaudeDesktopMappingsDoesNotOpenStoppedClaude(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousStore := appStore
	appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	if err := markClaudeDesktopIntegrationUsed(); err != nil {
		t.Fatal(err)
	}
	if err := launch.SaveClaudeDesktopModels([]string{"glm-5.2:cloud"}); err != nil {
		t.Fatal(err)
	}
	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	previousAddr := claudeProxyListenAddr
	previousLoader := claudeModelsLoader
	previousAccess := claudeAccessStateResolver
	previousLocal := claudeLocalModelsResolver
	previousCloud := claudeCloudModelsResolver
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAppProxy = nil
	claudeAvailableModels = nil
	claudeModelSource = ""
	claudeCatalogUpdated = time.Time{}
	claudeProxyMu.Unlock()
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = "127.0.0.1:0"
	fake := &fakeClaudeDesktopController{configured: true}
	claudeDesktop = fake
	plan := "team"
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		return proxy.DefaultClaudeDesktopModels(), "endpoint"
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{
			Cloud:   proxy.ClaudeDesktopCloudOn,
			Account: proxy.ClaudeDesktopAccountSignedIn,
			Plan:    plan,
		}, nil
	}
	claudeLocalModelsResolver = func(context.Context) ([]string, error) { return nil, nil }
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) { return nil, nil }
	t.Cleanup(func() {
		stopClaudeAppProxy()
		_ = appStore.Close()
		appStore = previousStore
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeProxyListenAddr = previousAddr
		claudeModelsLoader = previousLoader
		claudeAccessStateResolver = previousAccess
		claudeLocalModelsResolver = previousLocal
		claudeCloudModelsResolver = previousCloud
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	paidMappings := proxy.DefaultClaudeDesktopMappings(true)
	applied, err := resetClaudeDesktopMappings(false)
	if err != nil || !applied {
		t.Fatalf("reset mappings = %v/%v, want persisted change", applied, err)
	}
	if !fake.configured || !fake.installed || fake.opened || fake.restart {
		t.Fatalf("stopped Claude reset = %+v, want configured without open or restart", fake)
	}
	if got := launch.ClaudeDesktopModelMappings(); !maps.Equal(got, paidMappings) {
		t.Fatalf("persisted reset mappings = %v", got)
	}

	stopClaudeAppProxy()
	fake.configured = false
	fake.installed = false
	fake.configureCalls = 0
	plan = "free"
	disconnectedMappings := proxy.DefaultClaudeDesktopMappings(false)
	applied, err = resetClaudeDesktopMappings(false)
	if err != nil || !applied {
		t.Fatalf("disconnected reset mappings = %v/%v, want persisted change", applied, err)
	}
	if fake.configured || fake.installed || fake.opened || fake.restart || fake.configureCalls != 0 {
		t.Fatalf("disconnected Claude reset = %+v, want no connection side effects", fake)
	}
	claudeProxyMu.Lock()
	gateway := claudeAppProxy
	claudeProxyMu.Unlock()
	if gateway != nil {
		t.Fatal("resetting disconnected Claude must not start the gateway")
	}
	if got := launch.ClaudeDesktopModelMappings(); !maps.Equal(got, disconnectedMappings) {
		t.Fatalf("persisted disconnected reset mappings = %v", got)
	}
}

func TestResetClaudeDesktopMappingsSerializesDisconnectDuringCatalogRefresh(t *testing.T) {
	testResetClaudeDesktopMappingsSerializesLifecycleChange(t, "disconnect", func() error {
		return setClaudeDesktopConnection(false, false)
	})
}

func TestResetClaudeDesktopMappingsSerializesShutdownDuringCatalogRefresh(t *testing.T) {
	testResetClaudeDesktopMappingsSerializesLifecycleChange(t, "shutdown", func() error {
		return restoreClaudeAppForTermination(context.Background(), false, true)
	})
}

func testResetClaudeDesktopMappingsSerializesLifecycleChange(t *testing.T, name string, change func() error) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	previousMappings := sharedClaudeDesktopMappings("glm-5.2:cloud")
	if err := launch.SaveClaudeDesktopModelMappings(previousMappings); err != nil {
		t.Fatal(err)
	}

	previousStore := appStore
	appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	if err := markClaudeDesktopIntegrationUsed(); err != nil {
		t.Fatal(err)
	}

	current := proxy.MapClaudeDesktopModels(proxy.DefaultClaudeDesktopModels(), previousMappings)
	gateway, err := proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		ListenAddr: "127.0.0.1:0",
		OllamaURL:  "http://127.0.0.1:11434",
		Model:      current[0].OllamaModel,
		Models:     current,
	})
	if err != nil {
		t.Fatal(err)
	}

	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	previousLoader := claudeModelsLoader
	previousAccess := claudeAccessStateResolver
	previousLocal := claudeLocalModelsResolver
	previousCloud := claudeCloudModelsResolver
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAppProxy = gateway
	claudeAvailableModels = proxy.DefaultClaudeDesktopModels()
	claudeModelSource = "endpoint"
	claudeCatalogUpdated = claudeCatalogNow()
	claudeProxyMu.Unlock()

	claudeDesktopInstalled = func() bool { return true }
	fake := &fakeClaudeDesktopController{configured: true, configureOnSet: true}
	claudeDesktop = fake
	refreshStarted := make(chan struct{})
	releaseRefresh := make(chan struct{})
	loaderCalls := 0
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		loaderCalls++
		if loaderCalls == 2 {
			close(refreshStarted)
			<-releaseRefresh
		}
		return proxy.DefaultClaudeDesktopModels(), "endpoint"
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{
			Cloud:   proxy.ClaudeDesktopCloudOn,
			Account: proxy.ClaudeDesktopAccountSignedIn,
			Plan:    "team",
		}, nil
	}
	claudeLocalModelsResolver = func(context.Context) ([]string, error) { return nil, nil }
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) { return nil, nil }
	t.Cleanup(func() {
		stopClaudeAppProxy()
		_ = appStore.Close()
		appStore = previousStore
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeModelsLoader = previousLoader
		claudeAccessStateResolver = previousAccess
		claudeLocalModelsResolver = previousLocal
		claudeCloudModelsResolver = previousCloud
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	type resetResult struct {
		applied bool
		err     error
	}
	resetDone := make(chan resetResult, 1)
	go func() {
		applied, err := resetClaudeDesktopMappings(false)
		resetDone <- resetResult{applied: applied, err: err}
	}()

	select {
	case <-refreshStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("reset did not pause during the forced catalog refresh")
	}

	changeStarted := make(chan struct{})
	changeDone := make(chan error, 1)
	go func() {
		close(changeStarted)
		changeDone <- change()
	}()
	<-changeStarted

	completedEarly := false
	select {
	case <-changeDone:
		completedEarly = true
	case <-time.After(50 * time.Millisecond):
	}
	close(releaseRefresh)

	var reset resetResult
	select {
	case reset = <-resetDone:
	case <-time.After(2 * time.Second):
		t.Fatal("reset did not finish after the catalog refresh resumed")
	}
	if reset.err != nil || !reset.applied {
		t.Fatalf("reset mappings = %v/%v, want a persisted reset", reset.applied, reset.err)
	}
	if completedEarly {
		t.Fatalf("%s completed while reset still owned the Claude profile lifecycle", name)
	}
	select {
	case err := <-changeDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("%s did not finish after reset released the lifecycle", name)
	}

	if fake.configured || fake.installed {
		t.Fatalf("Claude was reconnected after %s: %+v", name, fake)
	}
	if name == "shutdown" && fake.restoreCalls != 1 {
		t.Fatalf("shutdown restores = %d, want 1", fake.restoreCalls)
	}
	claudeProxyMu.Lock()
	activeGateway := claudeAppProxy
	claudeProxyMu.Unlock()
	if activeGateway != nil {
		t.Fatalf("Claude gateway remained active after %s", name)
	}
}

func TestApplyClaudeDesktopMappingsKeepsCommittedMappingsWhenOpenFails(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	previousAddr := claudeProxyListenAddr
	previousCloudResolver := claudeCloudModelsResolver
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	claudeAppProxy = nil
	claudeProxyMu.Unlock()
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = "127.0.0.1:0"
	fake := &fakeClaudeDesktopController{openErr: errors.New("launch failed")}
	claudeDesktop = fake
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"kimi-k3:cloud"}), nil
	}
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeProxyListenAddr = previousAddr
		claudeCloudModelsResolver = previousCloudResolver
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeProxyMu.Unlock()
	})

	applied, err := applyClaudeDesktopMappings(sharedClaudeDesktopMappings("kimi-k3:cloud"), false)
	if !applied || err == nil || !strings.Contains(err.Error(), "were saved") {
		t.Fatalf("open failure = %v/%v, want committed mappings and launch error", applied, err)
	}
	if got := launch.ClaudeDesktopModelMappings(); !maps.Equal(got, sharedClaudeDesktopMappings("kimi-k3:cloud")) {
		t.Fatalf("saved mappings after open failure = %v", got)
	}
}

func TestApplyClaudeDesktopMappingsRejectsUnknownRoute(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	claudeDesktopInstalled = func() bool { return true }
	fake := &fakeClaudeDesktopController{configured: true}
	claudeDesktop = fake
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
	})

	_, err := applyClaudeDesktopMappings(map[string]string{
		"not-a-claude-route": "glm-5.2:cloud",
	}, true)
	if err == nil || !strings.Contains(err.Error(), "unknown Claude Desktop route") {
		t.Fatalf("error = %v, want an unknown route message", err)
	}
	if fake.installed {
		t.Fatal("the Claude profile must not change when the selection exceeds the model limit")
	}
}

func TestApplyClaudeDesktopMappingsRejectsEmptyMapping(t *testing.T) {
	previousInstalled := claudeDesktopInstalled
	claudeDesktopInstalled = func() bool { return true }
	t.Cleanup(func() { claudeDesktopInstalled = previousInstalled })

	_, err := applyClaudeDesktopMappings(nil, true)
	if err == nil || !strings.Contains(err.Error(), "at least one Claude Desktop route") {
		t.Fatalf("error = %v, want an empty mapping message", err)
	}
}

func TestClaudeDesktopIntegrationHistoryPersists(t *testing.T) {
	previousStore := appStore
	appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	t.Cleanup(func() {
		_ = appStore.Close()
		appStore = previousStore
	})

	if hasUsedClaudeDesktopIntegration() {
		t.Fatal("expected no Claude Desktop integration history initially")
	}
	if err := markClaudeDesktopIntegrationUsed(); err != nil {
		t.Fatal(err)
	}
	if !hasUsedClaudeDesktopIntegration() {
		t.Fatal("expected Claude Desktop integration history after marking it used")
	}
}

func TestPrepareClaudeDesktopConnectionPreservesFirstUseIntro(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("OLLAMA_HOST", "127.0.0.1:11434")
	if err := launch.SaveClaudeDesktopModels([]string{"qwen3:8b"}); err != nil {
		t.Fatal(err)
	}

	previousStore := appStore
	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	previousAddr := claudeProxyListenAddr
	previousLoader := claudeModelsLoader
	previousAccess := claudeAccessStateResolver
	previousLocal := claudeLocalModelsResolver
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAppProxy = nil
	claudeProxyMu.Unlock()

	appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	claudeDesktopInstalled = func() bool { return true }
	claudeDesktop = &fakeClaudeDesktopController{}
	claudeProxyListenAddr = "127.0.0.1:0"
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		t.Fatal("first-use preparation loaded model recommendations")
		return nil, ""
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		t.Fatal("first-use preparation resolved cloud access")
		return proxy.ClaudeDesktopAccessState{}, nil
	}
	localCalls := 0
	claudeLocalModelsResolver = func(context.Context) ([]string, error) {
		localCalls++
		return []string{"qwen3:8b"}, nil
	}
	t.Cleanup(func() {
		stopClaudeAppProxy()
		_ = appStore.Close()
		appStore = previousStore
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeProxyListenAddr = previousAddr
		claudeModelsLoader = previousLoader
		claudeAccessStateResolver = previousAccess
		claudeLocalModelsResolver = previousLocal
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		clearClaudeProxyFailure()
		claudeProxyMu.Unlock()
	})

	if err := prepareClaudeDesktopConnection(); err != nil {
		t.Fatal(err)
	}
	preparationLocalCalls := localCalls
	prepared := getClaudeDesktopConnectionSummary()
	if !prepared.Configured || !prepared.Connected || prepared.Used {
		t.Fatalf("prepared status = %+v, want connected with first-use intro still eligible", prepared)
	}
	if localCalls != preparationLocalCalls {
		t.Fatalf("local model lookups after status = %d, want %d", localCalls, preparationLocalCalls)
	}

	if err := setClaudeDesktopConnection(true, false); err != nil {
		t.Fatal(err)
	}
	continued := getClaudeDesktopConnectionSummary()
	if !continued.Used {
		t.Fatalf("continued status = %+v, want integration marked used", continued)
	}
	if localCalls != preparationLocalCalls {
		t.Fatalf("local model lookups after Continue = %d, want %d", localCalls, preparationLocalCalls)
	}
}

func TestLoadClaudeDesktopModelsFallsBackWithoutMLX(t *testing.T) {
	useTestOllamaRequestSigner(t)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	previousClient := claudeRecommendationsClient
	previousEndpoint := claudeRecommendationsEndpoint
	claudeRecommendationsClient = server.Client()
	claudeRecommendationsEndpoint = func() string { return server.URL }
	t.Cleanup(func() {
		claudeRecommendationsClient = previousClient
		claudeRecommendationsEndpoint = previousEndpoint
	})

	models, source := loadClaudeDesktopModels(context.Background())
	if source != "fallback" || len(models) != 5 {
		t.Fatalf("models/source = %+v/%q", models, source)
	}
	for _, model := range models {
		if strings.Contains(strings.ToLower(model.Name), "mlx") {
			t.Fatalf("fallback contains MLX model %q", model.Name)
		}
		access := proxy.EvaluateClaudeDesktopModelAccess(
			model,
			proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"},
			false,
			true,
		)
		if access.Reason != proxy.ClaudeDesktopAccessVerificationUnavailable {
			t.Fatalf("cold fallback access for %q = %+v, want verification unavailable", model.Name, access)
		}
	}
}

func TestResolveClaudeDesktopAccessState(t *testing.T) {
	tests := []struct {
		name        string
		cloudStatus func(context.Context) (*api.StatusResponse, error)
		whoami      func(context.Context) (*api.UserResponse, error)
		want        proxy.ClaudeDesktopAccessState
	}{
		{
			name: "signed in",
			whoami: func(context.Context) (*api.UserResponse, error) {
				return &api.UserResponse{Name: "parth", Plan: "pro"}, nil
			},
			want: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"},
		},
		{
			name: "signed out",
			whoami: func(context.Context) (*api.UserResponse, error) {
				return nil, api.AuthorizationError{StatusCode: http.StatusUnauthorized}
			},
			want: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut},
		},
		{
			name: "empty account",
			whoami: func(context.Context) (*api.UserResponse, error) {
				return &api.UserResponse{}, nil
			},
			want: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut},
		},
		{
			name: "cloud disabled",
			cloudStatus: func(context.Context) (*api.StatusResponse, error) {
				return &api.StatusResponse{Cloud: api.CloudStatus{Disabled: true}}, nil
			},
			whoami: func(context.Context) (*api.UserResponse, error) {
				t.Fatal("whoami called while cloud was disabled")
				return nil, nil
			},
			want: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff, Account: proxy.ClaudeDesktopAccountUnknown},
		},
		{
			name: "cloud status unavailable",
			cloudStatus: func(context.Context) (*api.StatusResponse, error) {
				return nil, errors.New("status unavailable")
			},
			whoami: func(context.Context) (*api.UserResponse, error) {
				return &api.UserResponse{Name: "parth"}, nil
			},
			want: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudUnknown, Account: proxy.ClaudeDesktopAccountUnknown},
		},
		{
			name: "account check unavailable",
			whoami: func(context.Context) (*api.UserResponse, error) {
				return nil, errors.New("account service unavailable")
			},
			want: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountUnknown},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cloudStatus := tt.cloudStatus
			if cloudStatus == nil {
				cloudStatus = func(context.Context) (*api.StatusResponse, error) {
					return &api.StatusResponse{}, nil
				}
			}
			got, _ := resolveClaudeDesktopAccessState(context.Background(), cloudStatus, tt.whoami)
			if got != tt.want {
				t.Fatalf("resolveClaudeDesktopAccessState() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestValidateClaudeDesktopModels(t *testing.T) {
	local := proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "qwen3:8b"}})
	free := proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "gemma4:31b-cloud", RequiredPlan: "free"}})
	pro := proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "glm-5.2:cloud", RequiredPlan: "pro"}})

	tests := []struct {
		name           string
		models         []proxy.ClaudeDesktopModel
		state          proxy.ClaudeDesktopAccessState
		localNames     []string
		inventoryKnown bool
		wantError      string
	}{
		{
			name:           "installed local works with cloud off",
			models:         local,
			state:          proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff},
			localNames:     []string{"qwen3:8b"},
			inventoryKnown: true,
		},
		{
			name:           "free cloud model works for free account",
			models:         free,
			state:          proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
			inventoryKnown: true,
		},
		{
			name:           "cloud disabled by configuration or environment",
			models:         free,
			state:          proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff},
			inventoryKnown: true,
			wantError:      "Cloud models are off",
		},
		{
			name:           "signed out",
			models:         free,
			state:          proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut},
			inventoryKnown: true,
			wantError:      "Sign in to Ollama",
		},
		{
			name:           "plan upgrade required",
			models:         pro,
			state:          proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
			inventoryKnown: true,
			wantError:      "Select another model in Settings",
		},
		{
			name:           "local model missing",
			models:         local,
			inventoryKnown: true,
			wantError:      "Install the selected model",
		},
		{
			name:      "access unavailable",
			models:    free,
			wantError: "couldn't verify",
		},
		{
			name:           "empty selection",
			inventoryKnown: true,
			wantError:      "Choose at least one model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateClaudeDesktopModels(tt.models, tt.state, tt.localNames, tt.inventoryKnown)
			if tt.wantError == "" {
				if err != nil {
					t.Fatalf("validateClaudeDesktopModels() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("validateClaudeDesktopModels() error = %v, want %q", err, tt.wantError)
			}
		})
	}
}

func TestEnsureClaudeDesktopModelsAvailableRetriesStartupRace(t *testing.T) {
	previousAccessResolver := claudeAccessStateResolver
	previousLocalResolver := claudeLocalModelsResolver
	previousRetryWait := claudeAccessRetryWait
	previousRetryPoll := claudeAccessRetryPoll
	claudeAccessRetryWait = time.Second
	claudeAccessRetryPoll = time.Millisecond
	calls := 0
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		calls++
		if calls == 1 {
			return proxy.ClaudeDesktopAccessState{}, errors.New("server starting")
		}
		return proxy.ClaudeDesktopAccessState{
			Cloud:   proxy.ClaudeDesktopCloudOn,
			Account: proxy.ClaudeDesktopAccountSignedIn,
			Plan:    "free",
		}, nil
	}
	claudeLocalModelsResolver = func(context.Context) ([]string, error) { return nil, nil }
	t.Cleanup(func() {
		claudeAccessStateResolver = previousAccessResolver
		claudeLocalModelsResolver = previousLocalResolver
		claudeAccessRetryWait = previousRetryWait
		claudeAccessRetryPoll = previousRetryPoll
	})

	models := proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{
		Model:        "gemma4:31b-cloud",
		RequiredPlan: "free",
	}})
	if err := ensureClaudeDesktopModelsAvailable(context.Background(), models); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("access checks = %d, want 2", calls)
	}
}

func TestResolveClaudeDesktopDefaultMappingsHandlesAccountVerificationRestartRace(t *testing.T) {
	previousLoader := claudeModelsLoader
	previousAccess := claudeAccessStateResolver
	previousRetryWait := claudeAccessRetryWait
	previousRetryPoll := claudeAccessRetryPoll
	claudeAccessRetryWait = 10 * time.Millisecond
	claudeAccessRetryPoll = time.Millisecond
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeAccessStateResolver = previousAccess
		claudeAccessRetryWait = previousRetryWait
		claudeAccessRetryPoll = previousRetryPoll
	})

	tests := []struct {
		name             string
		accessStateAfter int
		plan             string
		catalog          func() []proxy.ClaudeDesktopModel
		wantDefaults     map[string]string
	}{
		{
			name:             "retry restores paid defaults",
			accessStateAfter: 2,
			plan:             "team",
			wantDefaults:     proxy.DefaultClaudeDesktopMappings(true),
		},
		{
			name:             "free account restores only the free default",
			accessStateAfter: 1,
			plan:             "free",
			wantDefaults:     proxy.DefaultClaudeDesktopMappings(false),
		},
		{
			name:             "incomplete paid catalog does not clear routes",
			accessStateAfter: 1,
			plan:             "team",
			catalog: func() []proxy.ClaudeDesktopModel {
				return proxy.DefaultClaudeDesktopModels()[:4]
			},
		},
		{
			name:             "missing free default does not clear routes",
			accessStateAfter: 1,
			plan:             "free",
			catalog: func() []proxy.ClaudeDesktopModel {
				return proxy.DefaultClaudeDesktopModels()[:4]
			},
		},
		{
			name: "persistent failure does not synthesize free defaults",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
				if tt.catalog != nil {
					return tt.catalog(), "endpoint"
				}
				return proxy.DefaultClaudeDesktopModels(), "endpoint"
			}
			accessCalls := 0
			claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
				accessCalls++
				if tt.accessStateAfter == 0 || accessCalls < tt.accessStateAfter {
					return proxy.ClaudeDesktopAccessState{}, errors.New("server restarting")
				}
				return proxy.ClaudeDesktopAccessState{
					Cloud:   proxy.ClaudeDesktopCloudOn,
					Account: proxy.ClaudeDesktopAccountSignedIn,
					Plan:    tt.plan,
				}, nil
			}

			gotDefaults, err := resolveClaudeDesktopDefaultMappings(context.Background())
			if tt.wantDefaults == nil {
				if err == nil {
					t.Fatalf("reset defaults = %v, want an error", gotDefaults)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !maps.Equal(gotDefaults, tt.wantDefaults) {
				t.Fatalf("reset defaults = %v, want %v after %d access checks", gotDefaults, tt.wantDefaults, accessCalls)
			}
			if tt.accessStateAfter > 0 && accessCalls < tt.accessStateAfter {
				t.Fatalf("access checks = %d, want at least %d", accessCalls, tt.accessStateAfter)
			}
		})
	}
}

func TestClaudeDesktopDefaultAccessTierRequiresVerifiedAccount(t *testing.T) {
	tests := []struct {
		name      string
		state     proxy.ClaudeDesktopAccessState
		wantFull  bool
		wantKnown bool
	}{
		{name: "cloud off", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}},
		{name: "signed out", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}},
		{name: "missing plan", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn}},
		{name: "free", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}, wantKnown: true},
		{name: "team", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"}, wantFull: true, wantKnown: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			full, known := claudeDesktopDefaultAccessTier(tt.state)
			if full != tt.wantFull || known != tt.wantKnown {
				t.Fatalf("default access tier = %v/%v, want %v/%v", full, known, tt.wantFull, tt.wantKnown)
			}
		})
	}
}

func TestSetClaudeGatewayInstalledRejectsEmptyUsableCatalog(t *testing.T) {
	tests := []struct {
		name      string
		state     proxy.ClaudeDesktopAccessState
		selection []string
		wantError string
	}{
		{
			name: "signed out",
			state: proxy.ClaudeDesktopAccessState{
				Cloud:   proxy.ClaudeDesktopCloudOn,
				Account: proxy.ClaudeDesktopAccountSignedOut,
			},
			wantError: "Sign in to Ollama",
		},
		{
			name:      "OLLAMA_NO_CLOUD or Cloud setting disabled",
			state:     proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff},
			wantError: "Cloud models are off",
		},
		{
			name: "free account with only Pro selected",
			state: proxy.ClaudeDesktopAccessState{
				Cloud:   proxy.ClaudeDesktopCloudOn,
				Account: proxy.ClaudeDesktopAccountSignedIn,
				Plan:    "free",
			},
			selection: []string{"glm-5.2:cloud"},
			wantError: "Select another model in Settings",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOME", t.TempDir())
			previousStore := appStore
			appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
			if len(tt.selection) > 0 {
				if err := launch.SaveClaudeDesktopModels(tt.selection); err != nil {
					t.Fatal(err)
				}
			}

			previousInstalled := claudeDesktopInstalled
			previousAddr := claudeProxyListenAddr
			previousDesktop := claudeDesktop
			previousAccessResolver := claudeAccessStateResolver
			previousLocalResolver := claudeLocalModelsResolver
			claudeDesktopInstalled = func() bool { return true }
			claudeProxyListenAddr = "127.0.0.1:0"
			fake := &fakeClaudeDesktopController{}
			claudeDesktop = fake
			claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
				return tt.state, nil
			}
			claudeLocalModelsResolver = func(context.Context) ([]string, error) { return nil, nil }
			t.Cleanup(func() {
				stopClaudeAppProxy()
				_ = appStore.Close()
				appStore = previousStore
				claudeDesktopInstalled = previousInstalled
				claudeProxyListenAddr = previousAddr
				claudeDesktop = previousDesktop
				claudeAccessStateResolver = previousAccessResolver
				claudeLocalModelsResolver = previousLocalResolver
				claudeProxyMu.Lock()
				clearClaudeProxyFailure()
				claudeProxyMu.Unlock()
			})

			err := setClaudeGatewayInstalled(true, false)
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("setClaudeGatewayInstalled() error = %v, want %q", err, tt.wantError)
			}
			if fake.installed {
				t.Fatal("Claude profile changed without a usable model")
			}
			if claudeAppProxy != nil {
				t.Fatal("Claude gateway started without a usable model")
			}
			if !hasUsedClaudeDesktopIntegration() {
				t.Fatal("failed enable did not expose Claude recovery settings")
			}
		})
	}
}

func TestClaudeLocalModels(t *testing.T) {
	models, err := claudeLocalModels(context.Background(), func(context.Context) (*api.ListResponse, error) {
		return &api.ListResponse{Models: []api.ListModelResponse{
			{Name: "qwen3.8:27b-mlx", Model: "qwen3.8:27b-mlx"},
			{Name: "alias:latest", Model: "original:latest"},
			{Name: "remote-model:latest", Model: "remote-model:latest", RemoteModel: "upstream/model"},
			{Name: "remote-host:latest", Model: "remote-host:latest", RemoteHost: "https://ollama.com"},
			{Name: "cloud-source:cloud", Model: "cloud-source:cloud"},
			{Name: "legacy-cloud:31b-cloud", Model: "legacy-cloud:31b-cloud"},
		}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(models, ","); got != "qwen3.8:27b-mlx,alias:latest,alias,original:latest,original" {
		t.Fatalf("local models = %q", got)
	}
	models, err = claudeLocalModels(context.Background(), func(context.Context) (*api.ListResponse, error) {
		return &api.ListResponse{Models: []api.ListModelResponse{
			{Name: "qwen3.8:27b-mlx", Model: "qwen3.8:27b-mlx", RemoteModel: "qwen3.8:27b-mlx"},
		}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 0 {
		t.Fatalf("remote-only models = %v, want none", models)
	}

	wantErr := errors.New("list failed")
	if _, err := claudeLocalModels(context.Background(), func(context.Context) (*api.ListResponse, error) {
		return nil, wantErr
	}); !errors.Is(err, wantErr) {
		t.Fatalf("claudeLocalModels error = %v, want %v", err, wantErr)
	}
}

func TestClaudeGatewayStartupWithLocalSelectionSkipsCloudLookupsButSettingsLoadsCatalog(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("OLLAMA_HOST", "127.0.0.1:11434")
	if err := launch.SaveClaudeDesktopModels([]string{"qwen3:8b"}); err != nil {
		t.Fatal(err)
	}
	previousInstalled := claudeDesktopInstalled
	previousStore := appStore
	appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	if err := markClaudeDesktopIntegrationUsed(); err != nil {
		t.Fatal(err)
	}
	previousAddr := claudeProxyListenAddr
	previousLoader := claudeModelsLoader
	previousAccess := claudeAccessStateResolver
	previousLocal := claudeLocalModelsResolver
	claudeProxyMu.Lock()
	previousGateway := claudeAppProxy
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAppProxy = nil
	claudeProxyMu.Unlock()
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = "127.0.0.1:0"
	loaderCalls := 0
	accessCalls := 0
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		loaderCalls++
		return proxy.DefaultClaudeDesktopModels(), "fallback"
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		accessCalls++
		return proxy.ClaudeDesktopAccessState{
			Cloud:   proxy.ClaudeDesktopCloudOn,
			Account: proxy.ClaudeDesktopAccountSignedIn,
			Plan:    "pro",
		}, nil
	}
	claudeLocalModelsResolver = func(context.Context) ([]string, error) {
		return []string{"qwen3:8b"}, nil
	}
	t.Cleanup(func() {
		stopClaudeAppProxy()
		_ = appStore.Close()
		appStore = previousStore
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
		claudeModelsLoader = previousLoader
		claudeAccessStateResolver = previousAccess
		claudeLocalModelsResolver = previousLocal
		claudeProxyMu.Lock()
		claudeAppProxy = previousGateway
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	if err := startClaudeAppProxy(); err != nil {
		t.Fatal(err)
	}
	if loaderCalls != 0 || accessCalls != 0 {
		t.Fatalf("cloud startup calls = recommendations:%d access:%d, want zero", loaderCalls, accessCalls)
	}
	status := getClaudeDesktopConnectionStatus()
	var foundCloud, foundSelectedLocal bool
	for _, model := range status.Models {
		if model.Cloud {
			foundCloud = true
		}
		if model.Name == "qwen3:8b" && model.Selected {
			foundSelectedLocal = true
		}
	}
	if !foundCloud || !foundSelectedLocal {
		t.Fatalf("Settings models = %+v, want cloud choices and selected qwen3:8b", status.Models)
	}
	if len(status.Mappings) != proxy.MaxClaudeDesktopModels || status.Mappings[0].RouteID != "claude-fable-5" || status.Mappings[0].Model != "qwen3:8b" {
		t.Fatalf("local status mappings = %+v", status.Mappings)
	}
	for _, mapping := range status.Mappings[1:] {
		if mapping.Model != "" {
			t.Fatalf("unassigned route = %+v, want no model", mapping)
		}
	}
	if loaderCalls != 1 || accessCalls != 2 {
		t.Fatalf("Settings catalog calls = recommendations:%d access:%d, want recommendations:1 access:2", loaderCalls, accessCalls)
	}
	claudeProxyMu.Lock()
	models := claudeAppProxy.Models()
	claudeProxyMu.Unlock()
	if len(models) != 1 || models[0].OllamaModel != "qwen3:8b" {
		t.Fatalf("startup models = %+v", models)
	}
}

func TestClaudeGatewayLocalSelectionCatalogPolicy(t *testing.T) {
	tests := []struct {
		name               string
		access             proxy.ClaudeDesktopAccessState
		accessErr          error
		recommendations    []proxy.ClaudeDesktopModel
		wantLoaderCalls    int
		wantSettingsModels []string
	}{
		{
			name: "cloud off stays local",
			access: proxy.ClaudeDesktopAccessState{
				Cloud: proxy.ClaudeDesktopCloudOff,
			},
			wantSettingsModels: []string{"qwen3:8b"},
		},
		{
			name:               "unknown cloud policy stays local",
			accessErr:          errors.New("offline"),
			wantSettingsModels: []string{"qwen3:8b"},
		},
		{
			name: "cloud on restores recommendations",
			access: proxy.ClaudeDesktopAccessState{
				Cloud:   proxy.ClaudeDesktopCloudOn,
				Account: proxy.ClaudeDesktopAccountSignedIn,
				Plan:    "pro",
			},
			recommendations: proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{
				Model:        "glm-5.2:cloud",
				RequiredPlan: "pro",
			}}),
			wantLoaderCalls:    1,
			wantSettingsModels: []string{"glm-5.2:cloud", "qwen3:8b"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOME", t.TempDir())
			t.Setenv("OLLAMA_HOST", "127.0.0.1:11434")
			if err := launch.SaveClaudeDesktopModels([]string{"qwen3:8b"}); err != nil {
				t.Fatal(err)
			}
			previousInstalled := claudeDesktopInstalled
			previousStore := appStore
			appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
			if err := markClaudeDesktopIntegrationUsed(); err != nil {
				t.Fatal(err)
			}
			previousAddr := claudeProxyListenAddr
			previousLoader := claudeModelsLoader
			previousAccess := claudeAccessStateResolver
			previousLocal := claudeLocalModelsResolver
			claudeProxyMu.Lock()
			previousGateway := claudeAppProxy
			previousAvailable := claudeAvailableModels
			previousSource := claudeModelSource
			previousUpdated := claudeCatalogUpdated
			claudeAppProxy = nil
			claudeAvailableModels = nil
			claudeModelSource = ""
			claudeCatalogUpdated = time.Time{}
			claudeProxyMu.Unlock()
			claudeDesktopInstalled = func() bool { return true }
			claudeProxyListenAddr = "127.0.0.1:0"
			loaderCalls := 0
			accessCalls := 0
			claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
				loaderCalls++
				return tt.recommendations, "endpoint"
			}
			claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
				accessCalls++
				return tt.access, tt.accessErr
			}
			claudeLocalModelsResolver = func(context.Context) ([]string, error) {
				return []string{"qwen3:8b"}, nil
			}
			t.Cleanup(func() {
				stopClaudeAppProxy()
				_ = appStore.Close()
				appStore = previousStore
				claudeDesktopInstalled = previousInstalled
				claudeProxyListenAddr = previousAddr
				claudeModelsLoader = previousLoader
				claudeAccessStateResolver = previousAccess
				claudeLocalModelsResolver = previousLocal
				claudeProxyMu.Lock()
				claudeAppProxy = previousGateway
				claudeAvailableModels = previousAvailable
				claudeModelSource = previousSource
				claudeCatalogUpdated = previousUpdated
				claudeProxyMu.Unlock()
			})

			if err := startClaudeAppProxy(); err != nil {
				t.Fatal(err)
			}
			if loaderCalls != 0 || accessCalls != 0 {
				t.Fatalf("cloud startup calls = recommendations:%d access:%d, want zero", loaderCalls, accessCalls)
			}
			status := getClaudeDesktopConnectionStatus()
			gotSettingsModels := make([]string, len(status.Models))
			for i, model := range status.Models {
				gotSettingsModels[i] = model.Name
			}
			if !slices.Equal(gotSettingsModels, tt.wantSettingsModels) {
				t.Fatalf("Settings models = %v, want %v", gotSettingsModels, tt.wantSettingsModels)
			}
			wantAccessCalls := 1
			if tt.accessErr == nil && tt.access.Cloud == proxy.ClaudeDesktopCloudOn {
				wantAccessCalls++
			}
			if loaderCalls != tt.wantLoaderCalls || accessCalls != wantAccessCalls {
				t.Fatalf("cloud status calls = recommendations:%d access:%d, want recommendations:%d access:%d", loaderCalls, accessCalls, tt.wantLoaderCalls, wantAccessCalls)
			}
			claudeProxyMu.Lock()
			models := claudeAppProxy.Models()
			claudeProxyMu.Unlock()
			if len(models) != 1 || models[0].OllamaModel != "qwen3:8b" {
				t.Fatalf("active gateway models = %+v, want local selection unchanged", models)
			}
		})
	}
}

func TestRestoreClaudeBeforeQuit(t *testing.T) {
	called := false
	if err := restoreClaudeBeforeQuit(context.Background(), false, false, func(context.Context) error {
		called = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("restore called while Claude was not configured")
	}

	if err := restoreClaudeBeforeQuit(context.Background(), false, true, func(context.Context) error {
		called = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("restore was not called")
	}

	wantErr := errors.New("restore failed")
	if err := restoreClaudeBeforeQuit(context.Background(), false, true, func(context.Context) error {
		return wantErr
	}); !errors.Is(err, wantErr) {
		t.Fatalf("restore error = %v, want %v", err, wantErr)
	}

	called = false
	if err := restoreClaudeBeforeQuit(context.Background(), true, true, func(context.Context) error {
		called = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("restore called during an app replacement handoff")
	}
}

func TestRestoreClaudeAppForTerminationRequiresQuitConfirmation(t *testing.T) {
	previousDesktop := claudeDesktop
	fake := &fakeClaudeDesktopController{configured: true, running: true}
	claudeDesktop = fake
	t.Cleanup(func() {
		claudeDesktop = previousDesktop
	})

	err := restoreClaudeAppForTermination(context.Background(), false, false)
	if !errors.Is(err, launch.ErrClaudeDesktopQuitConfirmationRequired) {
		t.Fatalf("unconfirmed restore error = %v, want quit confirmation error", err)
	}
	if fake.restoreCalls != 1 || fake.restoreConfirmed {
		t.Fatalf("unconfirmed restore = calls:%d confirmed:%v", fake.restoreCalls, fake.restoreConfirmed)
	}
	if !fake.configured {
		t.Fatal("unconfirmed restore changed Claude's profile")
	}

	if err := restoreClaudeAppForTermination(context.Background(), false, true); err != nil {
		t.Fatalf("confirmed restore error = %v", err)
	}
	if fake.restoreCalls != 2 || !fake.restoreConfirmed {
		t.Fatalf("confirmed restore = calls:%d confirmed:%v", fake.restoreCalls, fake.restoreConfirmed)
	}
	if fake.configured {
		t.Fatal("confirmed restore left Claude configured")
	}
}

func TestSetClaudeGatewayInstalledRejectsMissingClaude(t *testing.T) {
	previousInstalled := claudeDesktopInstalled
	claudeDesktopInstalled = func() bool { return false }
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
	})

	err := setClaudeGatewayInstalled(true, false)
	if err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("setClaudeGatewayInstalled error = %v, want missing Claude error", err)
	}
	if claudeAppProxy != nil {
		t.Fatal("Claude gateway started without Claude Desktop installed")
	}
}

func TestClaudeDesktopConnectionStatusReportsMissingApp(t *testing.T) {
	previousInstalled := claudeDesktopInstalled
	claudeDesktopInstalled = func() bool { return false }
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
	})

	status := getClaudeDesktopConnectionStatus()
	if status.Installed || status.Configured || status.Connected {
		t.Fatalf("Claude status = %+v, want missing and disconnected", status)
	}
	if err := setClaudeDesktopConnection(true, false); err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("setClaudeDesktopConnection error = %v, want missing Claude error", err)
	}
	if err := prepareClaudeDesktopConnection(); err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("prepareClaudeDesktopConnection error = %v, want missing Claude error", err)
	}
}

func TestClaudeDesktopConnectionStatusKeepsConfiguredStateOnGatewayFailure(t *testing.T) {
	previousInstalled := claudeDesktopInstalled
	previousDesktop := claudeDesktop
	claudeProxyMu.Lock()
	previousErr := claudeProxyErr
	previousFailure := claudeProxyFail
	claudeProxyErr = errors.New("gateway failed")
	claudeProxyFail = claudeProxyFailurePortConflict
	claudeProxyMu.Unlock()
	claudeDesktopInstalled = func() bool { return true }
	claudeDesktop = &fakeClaudeDesktopController{configured: true}
	t.Cleanup(func() {
		claudeDesktopInstalled = previousInstalled
		claudeDesktop = previousDesktop
		claudeProxyMu.Lock()
		claudeProxyErr = previousErr
		claudeProxyFail = previousFailure
		claudeProxyMu.Unlock()
	})

	status := getClaudeDesktopConnectionStatus()
	if !status.Configured || status.Connected || !status.StartFailed {
		t.Fatalf("Claude status = %+v, want configured but unavailable", status)
	}
}

func TestClaudeDesktopInstallResultFromCode(t *testing.T) {
	for _, tt := range []struct {
		code int
		want claudeDesktopInstallResult
	}{
		{code: 0, want: claudeDesktopInstallCancelled},
		{code: 1, want: claudeDesktopInstallerOpened},
		{code: 2, want: claudeDesktopInstallFailed},
		{code: 99, want: claudeDesktopInstallFailed},
	} {
		if got := claudeDesktopInstallResultFromCode(tt.code); got != tt.want {
			t.Errorf("claudeDesktopInstallResultFromCode(%d) = %q, want %q", tt.code, got, tt.want)
		}
	}
}

func TestClaudeGatewayRejectsOllamaHostPortConflict(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "0.0.0.0:11435")

	previousInstalled := claudeDesktopInstalled
	previousAddr := claudeProxyListenAddr
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = proxy.DefaultClaudeDesktopListenAddr
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
	})

	err := startClaudeAppProxy()
	if err == nil || !strings.Contains(err.Error(), "port 11435") {
		t.Fatalf("startClaudeAppProxy error = %v, want reserved-port error", err)
	}
	if claudeAppProxy != nil {
		t.Fatal("Claude gateway started with a conflicting OLLAMA_HOST")
	}
	if !claudeGatewayStartFailed() {
		t.Fatal("expected the port conflict to remain visible to the menu")
	}
	if !claudeGatewayPortConflict() {
		t.Fatal("expected the menu failure to be classified as a port conflict")
	}
}

func TestClaudeGatewayPortTracksListenAddress(t *testing.T) {
	previousAddr := claudeProxyListenAddr
	claudeProxyListenAddr = "127.0.0.1:23001"
	t.Cleanup(func() {
		claudeProxyListenAddr = previousAddr
	})

	port, err := claudeGatewayPort()
	if err != nil {
		t.Fatal(err)
	}
	if port != "23001" || int(ClaudeGatewayPort()) != 23001 {
		t.Fatalf("Claude gateway port = %q/%d, want 23001", port, int(ClaudeGatewayPort()))
	}
}

func TestClaudeGatewayDoesNotReportPortConflictWithoutClaude(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "0.0.0.0:11435")

	previousInstalled := claudeDesktopInstalled
	previousAddr := claudeProxyListenAddr
	claudeDesktopInstalled = func() bool { return false }
	claudeProxyListenAddr = proxy.DefaultClaudeDesktopListenAddr
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeProxyMu.Lock()
		clearClaudeProxyFailure()
		claudeProxyMu.Unlock()
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
	})

	if err := startClaudeAppProxy(); err != nil {
		t.Fatalf("startClaudeAppProxy error = %v, want absent Claude to skip the gateway", err)
	}
	if claudeGatewayStartFailed() || claudeGatewayPortConflict() {
		t.Fatal("absent Claude exposed a gateway port conflict")
	}
}

func TestClaudeGatewayRecoversAfterPortConflict(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	setClaudeProxyRetry(t, 20*time.Millisecond, 5*time.Millisecond)
	conflict := httptest.NewServer(http.NotFoundHandler())
	t.Cleanup(conflict.Close)
	addr := conflict.Listener.Addr().String()

	previousInstalled := claudeDesktopInstalled
	previousAddr := claudeProxyListenAddr
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = addr
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
	})

	if err := startClaudeAppProxy(); err == nil {
		t.Fatal("startClaudeAppProxy succeeded while another service owned the port")
	}
	if !claudeGatewayStartFailed() {
		t.Fatal("expected the failed start to remain visible to the menu")
	}
	if !claudeGatewayPortConflict() {
		t.Fatal("expected an occupied gateway port to be classified as a conflict")
	}

	conflict.Close()
	if err := startClaudeAppProxy(); err != nil {
		t.Fatalf("startClaudeAppProxy did not recover after the port was released: %v", err)
	}
	if claudeGatewayStartFailed() {
		t.Fatal("expected a successful retry to clear the menu failure state")
	}
	if claudeGatewayPortConflict() {
		t.Fatal("expected a successful retry to clear the port conflict state")
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := proxy.ProbeClaudeDesktop(ctx, "http://"+addr); err != nil {
		t.Fatalf("recovered Claude gateway is not reachable: %v", err)
	}
}

func TestClaudeGatewayRejectsSpoofedExistingGateway(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	setClaudeProxyRetry(t, 20*time.Millisecond, 5*time.Millisecond)
	spoof := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Ollama-Claude-Gateway", "1")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer spoof.Close()

	previousInstalled := claudeDesktopInstalled
	previousAddr := claudeProxyListenAddr
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = spoof.Listener.Addr().String()
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
	})

	if err := startClaudeAppProxy(); err == nil {
		t.Fatal("startClaudeAppProxy trusted a listener with a spoofed health response")
	}
	if claudeAppProxy != nil || !claudeGatewayPortConflict() {
		t.Fatal("spoofed listener was not reported as a port conflict")
	}
}

func TestClaudeGatewayWaitsForPreviousListenerToExit(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	setClaudeProxyRetry(t, 500*time.Millisecond, 5*time.Millisecond)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	previousInstalled := claudeDesktopInstalled
	previousAddr := claudeProxyListenAddr
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = listener.Addr().String()
	t.Cleanup(func() {
		_ = listener.Close()
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
	})

	go func() {
		time.Sleep(30 * time.Millisecond)
		_ = listener.Close()
	}()
	if err := startClaudeAppProxy(); err != nil {
		t.Fatalf("startClaudeAppProxy did not acquire a released handoff port: %v", err)
	}
	if claudeGatewayStartFailed() || claudeAppProxy == nil {
		t.Fatal("released handoff port did not start the Claude gateway")
	}
}

func setClaudeProxyRetry(t *testing.T, wait, poll time.Duration) {
	t.Helper()
	previousWait := claudeProxyRetryWait
	previousPoll := claudeProxyRetryPoll
	claudeProxyRetryWait = wait
	claudeProxyRetryPoll = poll
	t.Cleanup(func() {
		claudeProxyRetryWait = previousWait
		claudeProxyRetryPoll = previousPoll
	})
}
