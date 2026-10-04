//go:build darwin

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
		mappings := api.ModelRecommendationMappings{
			"claude-opus-5": {Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		}
		return claudeDesktopRecommendationModelsForTest(t, []api.ModelRecommendation{{Model: "glm-5.2:cloud"}}, &mappings), "endpoint"
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

func TestClaudeDesktopCatalogListsAccountModelsWithoutChangingDefaults(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	mappings := api.ModelRecommendationMappings{
		"claude-sonnet-5": {Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
	}
	recommendations := claudeDesktopRecommendationModelsForTest(t, []api.ModelRecommendation{
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
	}, &mappings)

	previousLoader := claudeModelsLoader
	previousAccess := claudeAccessStateResolver
	previousCloud := claudeCloudModelsResolver
	claudeProxyMu.Lock()
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeProxyMu.Unlock()
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		return recommendations, "endpoint"
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{
			Cloud:   proxy.ClaudeDesktopCloudOn,
			Account: proxy.ClaudeDesktopAccountSignedIn,
			Plan:    "pro",
		}, nil
	}
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{
			"glm-5.3-flash:cloud",
			"deepseek-v4-flash:cloud",
		}), nil
	}
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeAccessStateResolver = previousAccess
		claudeCloudModelsResolver = previousCloud
		claudeProxyMu.Lock()
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	assertCatalog := func(t *testing.T, available, selected []proxy.ClaudeDesktopModel) {
		t.Helper()
		if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, map[string]string{
			"claude-sonnet-5": "glm-5.3-flash:cloud",
		}) {
			t.Fatalf("default mappings = %v, want GLM Flash for Sonnet 5", got)
		}
		for _, model := range available {
			if model.OllamaModel != "deepseek-v4-flash:cloud" {
				continue
			}
			if model.Recommended {
				t.Fatal("account DeepSeek model was marked as recommended")
			}
			if !model.AccountCloud {
				t.Fatal("account DeepSeek model is missing cloud inventory membership")
			}
			return
		}
		t.Fatal("account DeepSeek model is missing from the selectable catalog")
	}

	available, selected, source := resolveClaudeDesktopStartupCatalog(context.Background())
	if source != "endpoint" {
		t.Fatalf("source = %q, want endpoint", source)
	}
	assertCatalog(t, available, selected)

	available, selected, source = refreshClaudeDesktopCatalog(context.Background(), selected, true)
	if source != "endpoint" {
		t.Fatalf("refreshed source = %q, want endpoint", source)
	}
	assertCatalog(t, available, selected)
}

func TestResolveClaudeDesktopStartupCatalogUsesSafeFallback(t *testing.T) {
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
			want := proxy.DefaultClaudeDesktopMappings()
			if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, want) {
				t.Fatalf("startup mappings = %v, want %v (source %q)", got, want, source)
			}
		})
	}
}

func TestResolveClaudeDesktopStartupCatalogUsesEffectiveEndpointMappings(t *testing.T) {
	recommendations := []api.ModelRecommendation{
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free"},
	}
	tests := []struct {
		name     string
		state    proxy.ClaudeDesktopAccessState
		mappings api.ModelRecommendationMappings
		saved    map[string]string
		want     map[string]string
	}{
		{
			name:     "mapping with free plan metadata",
			state:    proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
			mappings: api.ModelRecommendationMappings{"claude-sonnet-5": {Model: "gemma4:31b-cloud", RequiredPlan: "free"}},
			want:     map[string]string{"claude-sonnet-5": "gemma4:31b-cloud"},
		},
		{
			name:     "mapping plan metadata is informational",
			state:    proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
			mappings: api.ModelRecommendationMappings{"claude-sonnet-5": {Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"}},
			want:     map[string]string{"claude-sonnet-5": "glm-5.3-flash:cloud"},
		},
		{
			name:     "persisted user mapping wins",
			state:    proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"},
			mappings: api.ModelRecommendationMappings{"claude-sonnet-5": {Model: "glm-5.3-flash:cloud"}},
			saved:    map[string]string{"claude-opus-5": "gemma4:31b-cloud"},
			want:     map[string]string{"claude-opus-5": "gemma4:31b-cloud"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOME", t.TempDir())
			if tt.saved != nil {
				if err := launch.SaveClaudeDesktopModelMappings(tt.saved); err != nil {
					t.Fatal(err)
				}
			}
			models := claudeDesktopRecommendationModelsForTest(t, recommendations, &tt.mappings)

			previousLoader := claudeModelsLoader
			previousAccess := claudeAccessStateResolver
			previousCloud := claudeCloudModelsResolver
			claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
				return models, "endpoint"
			}
			claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
				return tt.state, nil
			}
			claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
				return proxy.ClaudeDesktopModelsFromCloudInventory([]string{"glm-5.3-flash:cloud", "gemma4:31b-cloud"}), nil
			}
			t.Cleanup(func() {
				claudeModelsLoader = previousLoader
				claudeAccessStateResolver = previousAccess
				claudeCloudModelsResolver = previousCloud
			})

			_, selected, source := resolveClaudeDesktopStartupCatalog(context.Background())
			if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, tt.want) {
				t.Fatalf("startup mappings = %v, want %v (source %q)", got, tt.want, source)
			}
		})
	}
}

func TestResolveClaudeDesktopStartupCatalogUpdatesDefaultsAfterReconnect(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	endpointModels := claudeDesktopRecommendationModelsForTest(t, []api.ModelRecommendation{
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
	}, &api.ModelRecommendationMappings{"claude-sonnet-5": {Model: "glm-5.3-flash:cloud"}})

	previousLoader := claudeModelsLoader
	previousAccess := claudeAccessStateResolver
	previousCloud := claudeCloudModelsResolver
	load := 0
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		load++
		if load == 1 {
			return fallbackClaudeDesktopModels(), "fallback"
		}
		return endpointModels, "endpoint"
	}
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"}, nil
	}
	claudeCloudModelsResolver = func(context.Context) ([]proxy.ClaudeDesktopModel, error) {
		return proxy.ClaudeDesktopModelsFromCloudInventory([]string{
			"glm-5.3-flash:cloud", "glm-5.2:cloud", "kimi-k3:cloud", "deepseek-v4-pro:cloud", "deepseek-v4-flash:0731:cloud", "gemma4:31b-cloud",
		}), nil
	}
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeAccessStateResolver = previousAccess
		claudeCloudModelsResolver = previousCloud
	})

	_, offline, source := resolveClaudeDesktopStartupCatalog(context.Background())
	if got := proxy.ClaudeDesktopMappings(offline)["claude-sonnet-5"]; source != "fallback" || got != "gemma4:31b-cloud" {
		t.Fatalf("offline Sonnet/source = %q/%q", got, source)
	}
	_, reconnected, source := resolveClaudeDesktopStartupCatalog(context.Background())
	if got := proxy.ClaudeDesktopMappings(reconnected); source != "endpoint" || !maps.Equal(got, map[string]string{"claude-sonnet-5": "glm-5.3-flash:cloud"}) {
		t.Fatalf("reconnected mappings/source = %v/%q", got, source)
	}
}

func claudeDesktopRecommendationModelsForTest(t *testing.T, recommendations []api.ModelRecommendation, mappings *api.ModelRecommendationMappings) []proxy.ClaudeDesktopModel {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(api.ModelRecommendationsResponse{
			Recommendations: recommendations,
			Mappings:        mappings,
		})
	}))
	defer server.Close()
	req, err := http.NewRequest(http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	models, err := proxy.FetchClaudeDesktopModels(server.Client(), req)
	if err != nil {
		t.Fatal(err)
	}
	return models
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

func TestRefreshClaudeDesktopCatalogDropsStaleDefaultsOnFallback(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	endpointModels := claudeDesktopRecommendationModelsForTest(t, []api.ModelRecommendation{
		{Model: "glm-5.3-flash:cloud", RequiredPlan: "pro"},
		{Model: "glm-5.2:cloud", RequiredPlan: "pro"},
		{Model: "kimi-k3:cloud", RequiredPlan: "pro"},
		{Model: "deepseek-v4-pro", RequiredPlan: "pro"},
		{Model: "deepseek-v4-flash", RequiredPlan: "pro"},
		{Model: "gemma4:31b-cloud", RequiredPlan: "free"},
	}, &api.ModelRecommendationMappings{"claude-sonnet-5": {Model: "glm-5.3-flash:cloud"}})
	current := proxy.MapClaudeDesktopModels(endpointModels, map[string]string{"claude-sonnet-5": "glm-5.3-flash:cloud"})

	previousLoader := claudeModelsLoader
	claudeProxyMu.Lock()
	previousAvailable := claudeAvailableModels
	previousSource := claudeModelSource
	previousUpdated := claudeCatalogUpdated
	claudeAvailableModels = endpointModels
	claudeModelSource = "endpoint"
	claudeCatalogUpdated = time.Now()
	claudeProxyMu.Unlock()
	claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
		return fallbackClaudeDesktopModels(), "fallback"
	}
	t.Cleanup(func() {
		claudeModelsLoader = previousLoader
		claudeProxyMu.Lock()
		claudeAvailableModels = previousAvailable
		claudeModelSource = previousSource
		claudeCatalogUpdated = previousUpdated
		claudeProxyMu.Unlock()
	})

	available, selected, source := refreshClaudeDesktopCatalog(context.Background(), current, true)
	if source != "fallback" {
		t.Fatalf("source = %q, want fallback", source)
	}
	if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, map[string]string{"claude-sonnet-5": "glm-5.3-flash:cloud"}) {
		t.Fatalf("current mappings = %v, want preserved explicit mapping", got)
	}
	if got := proxy.DefaultClaudeDesktopMappingsForModels(available)["claude-sonnet-5"]; got != "gemma4:31b-cloud" {
		t.Fatalf("offline default Sonnet = %q, want compatibility fallback", got)
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

func sharedClaudeDesktopMappings(model string) map[string]string {
	mappings := make(map[string]string, proxy.MaxClaudeDesktopModels)
	for _, route := range proxy.ClaudeDesktopRoutes() {
		mappings[route.ID] = model
	}
	return mappings
}

func TestClaudeDesktopDefaultsDoNotDependOnAccountPlan(t *testing.T) {
	want := proxy.DefaultClaudeDesktopMappings()
	tests := []struct {
		name  string
		state proxy.ClaudeDesktopAccessState
	}{
		{name: "signed out", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedOut}},
		{name: "free", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}},
		{name: "Pro", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"}},
		{name: "Team", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "team"}},
		{name: "future paid plan", state: proxy.ClaudeDesktopAccessState{Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "enterprise"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := proxy.DefaultClaudeDesktopMappingsForModels(proxy.DefaultClaudeDesktopModels())
			if !maps.Equal(got, want) {
				t.Fatalf("default mappings for state %+v = %v, want %v", tt.state, got, want)
			}
		})
	}
}

type fakeClaudeDesktopController struct {
	configured     bool
	profileCurrent bool
	configureCalls int
	configureErr   error
	installed      bool
	autoMode       bool
	restoreCalls   int
}

func (f *fakeClaudeDesktopController) UsesOllamaGateway() bool { return f.configured }

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
	f.installed = installed
	f.autoMode = autoMode
	return nil
}

func (f *fakeClaudeDesktopController) RestoreForShutdown(context.Context) error {
	f.restoreCalls++
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
	recommended = mergeClaudeDesktopCloudInventory(recommended, accountCloud)
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

func TestClaudeGatewayStartupWithLocalSelectionSkipsCloudLookups(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("OLLAMA_HOST", "127.0.0.1:11434")
	if err := launch.SaveClaudeDesktopModels([]string{"qwen3:8b"}); err != nil {
		t.Fatal(err)
	}
	previousInstalled := claudeDesktopInstalled
	previousStore := appStore
	appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
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
	claudeProxyMu.Lock()
	models := claudeAppProxy.Models()
	claudeProxyMu.Unlock()
	if len(models) != 1 || models[0].OllamaModel != "qwen3:8b" {
		t.Fatalf("startup models = %+v", models)
	}
}

func TestRunAppSyncBarrierStopsOlderInstances(t *testing.T) {
	for _, test := range []struct {
		name      string
		exitOn    appProcessStopMode
		wantStops []appProcessStopMode
	}{
		{name: "handoff", exitOn: appProcessStopForHandoff, wantStops: []appProcessStopMode{appProcessStopForHandoff}},
		{name: "graceful", exitOn: appProcessStopGracefully, wantStops: []appProcessStopMode{appProcessStopForHandoff, appProcessStopGracefully}},
		{name: "forced", exitOn: appProcessStopForcefully, wantStops: []appProcessStopMode{appProcessStopForHandoff, appProcessStopGracefully, appProcessStopForcefully}},
	} {
		t.Run(test.name, func(t *testing.T) {
			self := appProcessIdentity{pid: 20, startedAt: 20}
			older := appProcessIdentity{pid: 10, startedAt: 10}
			alive := true
			var stops []appProcessStopMode
			controller := appProcessController{
				discover: func() ([]appProcessIdentity, error) {
					if alive {
						return []appProcessIdentity{older}, nil
					}
					return nil, nil
				},
				running: func(process appProcessIdentity) (bool, error) {
					return alive && process.sameProcess(older), nil
				},
				stop: func(process appProcessIdentity, mode appProcessStopMode) error {
					if !process.sameProcess(older) {
						t.Fatalf("stopped process %+v, want %+v", process, older)
					}
					stops = append(stops, mode)
					if mode == test.exitOn {
						alive = false
					}
					return nil
				},
			}

			err := runAppSyncBarrier(self, controller, appSyncBarrierConfig{
				killTimeout:  time.Second,
				pollInterval: time.Millisecond,
			})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(stops, test.wantStops) {
				t.Fatalf("stop modes = %v, want %v", stops, test.wantStops)
			}
		})
	}
}

func TestRunAppSyncBarrierStopsEveryOlderInstance(t *testing.T) {
	self := appProcessIdentity{pid: 30, startedAt: 30}
	graceful := appProcessIdentity{pid: 10, startedAt: 10}
	stubborn := appProcessIdentity{pid: 20, startedAt: 20}
	late := appProcessIdentity{pid: 25, startedAt: 25}
	alive := map[appProcessIdentity]bool{
		graceful: true,
		stubborn: true,
	}
	type stoppedProcess struct {
		process appProcessIdentity
		mode    appProcessStopMode
	}
	var stops []stoppedProcess
	lateDiscovered := false
	controller := appProcessController{
		discover: func() ([]appProcessIdentity, error) {
			if !alive[graceful] && !alive[stubborn] && !lateDiscovered {
				alive[late] = true
				lateDiscovered = true
			}
			var processes []appProcessIdentity
			for _, process := range []appProcessIdentity{graceful, stubborn, late} {
				if alive[process] {
					processes = append(processes, process)
				}
			}
			return processes, nil
		},
		running: func(process appProcessIdentity) (bool, error) {
			return alive[process], nil
		},
		stop: func(process appProcessIdentity, mode appProcessStopMode) error {
			if !alive[process] {
				return nil
			}
			stops = append(stops, stoppedProcess{process: process, mode: mode})
			if !process.sameProcess(stubborn) || mode == appProcessStopForcefully {
				alive[process] = false
			}
			return nil
		},
	}

	err := runAppSyncBarrier(self, controller, appSyncBarrierConfig{
		killTimeout:  time.Second,
		pollInterval: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []stoppedProcess{
		{process: graceful, mode: appProcessStopForHandoff},
		{process: stubborn, mode: appProcessStopForHandoff},
		{process: stubborn, mode: appProcessStopGracefully},
		{process: stubborn, mode: appProcessStopForcefully},
		{process: late, mode: appProcessStopForHandoff},
	}
	if !slices.Equal(stops, want) {
		t.Fatalf("stops = %+v, want %+v", stops, want)
	}
}

func TestRunAppSyncBarrierDefersToNewerInstance(t *testing.T) {
	self := appProcessIdentity{pid: 10, startedAt: 10}
	newer := appProcessIdentity{pid: 20, startedAt: 20}
	stopped := false
	err := runAppSyncBarrier(self, appProcessController{
		discover: func() ([]appProcessIdentity, error) {
			return []appProcessIdentity{newer}, nil
		},
		running: func(appProcessIdentity) (bool, error) { return true, nil },
		stop: func(appProcessIdentity, appProcessStopMode) error {
			stopped = true
			return nil
		},
	}, appSyncBarrierConfig{killTimeout: time.Second})
	if !errors.Is(err, errNewerAppInstance) {
		t.Fatalf("barrier error = %v, want newer-instance error", err)
	}
	if stopped {
		t.Fatal("older launch stopped the newer instance")
	}
}

func TestRunAppSyncBarrierRequiresSettledEmptyList(t *testing.T) {
	queries := 0
	err := runAppSyncBarrier(appProcessIdentity{pid: 1, startedAt: 1}, appProcessController{
		discover: func() ([]appProcessIdentity, error) {
			queries++
			return nil, nil
		},
	}, appSyncBarrierConfig{killTimeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	if queries != 2 {
		t.Fatalf("discovery queries = %d, want 2", queries)
	}
}

func TestContinueAfterBarrierErrorOnlyBlocksNewerInstance(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "success", err: nil, want: true},
		{name: "discovery failure", err: errors.New("discover other Ollama app processes"), want: true},
		{name: "handoff timeout", err: errors.New("timed out waiting for app instances to exit"), want: true},
		{name: "newer instance", err: fmt.Errorf("%w: pid 2", errNewerAppInstance), want: false},
		{name: "wrapped newer instance", err: fmt.Errorf("barrier: %w", fmt.Errorf("%w: pid 2", errNewerAppInstance)), want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := continueAfterBarrierError(tt.err); got != tt.want {
				t.Fatalf("continueAfterBarrierError(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestRestoreClaudeBeforeQuit(t *testing.T) {
	called := false
	if err := restoreClaudeBeforeQuit(context.Background(), false, func(context.Context) error {
		called = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("restore called while Claude was not configured")
	}

	if err := restoreClaudeBeforeQuit(context.Background(), true, func(context.Context) error {
		called = true
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("restore was not called")
	}

	wantErr := errors.New("restore failed")
	if err := restoreClaudeBeforeQuit(context.Background(), true, func(context.Context) error {
		return wantErr
	}); !errors.Is(err, wantErr) {
		t.Fatalf("restore error = %v, want %v", err, wantErr)
	}
}

func TestRestoreClaudeAppForTerminationPreservesHandoffProfile(t *testing.T) {
	previousDesktop := claudeDesktop
	fake := &fakeClaudeDesktopController{configured: true}
	claudeDesktop = fake
	t.Cleanup(func() { claudeDesktop = previousDesktop })

	if err := restoreClaudeAppForTermination(context.Background(), true); err != nil {
		t.Fatal(err)
	}
	if fake.restoreCalls != 0 || !fake.configured {
		t.Fatalf("handoff restore calls/configured = %d/%v, want 0/true", fake.restoreCalls, fake.configured)
	}
}

func TestHandleExistingInstanceSkipsDevelopmentBuild(t *testing.T) {
	oldIsApp := isApp
	oldKillOtherInstances := killOtherInstances
	t.Cleanup(func() {
		isApp = oldIsApp
		killOtherInstances = oldKillOtherInstances
	})

	isApp = false
	called := false
	killOtherInstances = func() bool {
		called = true
		return false
	}

	if !handleExistingInstance(false) {
		t.Fatal("development instance did not continue startup")
	}
	if called {
		t.Fatal("development instance entered the packaged app handoff barrier")
	}
}

func TestHandleExistingInstanceReturnsBarrierResult(t *testing.T) {
	oldIsApp := isApp
	oldKillOtherInstances := killOtherInstances
	t.Cleanup(func() {
		isApp = oldIsApp
		killOtherInstances = oldKillOtherInstances
	})

	isApp = true
	for _, want := range []bool{true, false} {
		killOtherInstances = func() bool { return want }
		if got := handleExistingInstance(false); got != want {
			t.Fatalf("handleExistingInstance() = %v, want %v", got, want)
		}
	}
}

func TestDarwinAppProcessRunningReportsExitedProcess(t *testing.T) {
	running, err := darwinAppProcessRunning(appProcessIdentity{pid: 1 << 30, startedAt: 1})
	if err != nil {
		t.Fatal(err)
	}
	if running {
		t.Fatal("nonexistent process reported as running")
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
