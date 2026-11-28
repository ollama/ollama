//go:build darwin

package main

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/proxy"
	"github.com/ollama/ollama/types/model"
)

func TestClaudeLocalChatModels(t *testing.T) {
	chat := []model.Capability{model.CapabilityCompletion}
	tools := []model.Capability{model.CapabilityCompletion, model.CapabilityTools}
	got, err := claudeLocalChatModels(context.Background(), func(context.Context) (*api.ListResponse, error) {
		return &api.ListResponse{Models: []api.ListModelResponse{
			{Name: "embedding:latest", Capabilities: []model.Capability{model.CapabilityEmbedding}},
			{Name: "phi:latest", Capabilities: chat},
			{Name: "qwen:8b", Capabilities: tools},
			{Name: "unreported:latest"},
			{Name: "cloud:latest", RemoteHost: "https://ollama.com", Capabilities: tools},
			{Name: "remote:latest", RemoteModel: "upstream", Capabilities: tools},
			{Name: "tagged:cloud", Capabilities: tools},
			{Name: "legacy:31b-cloud", Capabilities: tools},
			{Name: "image:latest", Capabilities: []model.Capability{model.CapabilityImage}},
			{Name: "other:latest", Capabilities: tools},
		}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"qwen:8b", "other:latest", "phi:latest"}; !slices.Equal(got, want) {
		t.Fatalf("chat models = %v, want %v", got, want)
	}
}

func TestDefaultClaudeDesktopSelection(t *testing.T) {
	previous := claudeLocalChatModelsResolver
	t.Cleanup(func() { claudeLocalChatModelsResolver = previous })
	cloud := proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{
		{Model: "gemma4:31b-cloud", RequiredPlan: "pro"},
	})
	for _, tt := range []struct {
		name       string
		state      proxy.ClaudeDesktopAccessState
		local      []string
		localErr   error
		wantModel  string
		wantLookup bool
	}{
		{name: "cloud off", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, local: []string{"qwen:8b"}, wantModel: "qwen:8b", wantLookup: true},
		{name: "signed out", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}, local: []string{"qwen:8b"}, wantModel: "qwen:8b", wantLookup: true},
		{name: "insufficient plan", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}, local: []string{"qwen:8b"}, wantModel: "qwen:8b", wantLookup: true},
		{name: "usable cloud default", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"}, wantModel: "gemma4:31b-cloud"},
		{name: "unverified account retains retry", state: proxy.ClaudeDesktopAccessState{}, wantModel: "gemma4:31b-cloud"},
		{name: "no eligible local model retains error", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, wantModel: "gemma4:31b-cloud", wantLookup: true},
		{name: "local lookup failed", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, localErr: errors.New("list unavailable"), wantModel: "gemma4:31b-cloud", wantLookup: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			lookedUp := false
			claudeLocalChatModelsResolver = func(context.Context) ([]string, error) {
				lookedUp = true
				return tt.local, tt.localErr
			}
			got := proxy.ClaudeDesktopMappings(defaultClaudeDesktopSelection(context.Background(), cloud, tt.state))
			if want := map[string]string{"claude-sonnet-5": tt.wantModel}; !maps.Equal(got, want) {
				t.Fatalf("default mappings = %v, want %v", got, want)
			}
			if lookedUp != tt.wantLookup {
				t.Fatalf("local inventory lookup = %v, want %v", lookedUp, tt.wantLookup)
			}
		})
	}
}

func TestFreshClaudeConnectionDefaultsMatchSettings(t *testing.T) {
	for _, tt := range []struct {
		name        string
		state       proxy.ClaudeDesktopAccessState
		accessErr   error
		wantModel   string
		wantAccess  proxy.ClaudeDesktopAvailability
		retryAccess bool
	}{
		{name: "cloud off", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, wantModel: "qwen:8b", wantAccess: proxy.ClaudeDesktopAvailabilityAvailable},
		{name: "signed out", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}, wantModel: "qwen:8b", wantAccess: proxy.ClaudeDesktopAvailabilityAvailable},
		{name: "cloud status unavailable", accessErr: errors.New("status unavailable"), wantModel: "gemma4:31b-cloud", wantAccess: proxy.ClaudeDesktopAvailabilityUnknown},
		{name: "account lookup unavailable", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountUnknown}, accessErr: errors.New("account unavailable"), wantModel: "gemma4:31b-cloud", wantAccess: proxy.ClaudeDesktopAvailabilityUnknown},
		{name: "cloud off after access retry", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, wantModel: "qwen:8b", wantAccess: proxy.ClaudeDesktopAvailabilityAvailable, retryAccess: true},
		{name: "signed out after access retry", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}, wantModel: "qwen:8b", wantAccess: proxy.ClaudeDesktopAvailabilityAvailable, retryAccess: true},
		{name: "insufficient plan after access retry", state: proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"}, wantModel: "qwen:8b", wantAccess: proxy.ClaudeDesktopAvailabilityAvailable, retryAccess: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("HOME", t.TempDir())
			previousStore, previousDesktop := appStore, claudeDesktop
			previousInstalled, previousAddr := claudeDesktopInstalled, claudeProxyListenAddr
			previousAccess, previousLocal, previousChat := claudeAccessStateResolver, claudeLocalModelsResolver, claudeLocalChatModelsResolver
			previousLoader := claudeModelsLoader
			appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
			claudeDesktop = &fakeClaudeDesktopController{configureOnSet: true}
			claudeDesktopInstalled = func() bool { return true }
			claudeProxyListenAddr = "127.0.0.1:0"
			claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) { return tt.state, tt.accessErr }
			claudeLocalModelsResolver = func(context.Context) ([]string, error) { return []string{"qwen:8b"}, nil }
			claudeLocalChatModelsResolver = func(context.Context) ([]string, error) { return []string{"qwen:8b"}, nil }
			claudeModelsLoader = func(context.Context) ([]proxy.ClaudeDesktopModel, string) {
				return proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "gemma4:31b-cloud", RequiredPlan: "pro"}}), "endpoint"
			}
			claudeProxyMu.Lock()
			previousAvailable, previousSource, previousUpdated := claudeAvailableModels, claudeModelSource, claudeCatalogUpdated
			claudeAvailableModels, claudeModelSource, claudeCatalogUpdated = nil, "", time.Time{}
			claudeProxyMu.Unlock()
			t.Cleanup(func() {
				stopClaudeAppProxy()
				_ = appStore.Close()
				appStore, claudeDesktop = previousStore, previousDesktop
				claudeDesktopInstalled, claudeProxyListenAddr = previousInstalled, previousAddr
				claudeAccessStateResolver, claudeLocalModelsResolver, claudeLocalChatModelsResolver = previousAccess, previousLocal, previousChat
				claudeModelsLoader = previousLoader
				claudeProxyMu.Lock()
				claudeAvailableModels, claudeModelSource, claudeCatalogUpdated = previousAvailable, previousSource, previousUpdated
				clearClaudeProxyFailure()
				claudeProxyMu.Unlock()
			})
			if err := markClaudeDesktopIntegrationUsed(); err != nil {
				t.Fatal(err)
			}
			checkStatus := func() {
				t.Helper()
				status := getClaudeDesktopConnectionStatus()
				mapped := false
				for _, mapping := range status.Mappings {
					mapped = mapped || mapping.RouteID == "claude-sonnet-5" && mapping.Model == tt.wantModel
				}
				listed := false
				for _, item := range status.Models {
					listed = listed || item.Name == tt.wantModel && item.Availability == tt.wantAccess
				}
				if !mapped || !listed {
					t.Fatalf("Settings must display default %q with access %q: %+v", tt.wantModel, tt.wantAccess, status)
				}
			}
			checkStatus()
			_, selected, _ := resolveClaudeDesktopStartupCatalog(context.Background())
			if got, want := proxy.ClaudeDesktopMappings(selected), map[string]string{"claude-sonnet-5": tt.wantModel}; !maps.Equal(got, want) {
				t.Fatalf("startup mappings = %v, want %v", got, want)
			}
			if tt.accessErr != nil {
				if err := validateClaudeDesktopModels(selected, tt.state, []string{"qwen:8b"}, true); !errors.Is(err, errClaudeDesktopAccessUnavailable) {
					t.Fatalf("startup validation = %v, want account verification retry", err)
				}
				return
			}
			if tt.retryAccess {
				attempts := 0
				claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
					attempts++
					if attempts == 1 {
						return proxy.ClaudeDesktopAccessState{}, errors.New("temporary account lookup failure")
					}
					return tt.state, nil
				}
			}
			if err := setClaudeGatewayInstalled(true, false); err != nil {
				t.Fatal(err)
			}
			if got := claudeAppProxy.Models(); len(got) != 1 || got[0].OllamaModel != "qwen:8b" || got[0].Cloud {
				t.Fatalf("gateway models = %+v, want local qwen:8b", got)
			}
			checkStatus()
			if got := launch.ClaudeDesktopModels(); len(got) != 0 {
				t.Fatalf("automatic default changed saved selection: %v", got)
			}
		})
	}
}

func TestClaudeStartupPreservesExplicitUnavailableSelection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	want := map[string]string{"claude-sonnet-5": "glm-5.2:cloud"}
	if err := launch.SaveClaudeDesktopModelMappings(want); err != nil {
		t.Fatal(err)
	}
	previousAccess, previousLocal, previousChat := claudeAccessStateResolver, claudeLocalModelsResolver, claudeLocalChatModelsResolver
	claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, nil
	}
	claudeLocalModelsResolver = func(context.Context) ([]string, error) { return []string{"qwen:8b"}, nil }
	claudeLocalChatModelsResolver = func(context.Context) ([]string, error) {
		t.Fatal("an explicit selection must not use automatic defaults")
		return []string{"qwen:8b"}, nil
	}
	t.Cleanup(func() {
		claudeAccessStateResolver, claudeLocalModelsResolver, claudeLocalChatModelsResolver = previousAccess, previousLocal, previousChat
	})
	_, selected, _ := resolveClaudeDesktopStartupCatalog(context.Background())
	if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, want) {
		t.Fatalf("saved mappings changed from %v to %v", want, got)
	}
}

func TestClaudeDisconnectedSettingsPreserveExplicitMappings(t *testing.T) {
	for _, cloudOff := range []bool{true, false} {
		for _, want := range []map[string]string{
			{"claude-sonnet-5": "qwen:8b"},
			{"claude-opus-5": "qwen:8b", "claude-sonnet-5": "qwen:8b"},
		} {
			t.Run(fmt.Sprintf("cloudOff=%t/mappings=%d", cloudOff, len(want)), func(t *testing.T) {
				t.Setenv("HOME", t.TempDir())
				previousStore, previousDesktop := appStore, claudeDesktop
				previousAccess, previousLocal := claudeAccessStateResolver, claudeLocalModelsResolver
				appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
				claudeDesktop = &fakeClaudeDesktopController{}
				claudeAccessStateResolver = func(context.Context) (proxy.ClaudeDesktopAccessState, error) {
					if cloudOff {
						return proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, nil
					}
					return proxy.ClaudeDesktopAccessState{}, errors.New("account lookup unavailable")
				}
				claudeLocalModelsResolver = func(context.Context) ([]string, error) { return []string{"qwen:8b"}, nil }
				t.Cleanup(func() {
					_ = appStore.Close()
					appStore, claudeDesktop = previousStore, previousDesktop
					claudeAccessStateResolver, claudeLocalModelsResolver = previousAccess, previousLocal
				})
				if err := launch.SaveClaudeDesktopModelMappings(want); err != nil {
					t.Fatal(err)
				}
				if err := markClaudeDesktopIntegrationUsed(); err != nil {
					t.Fatal(err)
				}
				status := getClaudeDesktopConnectionStatus()
				got := make(map[string]string)
				for _, mapping := range status.Mappings {
					if mapping.Model != "" {
						got[mapping.RouteID] = mapping.Model
					}
				}
				if !maps.Equal(got, want) {
					t.Fatalf("Settings mappings = %v, want saved mappings %v", got, want)
				}
				_, selected, _ := resolveClaudeDesktopStartupCatalog(context.Background())
				if got := proxy.ClaudeDesktopMappings(selected); !maps.Equal(got, want) {
					t.Fatalf("startup mappings = %v, want saved mappings %v", got, want)
				}
			})
		}
	}
}
