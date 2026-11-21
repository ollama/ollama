//go:build darwin

package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/proxy"
)

func TestDesktopModelsClaudeSummaryPreservesMappingsWithoutCatalog(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	previousStore, previousInstalled := appStore, claudeDesktopInstalled
	appStore = &store.Store{DBPath: filepath.Join(t.TempDir(), "db.sqlite")}
	claudeDesktopInstalled = func() bool { return false }
	claudeProxyMu.Lock()
	previousGateway, previousAvailable := claudeAppProxy, claudeAvailableModels
	claudeAppProxy, claudeAvailableModels = nil, nil
	claudeProxyMu.Unlock()
	t.Cleanup(func() {
		appStore.Close()
		appStore, claudeDesktopInstalled = previousStore, previousInstalled
		claudeProxyMu.Lock()
		claudeAppProxy, claudeAvailableModels = previousGateway, previousAvailable
		claudeProxyMu.Unlock()
	})
	if err := launch.SaveClaudeDesktopModelMappings(map[string]string{"claude-sonnet-5": "saved-model"}); err != nil {
		t.Fatal(err)
	}
	status := claudeDesktopSettingsSummary()
	for _, mapping := range status.Mappings {
		if mapping.RouteID == "claude-sonnet-5" && mapping.Model == "saved-model" {
			for _, model := range status.Models {
				if model.Name == mapping.Model && model.Availability == "unknown" {
					return
				}
			}
		}
	}
	t.Fatalf("saved mapping missing from summary: %#v", status)
}

func TestDesktopModelsSavedSettingsWithoutDiscovery(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	controller, factory := codexDesktop, codexDesktopClientFactory
	t.Cleanup(func() { codexDesktop, codexDesktopClientFactory = controller, factory })
	if err := config.SaveIntegration(codexDesktopIntegrationName, []string{"saved-model"}); err != nil {
		t.Fatal(err)
	}
	codexDesktopClientFactory = func() (*api.Client, error) {
		t.Error("saved settings must not discover models")
		return nil, errors.New("discovery unavailable")
	}
	for _, tt := range []struct {
		name, query string
		installed   bool
	}{
		{"saved settings", "catalog=false", true},
		{"uninstalled app", "catalog=true", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			codexDesktop = &fakeCodexDesktopController{installed: tt.installed}
			r := httptest.NewRequest(http.MethodGet, "/?"+tt.query, nil)
			r.SetPathValue("integration", "chatgpt")
			w := httptest.NewRecorder()
			desktopModelSettingsHandler().ServeHTTP(w, r)
			var result codexDesktopModelsSettingsResult
			if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if w.Code != http.StatusOK || !slices.Equal(result.Settings.Selected, []string{"saved-model"}) || result.Warning != "" {
				t.Fatalf("response = %d %s", w.Code, w.Body)
			}
		})
	}
}

func TestDesktopModelsSlowDiscoveryDoesNotBlockSavedSettings(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	controller, factory := codexDesktop, codexDesktopClientFactory
	recommendations, access, cloud := codexDesktopRecommendations, codexDesktopAccessState, codexDesktopCloudModels
	t.Cleanup(func() {
		codexDesktop, codexDesktopClientFactory = controller, factory
		codexDesktopRecommendations, codexDesktopAccessState, codexDesktopCloudModels = recommendations, access, cloud
	})
	codexDesktop = &fakeCodexDesktopController{installed: true}
	codexDesktopClientFactory = func() (*api.Client, error) {
		return api.NewClient(&url.URL{Scheme: "http", Host: "127.0.0.1"}, http.DefaultClient), nil
	}
	started := make(chan struct{})
	codexDesktopRecommendations = func(ctx context.Context) ([]api.ModelRecommendation, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	codexDesktopAccessState = func(ctx context.Context) (proxy.ClaudeDesktopAccessState, error) {
		return proxy.ClaudeDesktopAccessState{}, ctx.Err()
	}
	codexDesktopCloudModels = func(ctx context.Context) ([]string, error) { return nil, ctx.Err() }

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := httptest.NewRequest(http.MethodGet, "/?catalog=true", nil).WithContext(ctx)
	r.SetPathValue("integration", "chatgpt")
	w := httptest.NewRecorder()
	handler := desktopModelSettingsHandler()
	done := make(chan struct{})
	go func() { defer close(done); handler.ServeHTTP(w, r) }()
	t.Cleanup(func() { cancel(); <-done })
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("discovery did not start")
	}
	summary := httptest.NewRequest(http.MethodGet, "/?catalog=false", nil)
	summary.SetPathValue("integration", "chatgpt")
	summaryDone := make(chan struct{})
	go func() {
		defer close(summaryDone)
		handler.ServeHTTP(httptest.NewRecorder(), summary)
	}()
	select {
	case <-summaryDone:
	case <-time.After(time.Second):
		t.Fatal("saved settings waited for model discovery")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("discovery did not honor cancellation")
	}
	if w.Code != http.StatusGatewayTimeout {
		t.Fatalf("canceled discovery returned %d", w.Code)
	}
}
