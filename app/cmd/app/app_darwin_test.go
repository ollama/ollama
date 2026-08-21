//go:build darwin

package main

import (
	"context"
	"errors"
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
	code := m.Run()
	claudeModelsLoader = previousLoader
	claudeAccessStateResolver = previousAccessResolver
	claudeLocalModelsResolver = previousLocalResolver
	os.Exit(code)
}

func TestLoadClaudeDesktopModelsUsesAppEndpoint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/experimental/model-recommendations" || r.URL.Query().Get("app") != "claude-desktop" {
			t.Fatalf("request URL = %q", r.URL.String())
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
}

func TestClaudeDesktopDownloadEndpointUsesTypedZipContract(t *testing.T) {
	endpoint := claudeDesktopDownloadEndpoint("http://127.0.0.1:18080/")
	if endpoint != "http://127.0.0.1:18080/download-app?app=claude-desktop&type=mac-zip" {
		t.Fatalf("endpoint = %q", endpoint)
	}
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

type fakeClaudeDesktopController struct {
	configured bool
	installed  bool
	restart    bool
}

func (f *fakeClaudeDesktopController) UsesOllamaGateway() bool { return f.configured }

func (f *fakeClaudeDesktopController) SetInstalledFromDesktop(installed, restart bool) error {
	f.installed = installed
	f.restart = restart
	return nil
}

func (f *fakeClaudeDesktopController) RestoreForShutdown(context.Context) error { return nil }

func TestRestartClaudeDesktopWithModelsPersistsSelection(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	previousInstalled := claudeDesktopInstalled
	previousAddr := claudeProxyListenAddr
	previousDesktop := claudeDesktop
	claudeDesktopInstalled = func() bool { return true }
	claudeProxyListenAddr = "127.0.0.1:0"
	fake := &fakeClaudeDesktopController{configured: true}
	claudeDesktop = fake
	t.Cleanup(func() {
		stopClaudeAppProxy()
		claudeDesktopInstalled = previousInstalled
		claudeProxyListenAddr = previousAddr
		claudeDesktop = previousDesktop
	})

	if err := restartClaudeDesktopWithModels([]string{"kimi-k3:cloud"}); err != nil {
		t.Fatal(err)
	}
	if !fake.installed {
		t.Fatal("expected the Claude profile to be installed")
	}
	if got, want := launch.ClaudeDesktopModels(), []string{"kimi-k3:cloud"}; !slices.Equal(got, want) {
		t.Fatalf("persisted models = %v, want Ollama routes %v", got, want)
	}
}

func TestRestartClaudeDesktopWithModelsCapsSelectionAtLiteralSlots(t *testing.T) {
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

	err := restartClaudeDesktopWithModels([]string{
		"glm-5.2:cloud",
		"kimi-k3:cloud",
		"deepseek-v4-pro",
		"deepseek-v4-flash",
		"gemma4:26b:cloud",
		"qwen3:8b",
	})
	if err == nil || !strings.Contains(err.Error(), "at most 5") {
		t.Fatalf("error = %v, want a clear at most 5 message", err)
	}
	if fake.installed {
		t.Fatal("the Claude profile must not change when the selection exceeds the model limit")
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

func TestLoadClaudeDesktopModelsFallsBackWithoutMLX(t *testing.T) {
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
	if err := setClaudeDesktopConnection(true); err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("setClaudeDesktopConnection error = %v, want missing Claude error", err)
	}
	if err := prepareClaudeDesktopConnection(); err == nil || !strings.Contains(err.Error(), "not installed") {
		t.Fatalf("prepareClaudeDesktopConnection error = %v, want missing Claude error", err)
	}
}

func TestClaudeDesktopConnectionStatusSeparatesConfigurationFromGatewayHealth(t *testing.T) {
	status := claudeDesktopConnectionStatus(true, true, false, errors.New("gateway failed"))
	if !status.Configured || status.Connected || !status.StartFailed {
		t.Fatalf("Claude status = %+v, want configured with failed gateway", status)
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
