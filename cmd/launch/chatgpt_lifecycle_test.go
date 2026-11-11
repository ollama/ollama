package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/spf13/cobra"
)

// Exercises the production configuration writers. Discovery, native catalogs,
// process probes, quit, open, and router health are all isolated fixtures.
func chatGPTNativeFixture(t *testing.T) (string, *CodexApp, *ChatGPTModelSource, *int) {
	t.Helper()
	_, _, source := chatGPTFixture(t)
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	setTestHome(t, home)
	withCodexAppPlatform(t, "darwin")
	withCodexAppRouterHealth(t, func() error { return nil })
	// Only a configuration value; router health and all API calls are mocked.
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:1")
	oldStat, oldCanOpen, oldProfileRunning := codexAppStat, codexAppCanOpenID, codexAppProfileIsRunning
	codexAppStat = func(string) (os.FileInfo, error) { return nil, os.ErrNotExist }
	codexAppCanOpenID = func(context.Context) bool { return true }
	codexAppProfileIsRunning = func(context.Context) (bool, error) { return false, nil }
	t.Cleanup(func() {
		codexAppStat, codexAppCanOpenID, codexAppProfileIsRunning = oldStat, oldCanOpen, oldProfileRunning
	})
	opens := 0
	withCodexAppProcessHooks(t, func() bool { return false }, func() error { t.Fatal("stopped fixture must not quit an app"); return nil }, func() error { opens++; return nil })
	return home, &CodexApp{}, source, &opens
}

func TestChatGPTAppCLISharedLifecycleFixture(t *testing.T) {
	home, app, source, opens := chatGPTNativeFixture(t)
	configPath := filepath.Join(home, ".codex", "config.toml")
	original := "model = \"gpt-native\"\nmodel_provider = \"openai\"\napproval_policy = \"on-request\"\n\n[profiles.terminal]\nmodel = \"terminal-model\"\n\n[model_providers.custom]\nbase_url = \"https://provider.invalid/v1\"\n"
	files := map[string]string{
		configPath: original,
		filepath.Join(home, ".codex", "auth.json"):                 `{"auth_mode":"chatgpt","tokens":{"access_token":"fixture-token"}}`,
		filepath.Join(home, ".codex", "history.jsonl"):             "fixture history\n",
		filepath.Join(home, ".codex", "sessions", "fixture.jsonl"): "fixture conversation\n",
	}
	for path, content := range files {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for name, models := range map[string][]string{"codex": {"terminal-model"}, "claude-desktop": {"claude-choice"}} {
		if err := config.SaveIntegration(name, models); err != nil {
			t.Fatal(err)
		}
	}
	selected := []string{"custom-secondary", "custom-primary"}
	load := func(ctx context.Context, names []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, names, false)
	}
	if err := ApplyChatGPTModels(t.Context(), app, load, ChatGPTApplyRequest{Models: selected, OpenWhenStopped: true, Onboard: true}); err != nil {
		t.Fatal(err)
	}
	if *opens != 1 {
		t.Fatalf("initial app opens=%d, want 1 mocked open", *opens)
	}
	assertCatalog := func(primary string) {
		t.Helper()
		saved, err := LoadChatGPTConfiguration()
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(saved.Models, selected) || saved.Primary != primary || app.CurrentModel() != primary {
			t.Fatalf("saved=%+v live=%s, want models=%v primary=%s", saved, app.CurrentModel(), selected, primary)
		}
		data, err := os.ReadFile(mustCodexAppModelCatalogPath(t))
		if err != nil {
			t.Fatal(err)
		}
		var catalog struct {
			Models []map[string]any `json:"models"`
		}
		if err := json.Unmarshal(data, &catalog); err != nil {
			t.Fatal(err)
		}
		if got := catalogSlugs(catalog.Models); !slices.Equal(got, append(slices.Clone(selected), "gpt-5.6-sol")) {
			t.Fatalf("catalog=%v, want selected subset followed by native models", got)
		}
	}
	assertCatalog("custom-primary")
	for _, request := range []IntegrationLaunchRequest{
		{Name: "chatgpt", ConfigureOnly: true},
		{Name: "codex-app", ModelOverride: "custom-secondary", ConfigureOnly: true},
	} {
		if err := LaunchIntegration(t.Context(), request); err != nil {
			t.Fatal(err)
		}
	}
	assertCatalog("custom-secondary")
	if *opens != 1 {
		t.Fatalf("configure-only opened a stopped app: opens=%d", *opens)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, append(data, []byte("\n[user_edit]\nkeep = true\n")...), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "chatgpt", Restore: true}); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{`model = "gpt-native"`, `model_provider = "openai"`, `[profiles.terminal]`, `[model_providers.custom]`, `[user_edit]`} {
		if !strings.Contains(string(data), value) {
			t.Fatalf("restore lost %s", value)
		}
	}
	if strings.Contains(string(data), "/api/codex") {
		t.Fatal("disconnect retained the managed gateway in the native profile")
	}
	if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "chatgpt", ConfigureOnly: true}); err != nil {
		t.Fatal(err)
	}
	assertCatalog("custom-secondary")
	for path, want := range files {
		if path == configPath {
			continue
		}
		got, err := os.ReadFile(path)
		if err != nil || string(got) != want {
			t.Fatalf("fixture credential/history changed at %s", path)
		}
	}
	for name, want := range map[string][]string{"codex": {"terminal-model"}, "claude-desktop": {"claude-choice"}} {
		if got := config.IntegrationModels(name); !slices.Equal(got, want) {
			t.Fatalf("%s choices=%v, want=%v", name, got, want)
		}
	}
}

func TestChatGPTReviewCommandRelaunchSkipsListEditor(t *testing.T) {
	for _, configure := range []bool{false, true} {
		t.Run(map[bool]string{false: "plain relaunch", true: "explicit list edit"}[configure], func(t *testing.T) {
			_, app, _, opens := chatGPTNativeFixture(t)
			withInteractiveSession(t, true)
			withLauncherHooks(t)
			selected := []string{"custom-primary", "custom-secondary"}
			before := &config.IntegrationConfig{Models: selected, Primary: "custom-secondary"}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
				t.Fatal(err)
			}
			if err := app.ConfigureWithModels(before.Primary, testLaunchModels(selected...)); err != nil {
				t.Fatal(err)
			}
			pickerCalls := 0
			DefaultMultiSelectorWithUpdates = nil
			DefaultMultiSelector = func(_ string, _ []SelectionItem, checked []string) ([]string, error) {
				pickerCalls++
				return []string{"custom-secondary"}, nil
			}
			command := LaunchCmd(func(*cobra.Command, []string) error { return nil }, func(*cobra.Command) { t.Fatal("unexpected launcher menu") })
			command.SetOut(io.Discard)
			command.SetErr(io.Discard)
			args := []string{"chatgpt"}
			if configure {
				args = append(args, "--config")
			}
			command.SetArgs(args)
			if err := command.ExecuteContext(t.Context()); err != nil {
				t.Fatal(err)
			}
			wantPicker, wantOpens := 0, 1
			if configure {
				wantPicker, wantOpens = 1, 0
				selected = []string{"custom-secondary"}
			}
			saved, err := LoadChatGPTConfiguration()
			if err != nil {
				t.Fatal(err)
			}
			if pickerCalls != wantPicker || *opens != wantOpens || !slices.Equal(saved.Models, selected) || saved.Primary != "custom-secondary" {
				t.Fatalf("command picker=%d opens=%d saved=%+v, want picker=%d opens=%d models=%v", pickerCalls, *opens, saved, wantPicker, wantOpens, selected)
			}
		})
	}
}

func TestChatGPTReviewDisabledCloudCannotIgnoreExplicitModel(t *testing.T) {
	_, _, _, opens := chatGPTNativeFixture(t)
	withInteractiveSession(t, true)
	withLauncherHooks(t)
	DefaultMultiSelectorWithUpdates = nil
	DefaultMultiSelector = func(string, []SelectionItem, []string) ([]string, error) { return nil, ErrCancelled }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/status" {
			t.Errorf("unexpected fixture request %s", r.URL.Path)
		}
		io.WriteString(w, `{"cloud":{"disabled":true}}`)
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	before := &config.IntegrationConfig{Models: []string{"custom-primary"}, Primary: "custom-primary"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
		t.Fatal(err)
	}
	command := LaunchCmd(func(*cobra.Command, []string) error { return nil }, func(*cobra.Command) { t.Fatal("unexpected launcher menu") })
	command.SetOut(io.Discard)
	command.SetErr(io.Discard)
	command.SetArgs([]string{"chatgpt", "--model", "fixture:cloud"})
	err := command.ExecuteContext(t.Context())
	if err == nil || !strings.Contains(err.Error(), "cloud is disabled") {
		t.Fatalf("explicit cloud model was ignored: error=%v", err)
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if saved.Primary != before.Primary || !slices.Equal(saved.Models, before.Models) || *opens != 0 {
		t.Fatalf("disabled cloud override changed state: saved=%+v opens=%d", saved, *opens)
	}
}

func TestChatGPTReviewCancelledRestoreDoesNotWriteProfile(t *testing.T) {
	_, app, _, _ := chatGPTNativeFixture(t)
	withLauncherHooks(t)
	if err := app.ConfigureWithModels("custom-primary", testLaunchModels("custom-primary", "custom-secondary")); err != nil {
		t.Fatal(err)
	}
	path, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	codexAppIsRunning = func(context.Context) bool { return true }
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
	restorePolicy := withLaunchConfirmPolicy(launchConfirmPolicy{})
	defer restorePolicy()
	err = app.Restore()
	if !errors.Is(err, ErrCancelled) {
		t.Errorf("restore error=%v, want cancellation", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Fatal("cancelled restart had already changed the ChatGPT profile")
	}
}

func TestChatGPTReviewUnavailableRouterKeepsWorkingProfile(t *testing.T) {
	_, app, source, opens := chatGPTNativeFixture(t)
	before := &config.IntegrationConfig{Models: []string{"custom-primary", "custom-secondary"}, Primary: "custom-secondary"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
		t.Fatal(err)
	}
	if err := app.ConfigureWithModels(before.Primary, testLaunchModels(before.Models...)); err != nil {
		t.Fatal(err)
	}
	path, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	profile, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	unavailable := errors.New("fixture server unavailable; start Ollama and retry")
	withCodexAppRouterHealth(t, func() error { return unavailable })
	load := func(ctx context.Context, names []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, names, false)
	}
	err = ApplyChatGPTModels(t.Context(), app, load, ChatGPTApplyRequest{Models: []string{"custom-primary"}})
	if !errors.Is(err, unavailable) {
		t.Errorf("apply error=%v, want unavailable-router recovery", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := LoadChatGPTConfiguration()
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(profile) || !app.OllamaConfigured() || !slices.Equal(saved.Models, before.Models) || saved.Primary != before.Primary || *opens != 0 {
		t.Fatalf("a pre-write router failure changed the existing setup: saved=%+v configured=%t opens=%d", saved, app.OllamaConfigured(), *opens)
	}
}

func TestChatGPTFinalFallbackRestartRequiresReadyRouter(t *testing.T) {
	for _, running := range []bool{false, true} {
		name := "stopped"
		if running {
			name = "running"
		}
		t.Run(name, func(t *testing.T) {
			_, app, _, _ := chatGPTNativeFixture(t)
			before := &config.IntegrationConfig{Models: []string{"custom-primary", "custom-secondary"}, Primary: "custom-secondary"}
			if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
				t.Fatal(err)
			}
			if err := app.ConfigureWithModels(before.Primary, testLaunchModels(before.Models...)); err != nil {
				t.Fatal(err)
			}
			path, err := codexConfigPath()
			if err != nil {
				t.Fatal(err)
			}
			paths := []string{path, mustCodexAppModelCatalogPath(t), codexAppRoutingCatalogPathForConfig(path)}
			files := make(map[string][]byte, len(paths))
			for _, path := range paths {
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				files[path] = data
			}
			quits, opens := 0, 0
			withCodexAppProcessHooks(t, func() bool { return running }, func() error { quits++; running = false; return nil }, func() error { opens++; running = true; return nil })
			unavailable := errors.New("fixture routing unavailable; restart Ollama and try again")
			withCodexAppRouterHealth(t, func() error { return unavailable })
			load := func(context.Context, []string) (string, []LaunchModel, error) {
				return "", nil, errors.New("fixture inventory unavailable")
			}
			err = ApplyChatGPTModels(t.Context(), app, load, ChatGPTApplyRequest{UseSavedSelection: true, OpenWhenStopped: true, RestartOnInventoryFailure: true, RestartConfirmed: true})
			if !errors.Is(err, unavailable) || quits != 0 || opens != 0 {
				t.Fatalf("fallback interrupted ChatGPT with routing unavailable: error=%v quits=%d opens=%d", err, quits, opens)
			}
			for path, before := range files {
				after, err := os.ReadFile(path)
				if err != nil || !slices.Equal(before, after) {
					t.Fatalf("routing failure changed %s: %v", path, err)
				}
			}
			saved, err := LoadChatGPTConfiguration()
			if err != nil || saved.Primary != before.Primary || !slices.Equal(saved.Models, before.Models) || !app.OllamaConfigured() {
				t.Fatalf("routing failure changed the shared setup: saved=%+v error=%v", saved, err)
			}
		})
	}
}

func TestChatGPTFinalFallbackRestartRejectsStaleHost(t *testing.T) {
	_, app, _, _ := chatGPTNativeFixture(t)
	before := &config.IntegrationConfig{Models: []string{"custom-primary", "custom-secondary"}, Primary: "custom-secondary"}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
		t.Fatal(err)
	}
	if err := app.ConfigureWithModels(before.Primary, testLaunchModels(before.Models...)); err != nil {
		t.Fatal(err)
	}
	path, err := codexConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	profile, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Configuration-only ports: readiness and all native process actions are mocked.
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:2")
	if !app.OllamaConfigured() || app.CurrentModel() != "" {
		t.Fatal("fixture did not create a recognized but outdated connection")
	}
	running, quits, opens := true, 0, 0
	withCodexAppProcessHooks(t, func() bool { return running }, func() error { quits++; running = false; return nil }, func() error { opens++; running = true; return nil })
	load := func(context.Context, []string) (string, []LaunchModel, error) {
		return "", nil, errors.New("fixture inventory unavailable")
	}
	err = ApplyChatGPTModels(t.Context(), app, load, ChatGPTApplyRequest{UseSavedSelection: true, OpenWhenStopped: true, RestartOnInventoryFailure: true, RestartConfirmed: true})
	if err == nil || !strings.Contains(err.Error(), "reconfigure") || quits != 0 || opens != 0 {
		t.Fatalf("fallback restarted with a stale connection: error=%v quits=%d opens=%d", err, quits, opens)
	}
	after, readErr := os.ReadFile(path)
	saved, saveErr := LoadChatGPTConfiguration()
	if readErr != nil || saveErr != nil || !slices.Equal(after, profile) || saved.Primary != before.Primary || !slices.Equal(saved.Models, before.Models) || !app.OllamaConfigured() {
		t.Fatalf("stale-connection failure changed the shared setup: saved=%+v read=%v preferences=%v", saved, readErr, saveErr)
	}
}

type chatGPTUnavailableTransport struct{}

func (chatGPTUnavailableTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("fixture inventory unavailable")
}

func TestChatGPTPublicationCLIPreservesGracefulRestart(t *testing.T) {
	for _, fallback := range []bool{false, true} {
		for _, cancelRestart := range []bool{false, true} {
			t.Run(fmt.Sprintf("fallback=%t/cancel=%t", fallback, cancelRestart), func(t *testing.T) {
				home, app, source, opens := chatGPTNativeFixture(t)
				withInteractiveSession(t, true)
				withLauncherHooks(t)
				DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return true, nil }
				before := &config.IntegrationConfig{Models: []string{"custom-primary", "custom-secondary"}, Primary: "custom-primary"}
				if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, nil, before); err != nil {
					t.Fatal(err)
				}
				if err := app.ConfigureWithModels(before.Primary, testLaunchModels(before.Models...)); err != nil {
					t.Fatal(err)
				}
				path := filepath.Join(home, ".codex", "config.toml")
				profile, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				cancel := make(chan struct{})
				oldSpinner, oldSleep := DefaultSpinner, codexAppSleep
				stopped := false
				DefaultSpinner = func(string) *Spinner { return NewSpinner(func() { stopped = true }, cancel) }
				t.Cleanup(func() { DefaultSpinner, codexAppSleep = oldSpinner, oldSleep })
				running, quits := true, 0
				withCodexAppProcessHooks(t, func() bool { return running }, func() error { quits++; return nil }, func() error { *opens++; return nil })
				codexAppQuitApp = func(ctx context.Context) error {
					if _, bounded := ctx.Deadline(); bounded {
						t.Fatal("interactive CLI restart inherited the desktop quit deadline")
					}
					quits++
					return nil
				}
				codexAppSleep = func(time.Duration) {
					if cancelRestart {
						close(cancel)
					} else {
						running = false
					}
				}
				request := IntegrationLaunchRequest{Name: "chatgpt", ModelOverride: "custom-secondary"}
				if fallback {
					request.ModelOverride = ""
					source.Recommendations = func(context.Context) ([]api.ModelRecommendation, error) {
						return nil, errors.New("fixture inventory unavailable")
					}
					source.Client = api.NewClient(&url.URL{Scheme: "http", Host: "fixture.invalid"}, &http.Client{Transport: chatGPTUnavailableTransport{}})
				}
				err = LaunchIntegration(t.Context(), request)
				if cancelRestart {
					if !errors.Is(err, ErrCancelled) || !stopped || quits != 1 || *opens != 0 {
						t.Fatalf("cancel err=%v stopped=%t quit/open=%d/%d", err, stopped, quits, *opens)
					}
					after, readErr := os.ReadFile(path)
					saved, saveErr := LoadChatGPTConfiguration()
					if readErr != nil || saveErr != nil || !slices.Equal(after, profile) || saved.Primary != before.Primary || !slices.Equal(saved.Models, before.Models) {
						t.Fatalf("cancel changed profile/preferences: saved=%+v read=%v preferences=%v", saved, readErr, saveErr)
					}
				} else if err != nil || !stopped || quits != 1 || *opens != 1 {
					t.Fatalf("restart err=%v stopped=%t quit/open=%d/%d", err, stopped, quits, *opens)
				}
			})
		}
	}
}
