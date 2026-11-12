package launch

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/internal/proxy"
	"github.com/spf13/cobra"
)

func claudeDesktopCLIFixture(t *testing.T, state proxy.ClaudeDesktopAccessState) {
	t.Helper()
	claudeDesktopEditorClient(t)
	withClaudeDesktopPlatform(t, "darwin")
	if err := os.MkdirAll(filepath.Join(os.Getenv("HOME"), "Applications", "Claude.app"), 0o700); err != nil {
		t.Fatal(err)
	}
	available := append(proxy.SelectClaudeDesktopModels(nil, []string{"local-one", "local-two"}),
		proxy.ClaudeDesktopModelsFromRecommendations([]api.ModelRecommendation{{Model: "test:cloud", RequiredPlan: "pro"}})...)
	gateway, err := proxy.NewClaudeDesktop(proxy.ClaudeDesktopConfig{
		RefreshLocalModels: true,
		ListenAddr:         "127.0.0.1:0", OllamaURL: os.Getenv("OLLAMA_HOST"), Model: "local-one",
		Models: proxy.MapClaudeDesktopModels(available, map[string]string{"claude-sonnet-5": "local-one"}),
		RefreshModels: func(context.Context, []proxy.ClaudeDesktopModel) ([]proxy.ClaudeDesktopModel, error) {
			mappings := ClaudeDesktopModelMappings()
			if len(mappings) == 0 {
				mappings = map[string]string{"claude-sonnet-5": "local-one"}
			}
			return proxy.MapClaudeDesktopModels(available, mappings), nil
		},
		ListLocalModels:    func(context.Context) ([]string, error) { return []string{"local-one", "local-two"}, nil },
		ResolveAccessState: func(context.Context) (proxy.ClaudeDesktopAccessState, error) { return state, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := gateway.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = gateway.Close(context.Background()) })
	oldProbe, oldCatalog := claudeDesktopProbeGateway, claudeDesktopFetchCatalog
	t.Cleanup(func() { claudeDesktopProbeGateway, claudeDesktopFetchCatalog = oldProbe, oldCatalog })
	endpoint := "http://" + gateway.Addr()
	claudeDesktopProbeGateway = func(ctx context.Context, _ string) error { return proxy.ProbeClaudeDesktop(ctx, endpoint) }
	claudeDesktopFetchCatalog = func(ctx context.Context, client *http.Client, _, key string) ([]proxy.ClaudeDesktopCatalogModel, error) {
		return proxy.FetchClaudeDesktopCatalog(ctx, client, endpoint, key)
	}
}

func TestClaudeDesktopCLIEditorLaunchAndRestore(t *testing.T) {
	claudeDesktopCLIFixture(t, proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff})
	path := filepath.Join(os.Getenv("HOME"), ".ollama", "config.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"future":9007199254740993,"integrations":{"other":{"models":["keep"]},"claude-desktop":{"automode":true,"future":9007199254740993}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	DefaultClaudeDesktopMappingSelector = func(rows []ClaudeDesktopMapping, _ string) (ClaudeDesktopMappingChoice, error) {
		if len(rows) != 5 || rows[2].DefaultModel != "local-one" {
			t.Fatalf("fresh editor = %+v", rows)
		}
		return ClaudeDesktopMappingChoice{}, nil
	}
	running, opens := false, 0
	withClaudeDesktopProcessHooks(t, func() bool { return running }, func() error { t.Fatal("unchanged launch quit Claude"); return nil }, func() error { opens++; return nil })
	cmd := LaunchCmd(func(*cobra.Command, []string) error { return nil }, nil)
	cmd.SetArgs([]string{"claude-desktop", "--config"})
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	if opens != 1 || len(ClaudeDesktopModelMappings()) != 5 {
		t.Fatalf("fresh setup opens/mappings = %d/%v", opens, ClaudeDesktopModelMappings())
	}
	data, _ := os.ReadFile(path)
	if strings.Count(string(data), "9007199254740993") != 2 || !strings.Contains(string(data), "keep") {
		t.Fatalf("setup lost unrelated settings: %s", data)
	}
	paths, _ := claudeDesktopConfigPaths()
	profile := claudeDesktopReadJSON(t, paths.profile)
	if profile["inferenceGatewayBaseUrl"] != claudeDesktopGatewayBaseURL || profile["autoModeEnabled"] != false {
		t.Fatalf("profile = %v", profile)
	}
	running = true
	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "claude-desktop"}); err != nil {
		t.Fatal(err)
	}
	if opens != 2 {
		t.Fatalf("unchanged launch opened Claude %d times", opens)
	}
	running = false
	claudeDesktopProbeGateway = func(context.Context, string) error { t.Fatal("restore contacted the gateway"); return nil }
	if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "claude-desktop", Restore: true}); err != nil {
		t.Fatal(err)
	}
	if readClaudeDesktopDeploymentMode(paths.normalConfig) != "1p" || len(ClaudeDesktopModelMappings()) != 5 {
		t.Fatal("restore did not recover the normal profile and retain model choices")
	}
}

func TestClaudeDesktopCLIDeclinedRestartPreservesSettings(t *testing.T) {
	claudeDesktopCLIFixture(t, proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff})
	if err := SaveClaudeDesktopModelMappings(map[string]string{"claude-sonnet-5": "local-one"}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(os.Getenv("HOME"), ".ollama", "config.json")
	before, _ := os.ReadFile(path)
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
	withClaudeDesktopProcessHooks(t, func() bool { return true }, func() error { t.Fatal("declined restart quit Claude"); return nil }, func() error { t.Fatal("declined restart opened Claude"); return nil })
	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "claude-desktop", ModelOverride: "local-two"})
	after, _ := os.ReadFile(path)
	if !errors.Is(err, ErrCancelled) || string(before) != string(after) {
		t.Fatalf("declined restart = %v; changed settings: %t", err, string(before) != string(after))
	}
}

func TestClaudeDesktopCLICloudValidation(t *testing.T) {
	for _, state := range []proxy.ClaudeDesktopAccessState{
		{Cloud: proxy.ClaudeDesktopCloudOff},
		{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut},
		{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "free"},
		{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountUnknown},
		{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"},
	} {
		t.Run(string(state.Cloud)+"/"+string(state.Account)+"/"+state.Plan, func(t *testing.T) {
			claudeDesktopCLIFixture(t, state)
			before := map[string]string{"claude-sonnet-5": "local-one"}
			if err := SaveClaudeDesktopModelMappings(before); err != nil {
				t.Fatal(err)
			}
			withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { t.Fatal("unavailable cloud opened Claude"); return nil })
			// Generic launcher sign-in succeeds; the gateway then observes the
			// current policy or account change and must reject the selection.
			err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "claude-desktop", ModelOverride: "test:cloud", ConfigureOnly: true})
			if state.Plan == "pro" {
				if err != nil || ClaudeDesktopModelMappings()["claude-sonnet-5"] != "test:cloud" {
					t.Fatalf("available cloud = %v, saved mappings = %v", err, ClaudeDesktopModelMappings())
				}
			} else if err == nil || !maps.Equal(ClaudeDesktopModelMappings(), before) {
				t.Fatalf("unavailable cloud = %v, saved mappings = %v", err, ClaudeDesktopModelMappings())
			}
		})
	}
}

func TestClaudeDesktopCLIProfileFailureRollsBackSelection(t *testing.T) {
	claudeDesktopCLIFixture(t, proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff})
	before := map[string]string{"claude-sonnet-5": "local-one"}
	if err := SaveClaudeDesktopModelMappings(before); err != nil {
		t.Fatal(err)
	}
	paths, _ := claudeDesktopConfigPaths()
	if err := os.MkdirAll(paths.profile, 0o700); err != nil {
		t.Fatal(err)
	}
	withClaudeDesktopProcessHooks(t, func() bool { return false }, func() error { return nil }, func() error { t.Fatal("failed profile setup opened Claude"); return nil })
	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "claude-desktop", ModelOverride: "local-two"})
	if err == nil || !maps.Equal(ClaudeDesktopModelMappings(), before) {
		t.Fatalf("profile failure = %v; saved mappings = %v", err, ClaudeDesktopModelMappings())
	}
	catalog, err := claudeDesktopFetchCatalog(context.Background(), nil, claudeDesktopGatewayBaseURL, "ollama")
	if err != nil || !maps.Equal(claudeDesktopCatalogMappings(catalog), before) {
		t.Fatalf("runtime rollback = %v, %v", catalog, err)
	}
}

func TestClaudeDesktopCLIRollbackPreservesLaterEdits(t *testing.T) {
	for _, newerSelection := range []bool{false, true} {
		t.Run(fmt.Sprintf("newer_selection_%t", newerSelection), func(t *testing.T) {
			claudeDesktopCLIFixture(t, proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff})
			before := map[string]string{"claude-sonnet-5": "local-one"}
			if err := SaveClaudeDesktopModelMappings(before); err != nil {
				t.Fatal(err)
			}
			newer := map[string]string{"claude-opus-5": "local-one"}
			err := applyClaudeDesktopMappings(context.Background(), map[string]string{"claude-sonnet-5": "local-two"}, func() error {
				var err error
				if newerSelection {
					err = SaveClaudeDesktopModelMappings(newer)
				} else {
					err = config.SaveIntegration("other", []string{"later-edit"})
				}
				if err != nil {
					t.Fatal(err)
				}
				return errors.New("profile failed")
			})
			if newerSelection {
				if err == nil || !maps.Equal(ClaudeDesktopModelMappings(), newer) {
					t.Fatalf("rollback overwrote a newer selection: %v", err)
				}
			} else if err == nil || !maps.Equal(ClaudeDesktopModelMappings(), before) || config.IntegrationModel("other") != "later-edit" {
				t.Fatalf("rollback lost prior selection or later unrelated edit: %v", err)
			}
		})
	}
}
