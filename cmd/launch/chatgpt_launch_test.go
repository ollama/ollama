package launch

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/internal/proxy"
	"github.com/spf13/cobra"
)

type parityTransport struct{ t *testing.T }

func (p parityTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	body := `{"models":[{"name":"local-a"},{"name":"local-b"}]}`
	if r.URL.Path == "/api/show" {
		body = `{"capabilities":["completion","tools"],"details":{"family":"fixture"}}`
	} else if r.URL.Path != "/api/tags" {
		p.t.Fatalf("unexpected model request: %s", r.URL.Path)
	}
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
}

type parityButton struct {
	installed, running, configured bool
	primary                        string
	models                         []string
}

func (p *parityButton) Installed() bool        { return p.installed }
func (p *parityButton) Running() bool          { return p.running }
func (p *parityButton) OllamaConfigured() bool { return p.configured }
func (p *parityButton) Onboard() error         { return nil }
func (p *parityButton) UseOllamaFromDesktop(primary string, models []LaunchModel, confirmed bool) error {
	if p.running && !confirmed {
		return ErrCodexAppRestartConfirmationRequired
	}
	p.primary, p.models = primary, ChatGPTModelNames(models)
	p.configured = true
	return nil
}
func (p *parityButton) RestoreFromDesktop(bool) error { p.configured = false; return nil }

func parityFixture(t *testing.T, access proxy.ClaudeDesktopAccessState) (*CodexApp, *ChatGPTModelSource, *int) {
	t.Helper()
	setTestHome(t, t.TempDir())
	withCodexAppPlatform(t, "darwin")
	withInteractiveSession(t, true)
	withLauncherHooks(t)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:1") // A value only; routing and API transport are mocked.
	oldStat, oldProfile, oldFactory := codexAppStat, codexAppProfileIsRunning, chatGPTModelSourceFactory
	codexAppStat = func(string) (os.FileInfo, error) { return nil, os.ErrNotExist }
	codexAppProfileIsRunning = func(context.Context) (bool, error) { return false, nil }
	t.Cleanup(func() {
		codexAppStat, codexAppProfileIsRunning, chatGPTModelSourceFactory = oldStat, oldProfile, oldFactory
	})
	opens := 0
	withCodexAppProcessHooks(t, func() bool { return false }, func() error { t.Fatal("unexpected quit"); return nil }, func() error { opens++; return nil })
	codexAppCanOpenID = func(context.Context) bool { return true }
	withCodexAppRouterHealth(t, func() error { return nil })
	client := api.NewClient(&url.URL{Scheme: "http", Host: "fixture.invalid"}, &http.Client{Transport: parityTransport{t}})
	source := &ChatGPTModelSource{Client: client, Attempts: 1,
		Recommendations: func(context.Context) ([]api.ModelRecommendation, error) {
			return []api.ModelRecommendation{
				{Model: "kimi-k3", RequiredPlan: "pro", ContextLength: 32768, MaxOutputTokens: 2048},
				{Model: "glm-5.3-flash", RequiredPlan: "pro", ContextLength: 32768, MaxOutputTokens: 2048},
				{Model: "local-a", ContextLength: 32768},
			}, nil
		},
		CloudModels: func(context.Context) ([]string, error) { return []string{"kimi-k3:cloud", "glm-5.3-flash:cloud"}, nil },
		AccessState: func(context.Context) (proxy.ClaudeDesktopAccessState, error) { return access, nil },
	}
	chatGPTModelSourceFactory = func(*api.Client) *ChatGPTModelSource { return source }
	return &CodexApp{}, source, &opens
}

func TestChatGPTButtonCLIParity(t *testing.T) {
	for _, tc := range []struct {
		name    string
		access  proxy.ClaudeDesktopAccessState
		saved   []string
		primary string
	}{
		{"signed out", proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}, nil, "local-a"},
		{"pro", proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedIn, Plan: "pro"}, nil, "glm-5.3-flash:cloud"},
		{"cloud off", proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOff}, nil, "local-a"},
		{"existing choices", proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut}, []string{"local-b", "local-a"}, "local-a"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			app, source, opens := parityFixture(t, tc.access)
			if tc.saved != nil {
				if err := config.SaveIntegration(chatGPTIntegrationName, tc.saved); err != nil {
					t.Fatal(err)
				}
			}
			if err := config.SaveIntegration("codex", []string{"terminal-choice"}); err != nil {
				t.Fatal(err)
			}
			button := &parityButton{installed: true}
			load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
				return source.Resolve(ctx, selected, true)
			}
			if err := StartChatGPT(t.Context(), button, load, false); err != nil {
				t.Fatal(err)
			}
			cmd := LaunchCmd(func(*cobra.Command, []string) error { return nil }, nil)
			cmd.SetArgs([]string{"chatgpt"})
			if err := cmd.Execute(); err != nil {
				t.Fatal(err)
			}
			models, err := codexAppRoutingModels()
			if err != nil || app.CurrentModel() != button.primary || button.primary != tc.primary || !slices.Equal(models, button.models) || *opens != 1 {
				t.Fatalf("button=%s/%v CLI=%s/%v opens=%d err=%v", button.primary, button.models, app.CurrentModel(), models, *opens, err)
			}
			if !slices.Equal(config.IntegrationModels(chatGPTIntegrationName), tc.saved) || !slices.Equal(config.IntegrationModels("codex"), []string{"terminal-choice"}) {
				t.Fatal("plain ChatGPT launch changed saved model choices")
			}
			running := true
			restarts := 0
			codexAppIsRunning = func(context.Context) bool { return running }
			codexAppQuitApp = func(context.Context) error { running = false; restarts++; return nil }
			DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return true, nil }
			cmd = LaunchCmd(func(*cobra.Command, []string) error { return nil }, nil)
			cmd.SetArgs([]string{"chatgpt"})
			if err := cmd.Execute(); err != nil {
				t.Fatal(err)
			}
			if restarts != 1 || *opens != 2 || app.CurrentModel() != button.primary || !slices.Equal(config.IntegrationModels(chatGPTIntegrationName), tc.saved) {
				t.Fatal("confirmed restart changed model choices or did not restart once")
			}
			if err := app.RestoreFromDesktop(false); err != nil {
				t.Fatal(err)
			}
			cmd = LaunchCmd(func(*cobra.Command, []string) error { return nil }, nil)
			cmd.SetArgs([]string{"chatgpt"})
			if err := cmd.Execute(); err != nil {
				t.Fatal(err)
			}
			if app.CurrentModel() != button.primary || !slices.Equal(config.IntegrationModels(chatGPTIntegrationName), tc.saved) {
				t.Fatal("reconnect changed defaults or saved choices")
			}
		})
	}
}

func TestChatGPTButtonCLIConsentAndDiscoveryFailure(t *testing.T) {
	app, source, opens := parityFixture(t, proxy.ClaudeDesktopAccessState{Cloud: proxy.ClaudeDesktopCloudOn, Account: proxy.ClaudeDesktopAccountSignedOut})
	button := &parityButton{installed: true, running: true}
	load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
		return source.Resolve(ctx, selected, true)
	}
	if err := StartChatGPT(t.Context(), button, load, false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("button consent=%v", err)
	}
	codexAppIsRunning = func(context.Context) bool { return true }
	DefaultConfirmPrompt = func(string, ConfirmOptions) (bool, error) { return false, nil }
	client, _ := newLauncherClient(defaultLaunchPolicy(true, false))
	if err := client.launchChatGPT(t.Context(), app); !errors.Is(err, ErrCancelled) || *opens != 0 {
		t.Fatalf("CLI cancel=%v opens=%d", err, *opens)
	}
	failure := errors.New("fixture inventory unavailable")
	failedLoad := func(context.Context, []string) (string, []LaunchModel, error) { return "", nil, failure }
	if err := StartChatGPT(t.Context(), button, failedLoad, true); !errors.Is(err, failure) || button.configured {
		t.Fatalf("discovery failure changed button state: %v", err)
	}
}
