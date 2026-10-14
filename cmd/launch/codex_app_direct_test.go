package launch

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

type directAppFixture struct {
	regular, profile bool
	events           []string
	config, auth     string
}

func newDirectAppFixture(t *testing.T) *directAppFixture {
	t.Helper()
	setTestHome(t, t.TempDir())
	withCodexAppPlatform(t, "darwin")
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	f := &directAppFixture{regular: true}
	f.config, _ = codexConfigPath()
	f.auth = filepath.Join(filepath.Dir(f.config), "auth.json")
	if err := os.MkdirAll(filepath.Dir(f.config), 0o700); err != nil {
		t.Fatal(err)
	}
	writeDirectTestFile(t, f.config, "model = \"gpt-original\"\n[projects.saved]\ntrust_level = \"trusted\"\n")
	writeDirectTestFile(t, f.auth, `{"auth_mode":"chatgpt","tokens":{"access_token":"user-owned"}}`)
	withCodexAppProcessHooks(t, func() bool { return f.regular }, func() error {
		f.events = append(f.events, "stop-regular")
		f.regular = false
		return nil
	}, func() error {
		f.events = append(f.events, "open-regular")
		f.regular = true
		return nil
	})
	oldRunning, oldStop, oldOpen := codexAppProfileIsRunning, codexAppStopProfile, codexAppOpenProfile
	oldHealth, oldInstalled := codexAppDirectHealth, codexAppCanOpenID
	codexAppProfileIsRunning = func() bool { return f.profile }
	codexAppStopProfile = func() error {
		f.events = append(f.events, "stop-profile")
		f.profile = false
		return nil
	}
	codexAppOpenProfile = func() error {
		if codexAppDirectCurrentModel() == "" {
			t.Fatal("opened profile without a valid model and catalog")
		}
		f.events = append(f.events, "open-profile")
		f.profile = true
		return nil
	}
	codexAppDirectHealth = func() error { return nil }
	codexAppCanOpenID = func() bool { return true }
	t.Cleanup(func() {
		codexAppProfileIsRunning, codexAppStopProfile, codexAppOpenProfile = oldRunning, oldStop, oldOpen
		codexAppDirectHealth, codexAppCanOpenID = oldHealth, oldInstalled
	})
	return f
}

func writeDirectTestFile(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil {
		t.Fatal(err)
	}
}

func readDirectTestFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestCodexAppDirectConnectUpdateAndDisconnect(t *testing.T) {
	f := newDirectAppFixture(t)
	rootConfig, rootAuth := readDirectTestFile(t, f.config), readDirectTestFile(t, f.auth)
	app := &CodexDesktop{}
	withCodexAppRouterHealth(t, func() error { t.Fatal("direct profile checked the gateway"); return nil })
	if app.Running() {
		t.Fatal("regular ChatGPT window should not require an Ollama restart")
	}
	if err := app.UseOllamaFromDesktop("qwen3.5:latest", testLaunchModels("qwen3.5:latest", "glm-5.3:cloud"), false); err != nil {
		t.Fatal(err)
	}
	path, _ := codexAppDirectConfigPath()
	configText := readDirectTestFile(t, path)
	config, err := codexParseConfig(configText)
	if err != nil {
		t.Fatal(err)
	}
	if config.RootString(codexRootModelProviderKey) != codexAppProfileName || config.ProviderString(codexAppProfileName, "base_url") != "http://127.0.0.1:11434/v1/" || !strings.Contains(configText, "requires_openai_auth = false") {
		t.Fatalf("direct profile config = %s", configText)
	}
	catalog := readDirectTestFile(t, config.RootString(codexRootModelCatalogJSONKey))
	if !strings.Contains(catalog, "qwen3.5:latest") || !strings.Contains(catalog, "glm-5.3:cloud") || strings.Contains(catalog, "gpt-") {
		t.Fatalf("direct catalog = %s", catalog)
	}
	writeDirectTestFile(t, path, configText+"\n[projects.example]\ntrust_level = \"trusted\"\n")
	session := filepath.Join(filepath.Dir(path), "sessions", "saved.jsonl")
	writeDirectTestFile(t, session, "saved conversation\n")
	if err := app.UpdateOllamaModelsFromDesktop("glm-5.3:cloud", testLaunchModels("glm-5.3:cloud"), false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("unconfirmed update = %v", err)
	}
	if codexAppDirectCurrentModel() != "qwen3.5:latest" || !slices.Equal(f.events, []string{"open-profile"}) {
		t.Fatalf("unconfirmed update changed state: %v", f.events)
	}
	if err := app.UpdateOllamaModelsFromDesktop("glm-5.3:cloud", testLaunchModels("glm-5.3:cloud"), true); err != nil {
		t.Fatal(err)
	}
	if codexAppDirectCurrentModel() != "glm-5.3:cloud" || !strings.Contains(readDirectTestFile(t, path), "[projects.example]") {
		t.Fatal("model update lost the selected model or unrelated profile settings")
	}
	if err := app.RestoreFromDesktop(false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("unconfirmed disconnect = %v", err)
	}
	if err := app.RestoreFromDesktop(true); err != nil {
		t.Fatal(err)
	}
	if app.OllamaConfigured() || app.Running() || readDirectTestFile(t, session) != "saved conversation\n" {
		t.Fatal("disconnect must stop only the isolated profile and retain its conversations")
	}
	if !slices.Equal(f.events, []string{"open-profile", "stop-profile", "open-profile", "stop-profile"}) {
		t.Fatalf("profile transitions = %v", f.events)
	}
	if readDirectTestFile(t, f.config) != rootConfig || readDirectTestFile(t, f.auth) != rootAuth || !f.regular {
		t.Fatal("regular profile, credentials, or process changed")
	}
	if err := app.UpdateOllamaModelsFromDesktop("qwen3.5:latest", testLaunchModels("qwen3.5:latest"), false); err != nil {
		t.Fatal(err)
	}
	if f.profile || !strings.Contains(readDirectTestFile(t, path), "[projects.example]") || readDirectTestFile(t, session) != "saved conversation\n" {
		t.Fatal("reconnecting a stopped profile must preserve its settings and history without opening it")
	}
}

func TestCodexAppDirectMigratesLegacyGateway(t *testing.T) {
	f := newDirectAppFixture(t)
	original, auth := readDirectTestFile(t, f.config), readDirectTestFile(t, f.auth)
	app := &CodexDesktop{}
	if err := app.CodexApp.ConfigureWithModels("qwen3.5:latest", testLaunchModels("qwen3.5:latest")); err != nil {
		t.Fatal(err)
	}
	legacy := readDirectTestFile(t, f.config)
	if err := app.RestartFromDesktop(false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("unconfirmed migration = %v", err)
	}
	if len(f.events) != 0 || readDirectTestFile(t, f.config) != legacy {
		t.Fatal("migration changed the running profile before confirmation")
	}
	if err := app.RestartFromDesktop(true); err != nil {
		t.Fatal(err)
	}
	wantConfig, _ := codexParseConfigText(original)
	gotConfig, parseErr := codexParseConfigText(readDirectTestFile(t, f.config))
	if parseErr != nil || !reflect.DeepEqual(gotConfig, wantConfig) || readDirectTestFile(t, f.auth) != auth {
		t.Fatalf("migration config = %q, want %q; auth unchanged = %v", readDirectTestFile(t, f.config), original, readDirectTestFile(t, f.auth) == auth)
	}
	if !slices.Equal(f.events, []string{"stop-regular", "open-regular", "open-profile"}) || codexAppDirectCurrentModel() != "qwen3.5:latest" {
		t.Fatalf("migration transitions = %v, model = %q", f.events, codexAppDirectCurrentModel())
	}
	if codexAppNeedsLegacyRestore() {
		t.Fatal("gateway configuration remains after migration")
	}
}

func TestCodexAppDirectFailuresPreserveRunningProfile(t *testing.T) {
	for _, failure := range []string{"health", "stop", "config"} {
		t.Run(failure, func(t *testing.T) {
			f := newDirectAppFixture(t)
			app := &CodexDesktop{}
			if err := configureCodexAppDirect("qwen3.5:latest", testLaunchModels("qwen3.5:latest")); err != nil {
				t.Fatal(err)
			}
			f.profile = true
			path, _ := codexAppDirectConfigPath()
			switch failure {
			case "health":
				codexAppDirectHealth = func() error { return errors.New("offline") }
			case "stop":
				codexAppStopProfile = func() error { return errors.New("still running") }
			case "config":
				writeDirectTestFile(t, path, "malformed = [")
				codexAppOpenProfile = func() error { f.profile = true; return nil }
			}
			before := readDirectTestFile(t, path)
			if err := app.UseOllamaFromDesktop("changed", testLaunchModels("changed"), true); err == nil {
				t.Fatal("expected profile update failure")
			}
			if !f.profile || !f.regular || readDirectTestFile(t, path) != before {
				t.Fatal("failed update changed the saved profile or left a previously running app closed")
			}
		})
	}
}

func TestCodexAppDirectMigrationRechecksRestartConfirmation(t *testing.T) {
	f := newDirectAppFixture(t)
	app := &CodexDesktop{}
	if err := app.CodexApp.Configure("qwen3.5:latest"); err != nil {
		t.Fatal(err)
	}
	before := readDirectTestFile(t, f.config)
	checks := 0
	codexAppIsRunning = func() bool {
		checks++
		return checks > 1
	}
	if err := app.RestartFromDesktop(false); !errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		t.Fatalf("late-running app restart = %v", err)
	}
	if len(f.events) != 0 || readDirectTestFile(t, f.config) != before || codexAppDirectConfigured() {
		t.Fatal("migration changed a newly running app without confirmation")
	}
}

func TestCodexAppDirectProcessIdentityRequiresExactProfile(t *testing.T) {
	setTestHome(t, t.TempDir())
	oldExecutable, oldCommand := codexAppProfileExecutable, codexAppProcessCommand
	t.Cleanup(func() {
		codexAppProfileExecutable, codexAppProcessCommand = oldExecutable, oldCommand
	})
	executable := "/Applications/ChatGPT.app/Contents/MacOS/ChatGPT"
	codexAppProfileExecutable = func() (string, error) { return executable, nil }
	userData, _ := codexAppOllamaProfileUserDataDir()
	for _, test := range []struct {
		suffix string
		match  bool
	}{{"", true}, {" --extra-flag", true}, {"-other", false}} {
		codexAppProcessCommand = func(int) (string, error) {
			return executable + " --user-data-dir=" + userData + test.suffix, nil
		}
		if running, matches := codexAppOllamaProfileProcessIdentity(424242); !running || matches != test.match {
			t.Fatalf("profile suffix %q: running = %v, matches = %v", test.suffix, running, matches)
		}
	}
}

func TestCodexAppDirectHealthAndLaunchWithoutGateway(t *testing.T) {
	setTestHome(t, t.TempDir())
	withCodexAppPlatform(t, "darwin")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead || r.URL.Path != "/" {
			t.Errorf("unexpected health request %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	if err := codexAppDirectHealth(); err != nil {
		t.Fatal(err)
	}
	path, _ := codexAppDirectConfigPath()
	userData, _ := codexAppOllamaProfileUserDataDir()
	want := []string{"-n", "--env", "CODEX_HOME=" + filepath.Dir(path), "--env", "CODEX_ELECTRON_USER_DATA_PATH=" + userData, "/Applications/ChatGPT.app", "--args", "--user-data-dir=" + userData}
	if got := codexAppDirectOpenArgs("/Applications/ChatGPT.app", path, userData); !slices.Equal(got, want) {
		t.Fatalf("isolated launch = %q, want %q", got, want)
	}
}

func TestCodexAppDirectCountsOnlyItsOwnRequests(t *testing.T) {
	f := newDirectAppFixture(t)
	app := &CodexDesktop{}
	if err := app.UseOllamaFromDesktop("qwen3.5:latest", testLaunchModels("qwen3.5:latest"), false); err != nil {
		t.Fatal(err)
	}
	path, _ := codexAppDirectConfigPath()
	line := `{"timestamp":"` + time.Now().UTC().Format(time.RFC3339Nano) + `","type":"response_item","payload":{"type":"message","role":"user","internal_chat_message_metadata_passthrough":{"content_item_kinds":["user.text"]}}}` + "\n"
	writeDirectTestFile(t, filepath.Join(filepath.Dir(f.config), "sessions", "regular.jsonl"), line+line)
	writeDirectTestFile(t, filepath.Join(filepath.Dir(path), "sessions", "ollama.jsonl"), line)
	if got := app.OllamaRequestCount(); got != 1 {
		t.Fatalf("request count = %d, want one isolated-profile prompt", got)
	}
}
