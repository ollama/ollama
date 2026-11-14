package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/spf13/cobra"
)

func TestLaunchCmdPiPickerPolicy(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}
	type testCase struct {
		name          string
		saved         []string
		args          []string
		wantPicker    int
		wantModel     string
		wantError     string
		wantPiArgs    []string
		wantCurrent   string
		headless      bool
		passthrough   bool
		corruptConfig bool
		provider      string
		cloudDisabled bool
		scoped        bool
	}
	cases := []testCase{
		{name: "available saved model remains highlighted", saved: []string{"keep-model"}, wantPicker: 1, wantCurrent: "keep-model", wantModel: "chosen"},
		{name: "legacy selection migrates to one starting model", saved: []string{"keep-model", "remote:cloud"}, wantPicker: 1, wantCurrent: "keep-model", wantModel: "chosen"},
		{name: "headless saved model skips picker", saved: []string{"keep-model"}, headless: true, wantModel: "keep-model"},
		{name: "first setup", wantPicker: 1, wantModel: "chosen"},
		{name: "missing saved default", saved: []string{"missing", "keep-model"}, wantPicker: 1, wantModel: "chosen"},
		{name: "explicit configuration", saved: []string{"keep-model"}, args: []string{"--config"}, wantPicker: 1, wantModel: "chosen"},
		{name: "picker replaces disabled saved cloud default", saved: []string{"remote:cloud", "keep-model"}, wantPicker: 1, wantModel: "chosen"},
		{name: "disabled explicit cloud override", saved: []string{"keep-model"}, args: []string{"--model", "remote:cloud"}, wantError: "cloud is disabled"},
		{name: "configure replaces disabled cloud default", saved: []string{"remote:cloud", "keep-model"}, args: []string{"--config"}, wantPicker: 1, wantModel: "chosen"},
		{name: "configure replaces disabled explicit cloud target", saved: []string{"keep-model"}, args: []string{"--config", "--model", "remote:cloud"}, wantPicker: 1, wantModel: "chosen"},
		{name: "saved embedding model requires a chat model", saved: []string{"embeddings", "keep-model"}, wantPicker: 1, wantModel: "chosen"},
		{name: "short model flag", saved: []string{"keep-model"}, args: []string{"-m", "chosen"}, wantModel: "chosen", wantPiArgs: []string{"--provider", "ollama", "--model", "chosen", "--models", "ollama/*"}},
		{name: "model flag after Pi separator is prompt text", saved: []string{"keep-model"}, args: []string{"--model", "chosen", "--", "--", "--model", "example"}, wantModel: "chosen", wantPiArgs: []string{"--provider", "ollama", "--model", "chosen", "--models", "ollama/*", "--", "--model", "example"}},
		{name: "provider flag after Pi separator is prompt text", saved: []string{"keep-model"}, args: []string{"--model", "chosen", "--", "--", "--provider", "example"}, wantModel: "chosen", wantPiArgs: []string{"--provider", "ollama", "--model", "chosen", "--models", "ollama/*", "--", "--provider", "example"}},
		{name: "explicit embedding model is rejected", saved: []string{"keep-model"}, args: []string{"--model", "embeddings"}, wantError: "does not support chat"},
		{name: "legacy saved embedding model requires a chat model", saved: []string{"legacy-embedding"}, wantPicker: 1, wantModel: "chosen"},
		{name: "picker replaces disabled saved cloud alias", saved: []string{"remote-alias", "keep-model"}, wantPicker: 1, wantModel: "chosen"},
		{name: "disabled explicit cloud alias", saved: []string{"keep-model"}, args: []string{"--model", "remote-alias"}, wantError: "cloud is disabled"},
		{name: "cloud alias identified by show", saved: []string{"keep-model"}, args: []string{"--model", "hidden-alias"}, wantError: "cloud is disabled"},
		{name: "forwarded qualified model", passthrough: true, saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "native override before Pi separator wins", passthrough: true, saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--model", "fixture-user/user-choice", "--", "--model", "example"}, wantPiArgs: []string{"--model", "fixture-user/user-choice", "--", "--model", "example"}},
		{name: "separator consumed by a Pi option is a value", passthrough: true, saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--name", "--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--name", "--", "--model", "fixture-user/user-choice"}},
		{name: "native override ignores unused malformed launch config", passthrough: true, corruptConfig: true, saved: []string{"keep-model"}, args: []string{"--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "forwarded provider and model", passthrough: true, saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--provider", "fixture-user", "--model", "user-choice"}, wantPiArgs: []string{"--provider", "fixture-user", "--model", "user-choice"}},
		{name: "forwarded model replaces explicit cloud target", passthrough: true, saved: []string{"keep-model"}, args: []string{"--model", "remote:cloud", "--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "prompt after Pi separator keeps startup picker", saved: []string{"keep-model"}, args: []string{"--", "--", "--model", "example"}, wantPicker: 1, wantModel: "chosen", wantPiArgs: []string{"--provider", "ollama", "--model", "chosen", "--models", "ollama/*", "--", "--model", "example"}},
		{name: "headless native help with yes", passthrough: true, headless: true, args: []string{"--yes", "--", "--help"}, wantPiArgs: []string{"--help"}},
		{name: "headless provider ignores missing launcher model", passthrough: true, headless: true, provider: "fixture-provider", saved: []string{"missing"}, wantPiArgs: []string{}},
		{name: "headless provider ignores disabled cloud launcher model", passthrough: true, headless: true, provider: "fixture-provider", saved: []string{"remote:cloud"}, cloudDisabled: true, wantPiArgs: []string{}},
		{name: "headless provider ignores transient launcher failure", passthrough: true, headless: true, provider: "fixture-provider", saved: []string{"transient"}, wantPiArgs: []string{}},
		{name: "headless saved scope ignores unrelated launcher model", passthrough: true, headless: true, scoped: true, saved: []string{"missing"}, wantPiArgs: []string{}},
		{name: "text mode remains interactive", saved: []string{"keep-model"}, wantPicker: 1, wantModel: "chosen", args: []string{"--", "--mode", "text"}, wantPiArgs: []string{"--provider", "ollama", "--model", "chosen", "--models", "ollama/*", "--mode", "text"}},
		{name: "model flag inside a prompt is not an override", wantPicker: 1, wantModel: "chosen", saved: []string{"keep-model"}, args: []string{"--", "--system-prompt", "--model", "probe"}, wantPiArgs: []string{"--provider", "ollama", "--model", "chosen", "--models", "ollama/*", "--system-prompt", "--model", "probe"}},
	}
	for _, args := range [][]string{
		{"--continue"},
		{"--resume"},
		{"--session", "session.jsonl"},
		{"--fork", "session.jsonl"},
		{"--print", "probe"},
		{"--mode", "json"},
		{"--models", "fixture-user/*"},
		{"--help"},
		{"--version"},
		{"--list-models"},
		{"--export", "session.jsonl"},
		{"list"},
		{"install", "fixture-package"},
		{"remove", "fixture-package"},
		{"uninstall", "fixture-package"},
		{"update"},
		{"config"},
	} {
		for _, saved := range [][]string{nil, {"missing"}} {
			cases = append(cases, testCase{name: fmt.Sprintf("native %v saved=%v", args, saved), passthrough: true, saved: saved, args: append([]string{"--"}, args...), wantPiArgs: args})
		}
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			t.Chdir(t.TempDir())
			t.Setenv("PI_CODING_AGENT_DIR", "")
			withLauncherHooks(t)
			withInteractiveSession(t, !tc.headless)
			if tc.saved != nil {
				if err := config.SaveIntegration("pi", tc.saved); err != nil {
					t.Fatal(err)
				}
			}
			if tc.corruptConfig {
				if err := os.WriteFile(filepath.Join(home, ".ollama", "config.json"), []byte("{invalid"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			preserved := map[string]string{}
			if tc.passthrough && tc.saved != nil {
				provider := tc.provider
				if provider == "" {
					provider = "ollama"
				}
				scope := ""
				if tc.scoped {
					scope = `,"enabledModels":["fixture-user/*"]`
				}
				preserved = map[string]string{
					filepath.Join(home, ".pi", "agent", "settings.json"):             fmt.Sprintf(`{"defaultProvider":%q,"defaultModel":"missing","editorPaddingX":3%s}`, provider, scope),
					filepath.Join(home, ".pi", "agent", "models.json"):               `{"providers":{"ollama":{"models":[{"id":"missing","_launch":true}]},"fixture-user":{"apiKey":"synthetic"}}}`,
					filepath.Join(home, ".pi", "agent", "sessions", "history.jsonl"): "existing history\n",
				}
				for path, data := range preserved {
					if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
						t.Fatal(err)
					}
				}
			}
			binDir := t.TempDir()
			t.Setenv("PATH", binDir)
			argsPath := filepath.Join(home, "pi-args")
			for name, data := range map[string]string{
				"pi":           fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' \"$@\" >> %q\n", argsPath),
				"npm":          "#!/bin/sh\nexit 1\n",
				"package.json": `{"name":"@earendil-works/pi-coding-agent"}`,
			} {
				if err := os.WriteFile(filepath.Join(binDir, name), []byte(data), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			pickerCalls := 0
			DefaultSingleSelector = func(_ string, items []SelectionItem, current string) (string, error) {
				pickerCalls++
				if tc.wantCurrent != "" && current != tc.wantCurrent {
					t.Errorf("highlighted model = %q, want %q", current, tc.wantCurrent)
				}
				for _, item := range items {
					if item.Name == "embeddings" || item.Name == "remote-alias" {
						t.Errorf("unusable model appears in Pi picker: %s", item.Name)
					}
				}
				return "chosen", nil
			}
			DefaultMultiSelector = func(_ string, _ []SelectionItem, _ []string) ([]string, error) {
				t.Fatal("Pi must select only its starting model")
				return nil, ErrCancelled
			}
			DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
				if prompt != "Launch Pi now?" || len(tc.args) == 0 {
					t.Errorf("unexpected confirmation: %s", prompt)
				}
				return true, nil
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if tc.passthrough && r.URL.Path != "/api/status" {
					t.Errorf("native invocation checked model readiness: %s", r.URL.Path)
				}
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprintf(w, `{"cloud":{"disabled":%t}}`, !tc.passthrough || tc.cloudDisabled)
				case "/api/experimental/model-recommendations":
					fmt.Fprint(w, `{"recommendations":[{"model":"chosen"}]}`)
				case "/api/tags":
					fmt.Fprint(w, `{"models":[{"name":"chosen","details":{"context_length":65536},"capabilities":["completion","tools"]},{"name":"keep-model","details":{"context_length":32768},"capabilities":["completion","tools","vision"]},{"name":"embeddings","details":{"context_length":8192},"capabilities":["embedding"]},{"name":"remote-alias","remote_model":"remote","details":{"context_length":32768},"capabilities":["completion","tools"]}]}`)
				case "/api/show":
					var req apiShowRequest
					_ = json.NewDecoder(r.Body).Decode(&req)
					if req.Model == "remote:cloud" {
						http.Error(w, `{"error":"cloud is disabled"}`, http.StatusForbidden)
					} else if req.Model == "remote-alias" || req.Model == "hidden-alias" {
						fmt.Fprint(w, `{"remote_model":"remote","capabilities":["completion","tools"]}`)
					} else if req.Model == "embeddings" || req.Model == "legacy-embedding" {
						fmt.Fprint(w, `{"capabilities":["embedding"]}`)
					} else if req.Model == "missing" {
						http.Error(w, `{"error":"missing"}`, http.StatusNotFound)
					} else if req.Model == "transient" {
						http.Error(w, `{"error":"temporary failure"}`, http.StatusServiceUnavailable)
					} else {
						fmt.Fprint(w, `{}`)
					}
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			t.Setenv("OLLAMA_HOST", srv.URL)
			cmd := LaunchCmd(func(*cobra.Command, []string) error { return nil }, nil)
			cmd.SetArgs(append([]string{"pi"}, tc.args...))
			err := cmd.Execute()
			if tc.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("launch error = %v, want %q", err, tc.wantError)
				}
				if pickerCalls != 0 {
					t.Fatal("disabled cloud target unexpectedly opened the picker")
				}
				if _, err := os.Stat(argsPath); !os.IsNotExist(err) {
					t.Fatalf("Pi started despite unavailable cloud target: %v", err)
				}
				saved, err := config.LoadIntegration("pi")
				if err != nil || !slices.Equal(saved.Models, tc.saved) {
					t.Fatalf("saved choices changed on disabled cloud failure: %v, %v", saved, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if pickerCalls != tc.wantPicker {
				t.Fatalf("picker called %d times, want %d", pickerCalls, tc.wantPicker)
			}
			wantSaved := []string{tc.wantModel}
			if tc.passthrough {
				wantSaved = tc.saved
			}
			saved, err := config.LoadIntegration("pi")
			if tc.corruptConfig {
				data, readErr := os.ReadFile(filepath.Join(home, ".ollama", "config.json"))
				if readErr != nil || string(data) != "{invalid" {
					t.Fatalf("unused malformed config changed: %s, %v", data, readErr)
				}
			} else if tc.passthrough && tc.saved == nil {
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("native invocation created launcher config: %v, %v", saved, err)
				}
			} else if err != nil || !slices.Equal(saved.Models, wantSaved) {
				t.Fatalf("saved starting model = %v, %v; want %v", saved, err, wantSaved)
			}
			if !tc.passthrough {
				models := (&Pi{}).Models()
				if !slices.Equal(models, []string{"chosen", "keep-model"}) {
					t.Fatalf("Pi catalog = %v, want all available models", models)
				}
			} else if tc.saved == nil {
				if _, err := os.Stat(filepath.Join(home, ".pi")); !os.IsNotExist(err) {
					t.Fatalf("native invocation created a Pi profile: %v", err)
				}
			}
			for path, want := range preserved {
				data, err := os.ReadFile(path)
				if err != nil || string(data) != want {
					t.Fatalf("native invocation changed %s: %q, %v", filepath.Base(path), data, err)
				}
			}

			data, err := os.ReadFile(argsPath)
			want := []string{"--models", "ollama/*"}
			if tc.wantPicker > 0 {
				want = append([]string{"--provider", "ollama", "--model", tc.wantModel}, want...)
			}
			if tc.wantPiArgs != nil {
				want = tc.wantPiArgs
			}
			if err != nil || !slices.Equal(strings.Fields(string(data)), want) {
				t.Fatalf("Pi arguments = %q, %v; want %v", data, err, want)
			}
		})
	}
}

func TestLaunchIntegration_PiDownloadPromptCancellationPreservesConfig(t *testing.T) {
	home := t.TempDir()
	setLaunchTestHome(t, home)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	if err := config.SaveIntegration("pi", []string{"previous"}); err != nil {
		t.Fatal(err)
	}
	DefaultSingleSelector = func(_ string, _ []SelectionItem, _ string) (string, error) {
		return "missing-new", nil
	}
	DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
		if prompt != "Download missing-new?" {
			t.Errorf("unexpected prompt: %s", prompt)
		}
		return false, ErrCancelled
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"missing-new"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"previous"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "previous" {
				fmt.Fprint(w, `{}`)
			} else {
				http.Error(w, `{"error":"missing"}`, http.StatusNotFound)
			}
		default:
			t.Errorf("unexpected request: %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name: "pi", ForceConfigure: true, ConfigureOnly: true,
	})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("launch error = %v, want cancellation", err)
	}
	saved, err := config.LoadIntegration("pi")
	if err != nil || !slices.Equal(saved.Models, []string{"previous"}) {
		t.Fatalf("saved config changed on cancellation: %v, %v", saved, err)
	}
	if _, err := os.Stat(filepath.Join(home, ".pi")); !os.IsNotExist(err) {
		t.Fatalf("Pi config was written on cancellation: %v", err)
	}
}

func TestPiRun_PassesSelectedModelToClient(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}
	dir := t.TempDir()
	setTestHome(t, dir)
	t.Chdir(t.TempDir())
	t.Setenv("PI_CODING_AGENT_DIR", "")
	t.Setenv("PATH", dir)
	argsPath := filepath.Join(dir, "args")
	for name, content := range map[string]string{
		"pi":           fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' \"$@\" > %q\n", argsPath),
		"npm":          "#!/bin/sh\nexit 1\n",
		"package.json": `{"name":"@earendil-works/pi-coding-agent"}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
	}))
	defer srv.Close()
	t.Setenv("OLLAMA_HOST", srv.URL)
	if err := (&Pi{}).Run("chosen:latest", nil, []string{"--offline", "--print", "probe"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--provider", "ollama", "--model", "chosen:latest", "--offline", "--print", "probe"}
	if got := strings.Split(strings.TrimSpace(string(data)), "\n"); !slices.Equal(got, want) {
		t.Fatalf("Pi process arguments = %v, want %v", got, want)
	}
}

func TestLaunchIntegrationPiPreservesModelChoice(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}
	for _, tc := range []struct {
		name         string
		current      string
		override     string
		configure    bool
		stallCatalog bool
		extra        []string
		want         string
		wantArgs     []string
		wantPicker   bool
		customAgent  string
		provider     string
		project      string
	}{
		{name: "saved Pi default", current: "model-b", want: "model-b", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-b", "--models", "ollama/*"}},
		{name: "project Ollama model is highlighted", current: "model-b", project: `{"defaultProvider":"ollama","defaultModel":"model-a"}`, want: "model-a", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-a", "--models", "ollama/*"}},
		{name: "project provider wins over global preference", current: "model-b", project: `{"defaultProvider":"fixture-provider","defaultModel":"external-model"}`, want: "model-a", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-a"}},
		{name: "malformed project settings are left to Pi", current: "model-b", project: `{invalid`, want: "model-b", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-b"}},
		{name: "custom Pi profile", customAgent: "absolute", current: "model-b", want: "model-b", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-b", "--models", "ollama/*"}},
		{name: "tilde Pi profile", customAgent: "tilde", current: "model-b", want: "model-b", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-b", "--models", "ollama/*"}},
		{name: "saved provider preference", provider: "fixture-provider", current: "external-model", want: "external-model", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-a"}},
		{name: "custom profile provider preference", customAgent: "absolute", provider: "fixture-provider", current: "external-model", want: "external-model", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-a"}},
		{name: "explicit configuration changes provider preference", configure: true, provider: "fixture-provider", current: "external-model", want: "model-a", wantPicker: true, wantArgs: []string{"--provider", "ollama", "--model", "model-a", "--models", "ollama/*"}},
		{name: "resume restores its own model", current: "model-a", extra: []string{"--continue"}, want: "model-a", wantArgs: []string{"--continue"}},
		{name: "explicit override wins", current: "model-b", override: "model-a", extra: []string{"--continue"}, want: "model-a", wantArgs: []string{"--provider", "ollama", "--model", "model-a", "--continue"}},
		{name: "explicit lookup precedes timed out catalog", current: "model-b", override: "model-a", stallCatalog: true, want: "model-a", wantArgs: []string{"--provider", "ollama", "--model", "model-a", "--models", "ollama/*"}},
		{name: "explicit configuration wins", current: "model-b", configure: true, wantPicker: true, want: "model-a", wantArgs: []string{"--provider", "ollama", "--model", "model-a", "--models", "ollama/*"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			t.Chdir(t.TempDir())
			t.Setenv("PI_CODING_AGENT_DIR", "")
			projectPath := filepath.Join(".pi", "settings.json")
			if tc.project != "" {
				if err := os.MkdirAll(filepath.Dir(projectPath), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(projectPath, []byte(tc.project), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			withLauncherHooks(t)
			withInteractiveSession(t, true)
			if err := config.SaveIntegration("pi", []string{"model-a", "stale"}); err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(home, ".pi", "agent")
			if tc.customAgent != "" {
				dir = filepath.Join(home, "custom-agent")
				t.Setenv("PI_CODING_AGENT_DIR", dir)
				if tc.customAgent == "tilde" {
					t.Setenv("PI_CODING_AGENT_DIR", "~/custom-agent")
				}
			}
			if err := os.MkdirAll(dir, 0o755); err != nil {
				t.Fatal(err)
			}
			if tc.stallCatalog {
				oldTimeout := piModelDiscoveryTimeout
				piModelDiscoveryTimeout = 50 * time.Millisecond
				t.Cleanup(func() { piModelDiscoveryTimeout = oldTimeout })
				if err := os.WriteFile(filepath.Join(dir, "models.json"), []byte(`{"providers":{"ollama":{"models":[{"id":"model-b","_launch":true,"contextWindow":8192}]}}}`), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			firstLookup := make(chan string, 1)
			settingsPath := filepath.Join(dir, "settings.json")
			provider := tc.provider
			if provider == "" {
				provider = "ollama"
			}
			settings := fmt.Sprintf(`{"defaultProvider":%q,"defaultModel":%q,"editorPaddingX":7}`, provider, tc.current)
			if err := os.WriteFile(settingsPath, []byte(settings), 0o644); err != nil {
				t.Fatal(err)
			}
			bin := t.TempDir()
			argsPath := filepath.Join(home, "pi-args")
			for name, content := range map[string]string{
				"pi":           fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' \"$@\" > %q\n", argsPath),
				"npm":          "#!/bin/sh\nexit 1\n",
				"package.json": `{"name":"@earendil-works/pi-coding-agent"}`,
			} {
				if err := os.WriteFile(filepath.Join(bin, name), []byte(content), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			t.Setenv("PATH", bin)
			DefaultSingleSelector = func(_ string, _ []SelectionItem, current string) (string, error) {
				if !tc.wantPicker {
					t.Fatal("explicit model or resume unexpectedly opened the picker")
				}
				if !tc.configure {
					return current, nil
				}
				return "model-a", nil
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
				case "/api/tags":
					if tc.stallCatalog {
						select {
						case firstLookup <- "tags":
						default:
						}
						<-r.Context().Done()
						return
					}
					fmt.Fprint(w, `{"models":[{"name":"model-a","capabilities":["completion"],"details":{"context_length":32768}},{"name":"model-b","capabilities":["completion"],"details":{"context_length":32768}}]}`)
				case "/api/show":
					select {
					case firstLookup <- "show":
					default:
					}
					fmt.Fprint(w, `{"capabilities":["completion"],"model_info":{"test.context_length":32768}}`)
				case "/api/experimental/model-recommendations":
					fmt.Fprint(w, `{"recommendations":[{"model":"model-a"}]}`)
				default:
					t.Errorf("unexpected request: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			for range 2 {
				select {
				case <-firstLookup:
				default:
				}
				err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{Name: "pi", ModelOverride: tc.override, ForceConfigure: tc.configure, ExtraArgs: tc.extra})
				if err != nil {
					t.Fatal(err)
				}
				data, err := os.ReadFile(settingsPath)
				if err != nil {
					t.Fatal(err)
				}
				var settings map[string]any
				if err := json.Unmarshal(data, &settings); err != nil {
					t.Fatal(err)
				}
				if settings["defaultModel"] != tc.want || settings["editorPaddingX"] != float64(7) {
					t.Errorf("Pi settings = %s, want default %s and preserved padding", data, tc.want)
				}
				wantProvider := provider
				if tc.configure || tc.override != "" {
					wantProvider = "ollama"
				}
				if settings["defaultProvider"] != wantProvider {
					t.Errorf("Pi provider = %v, want %s", settings["defaultProvider"], wantProvider)
				}
				if tc.project != "" {
					data, err := os.ReadFile(projectPath)
					if err != nil || string(data) != tc.project {
						t.Fatalf("project preference changed: %s, %v", data, err)
					}
				}
				if tc.customAgent != "" {
					pi := &Pi{}
					if !slices.Equal(pi.Paths(), []string{filepath.Join(dir, "models.json"), settingsPath}) || !slices.Equal(pi.Models(), []string{"model-a", "model-b"}) {
						t.Fatalf("custom profile paths/models = %v/%v", pi.Paths(), pi.Models())
					}
					if _, err := os.Stat(filepath.Join(home, ".pi")); !os.IsNotExist(err) {
						t.Fatalf("custom launch changed default Pi profile: %v", err)
					}
				}
				if tc.stallCatalog {
					if lookup := <-firstLookup; lookup != "show" {
						t.Errorf("first model lookup = %s, want show", lookup)
					}
					data, err := os.ReadFile(filepath.Join(dir, "models.json"))
					if err != nil || !strings.Contains(string(data), `"model-b"`) {
						t.Errorf("partial discovery discarded known catalog: %s, %v", data, err)
					}
				}
				args, err := os.ReadFile(argsPath)
				if err != nil || !slices.Equal(strings.Fields(string(args)), tc.wantArgs) {
					t.Errorf("Pi args = %q, %v; want %v", args, err, tc.wantArgs)
				}
			}
		})
	}
}
