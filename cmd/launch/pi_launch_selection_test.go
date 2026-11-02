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
	for _, tc := range []struct {
		name            string
		saved           []string
		args            []string
		wantPicker      int
		wantModel       string
		wantError       string
		wantPiArgs      []string
		runtimeOverride bool
		corruptConfig   bool
	}{
		{name: "available saved model", saved: []string{"keep-model"}, wantModel: "keep-model"},
		{name: "legacy selection migrates to one starting model", saved: []string{"keep-model", "remote:cloud"}, wantModel: "keep-model"},
		{name: "first setup", wantPicker: 1, wantModel: "chosen"},
		{name: "missing saved default", saved: []string{"missing", "keep-model"}, wantPicker: 1, wantModel: "chosen"},
		{name: "explicit configuration", saved: []string{"keep-model"}, args: []string{"--config"}, wantPicker: 1, wantModel: "chosen"},
		{name: "disabled saved cloud default", saved: []string{"remote:cloud", "keep-model"}, wantError: "cloud is disabled"},
		{name: "disabled explicit cloud override", saved: []string{"keep-model"}, args: []string{"--model", "remote:cloud"}, wantError: "cloud is disabled"},
		{name: "configure replaces disabled cloud default", saved: []string{"remote:cloud", "keep-model"}, args: []string{"--config"}, wantPicker: 1, wantModel: "chosen"},
		{name: "configure replaces disabled explicit cloud target", saved: []string{"keep-model"}, args: []string{"--config", "--model", "remote:cloud"}, wantPicker: 1, wantModel: "chosen"},
		{name: "saved embedding model requires a chat model", saved: []string{"embeddings", "keep-model"}, wantPicker: 1, wantModel: "chosen"},
		{name: "short model flag", saved: []string{"keep-model"}, args: []string{"-m", "chosen"}, wantModel: "chosen", wantPiArgs: []string{"--provider", "ollama", "--model", "chosen"}},
		{name: "explicit embedding model is rejected", saved: []string{"keep-model"}, args: []string{"--model", "embeddings"}, wantError: "does not support chat"},
		{name: "legacy saved embedding model requires a chat model", saved: []string{"legacy-embedding"}, wantPicker: 1, wantModel: "chosen"},
		{name: "disabled saved cloud alias", saved: []string{"remote-alias", "keep-model"}, wantError: "cloud is disabled"},
		{name: "disabled explicit cloud alias", saved: []string{"keep-model"}, args: []string{"--model", "remote-alias"}, wantError: "cloud is disabled"},
		{name: "cloud alias identified by show", saved: []string{"keep-model"}, args: []string{"--model", "hidden-alias"}, wantError: "cloud is disabled"},
		{name: "forwarded qualified model", runtimeOverride: true, saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "native override ignores unused malformed launch config", runtimeOverride: true, corruptConfig: true, saved: []string{"keep-model"}, args: []string{"--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "forwarded provider and model", runtimeOverride: true, saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--provider", "fixture-user", "--model", "user-choice"}, wantPiArgs: []string{"--provider", "fixture-user", "--model", "user-choice"}},
		{name: "forwarded model replaces explicit cloud target", runtimeOverride: true, saved: []string{"keep-model"}, args: []string{"--model", "remote:cloud", "--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "model flag inside a prompt is not an override", wantModel: "keep-model", saved: []string{"keep-model"}, args: []string{"--", "--system-prompt", "--model", "probe"}, wantPiArgs: []string{"--system-prompt", "--model", "probe"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			withLauncherHooks(t)
			withInteractiveSession(t, true)
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
			binDir := t.TempDir()
			t.Setenv("PATH", binDir)
			argsPath := filepath.Join(home, "pi-args")
			for name, data := range map[string]string{
				"pi":           fmt.Sprintf("#!/bin/sh\nprintf '%%s\\n' \"$@\" > %q\n", argsPath),
				"npm":          "#!/bin/sh\nexit 1\n",
				"package.json": `{"name":"@earendil-works/pi-coding-agent"}`,
			} {
				if err := os.WriteFile(filepath.Join(binDir, name), []byte(data), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			pickerCalls := 0
			DefaultSingleSelector = func(_ string, items []SelectionItem, _ string) (string, error) {
				pickerCalls++
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
				switch r.URL.Path {
				case "/api/status":
					fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
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
			if tc.runtimeOverride {
				wantSaved = tc.saved
			}
			saved, err := config.LoadIntegration("pi")
			if tc.corruptConfig {
				data, readErr := os.ReadFile(filepath.Join(home, ".ollama", "config.json"))
				if readErr != nil || string(data) != "{invalid" {
					t.Fatalf("unused malformed config changed: %s, %v", data, readErr)
				}
			} else if err != nil || !slices.Equal(saved.Models, wantSaved) {
				t.Fatalf("saved starting model = %v, %v; want %v", saved, err, wantSaved)
			}
			if !tc.runtimeOverride {
				models := (&Pi{}).Models()
				if !slices.Equal(models, []string{"chosen", "keep-model"}) {
					t.Fatalf("Pi catalog = %v, want all available models", models)
				}
			}

			data, err := os.ReadFile(argsPath)
			var want []string
			if tc.wantPicker > 0 {
				want = []string{"--provider", "ollama", "--model", tc.wantModel}
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
	}{
		{name: "saved Pi default", current: "model-b", want: "model-b"},
		{name: "resume restores its own model", current: "model-a", extra: []string{"--continue"}, want: "model-a", wantArgs: []string{"--continue"}},
		{name: "explicit override wins", current: "model-b", override: "model-a", extra: []string{"--continue"}, want: "model-a", wantArgs: []string{"--provider", "ollama", "--model", "model-a", "--continue"}},
		{name: "explicit lookup precedes timed out catalog", current: "model-b", override: "model-a", stallCatalog: true, want: "model-a", wantArgs: []string{"--provider", "ollama", "--model", "model-a"}},
		{name: "explicit configuration wins", current: "model-b", configure: true, want: "model-a", wantArgs: []string{"--provider", "ollama", "--model", "model-a"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			setLaunchTestHome(t, home)
			withLauncherHooks(t)
			withInteractiveSession(t, true)
			if err := config.SaveIntegration("pi", []string{"model-a", "stale"}); err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join(home, ".pi", "agent")
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
			settings := fmt.Sprintf(`{"defaultProvider":"ollama","defaultModel":%q,"editorPaddingX":7}`, tc.current)
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
			DefaultSingleSelector = func(string, []SelectionItem, string) (string, error) {
				if !tc.configure {
					t.Fatal("ordinary launch unexpectedly opened the picker")
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
