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

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/spf13/cobra"
)

func TestLaunchCmdPiPickerPolicy(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}
	for _, tc := range []struct {
		name       string
		saved      []string
		args       []string
		wantPicker int
		wantModel  string
		wantError  string
		wantPiArgs []string
	}{
		{name: "available saved model", saved: []string{"keep-model"}, wantModel: "keep-model"},
		{name: "disabled cloud secondary is retained", saved: []string{"keep-model", "remote:cloud"}, wantModel: "keep-model"},
		{name: "first setup", wantPicker: 1, wantModel: "chosen"},
		{name: "missing saved default", saved: []string{"missing", "keep-model"}, wantPicker: 1, wantModel: "chosen"},
		{name: "explicit configuration", saved: []string{"keep-model"}, args: []string{"--config"}, wantPicker: 1, wantModel: "chosen"},
		{name: "disabled saved cloud default", saved: []string{"remote:cloud", "keep-model"}, wantError: "cloud is disabled"},
		{name: "disabled explicit cloud override", saved: []string{"keep-model"}, args: []string{"--model", "remote:cloud"}, wantError: "cloud is disabled"},
		{name: "configure replaces disabled cloud default", saved: []string{"remote:cloud", "keep-model"}, args: []string{"--config"}, wantPicker: 1, wantModel: "chosen"},
		{name: "forwarded qualified model", saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "forwarded provider and model", saved: []string{"remote:cloud", "keep-model"}, args: []string{"--", "--provider", "fixture-user", "--model", "user-choice"}, wantPiArgs: []string{"--provider", "fixture-user", "--model", "user-choice"}},
		{name: "forwarded model replaces explicit cloud target", saved: []string{"keep-model"}, args: []string{"--model", "remote:cloud", "--", "--model", "fixture-user/user-choice"}, wantPiArgs: []string{"--model", "fixture-user/user-choice"}},
		{name: "model flag inside a prompt is not an override", saved: []string{"keep-model"}, args: []string{"--", "--system-prompt", "--model", "probe"}, wantPiArgs: []string{"--provider", "ollama", "--model", "keep-model", "--system-prompt", "--model", "probe"}},
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
			DefaultMultiSelector = func(_ string, _ []SelectionItem, _ []string) ([]string, error) {
				pickerCalls++
				return []string{"chosen"}, nil
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
					fmt.Fprint(w, `{"models":[{"name":"chosen"},{"name":"keep-model"}]}`)
				case "/api/show":
					var req apiShowRequest
					_ = json.NewDecoder(r.Body).Decode(&req)
					if req.Model == "remote:cloud" {
						http.Error(w, `{"error":"cloud is disabled"}`, http.StatusForbidden)
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
			if tc.wantPicker == 0 {
				saved, err := config.LoadIntegration("pi")
				if err != nil || !slices.Equal(saved.Models, tc.saved) {
					t.Fatalf("direct launch changed saved choices: %v, %v", saved, err)
				}
			}
			data, err := os.ReadFile(argsPath)
			want := []string{"--provider", "ollama", "--model", tc.wantModel}
			if tc.wantPiArgs != nil {
				want = tc.wantPiArgs
			}
			if err != nil || !slices.Equal(strings.Fields(string(data)), want) {
				t.Fatalf("Pi arguments = %q, %v; want %v", data, err, want)
			}
		})
	}
}

func TestSelectReadyModelsForSave_NonPiRetainedSecondaryEscapeStillSkips(t *testing.T) {
	withLauncherHooks(t)
	var prompts []string
	DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
		prompts = append(prompts, prompt)
		return false, ErrCancelled
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model == "missing-secondary" {
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
	client, err := newLauncherClient(defaultLaunchPolicy(true, false))
	if err != nil {
		t.Fatal(err)
	}
	models := []string{"available-primary", "missing-secondary"}
	accepted, skipped, err := client.selectReadyModelsForSave(t.Context(), models, models, "Droid", "droid")
	if err != nil || !slices.Equal(accepted, models[:1]) || len(skipped) != 1 || !slices.Equal(prompts, []string{"Download missing-secondary?"}) {
		t.Fatalf("accepted %v, skipped %v, prompts %v, error %v", accepted, skipped, prompts, err)
	}
}

func TestLaunchIntegration_PiRemovesUnavailableSavedModels(t *testing.T) {
	home := t.TempDir()
	setLaunchTestHome(t, home)
	withLauncherHooks(t)
	withInteractiveSession(t, true)
	previous := []string{"test:latest", "keep-model"}
	if err := config.SaveIntegration("pi", previous); err != nil {
		t.Fatal(err)
	}
	DefaultMultiSelector = func(_ string, items []SelectionItem, preChecked []string) ([]string, error) {
		for _, item := range items {
			if item.Name == "test:latest" {
				t.Fatal("missing saved model unexpectedly visible in the picker")
			}
		}
		// The picker adds a new default while retaining previously configured models.
		return append([]string{"chosen"}, preChecked...), nil
	}
	DefaultConfirmPrompt = func(prompt string, _ ConfirmOptions) (bool, error) {
		if prompt == "Launch Pi now?" {
			return false, nil
		}
		t.Errorf("unexpected prompt for a retained model: %s", prompt)
		return false, nil
	}
	var shown []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/status":
			fmt.Fprint(w, `{"cloud":{"disabled":true}}`)
		case "/api/experimental/model-recommendations":
			fmt.Fprint(w, `{"recommendations":[{"model":"chosen"}]}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"chosen"},{"name":"keep-model"}]}`)
		case "/api/show":
			var req apiShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			shown = append(shown, req.Model)
			if req.Model == "chosen" || req.Model == "keep-model" {
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

	for range 2 {
		if err := LaunchIntegration(t.Context(), IntegrationLaunchRequest{
			Name: "pi", ForceConfigure: true, ConfigureOnly: true,
		}); err != nil {
			t.Fatal(err)
		}
		previous = []string{"chosen", "keep-model"}
		saved, err := config.LoadIntegration("pi")
		if err != nil || !slices.Equal(saved.Models, previous) {
			t.Fatalf("saved models = %v, %v; want %v", saved, err, previous)
		}
		settings, err := fileutil.ReadJSON(filepath.Join(home, ".pi", "agent", "settings.json"))
		if err != nil || settings["defaultModel"] != "chosen" {
			t.Fatalf("Pi settings = %v, %v", settings, err)
		}
	}
	missingChecks := 0
	for _, name := range shown {
		if name == "test:latest" {
			missingChecks++
		}
	}
	if missingChecks != 1 {
		t.Fatalf("missing model checked %d times across both launches; want once", missingChecks)
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
	DefaultMultiSelector = func(_ string, _ []SelectionItem, _ []string) ([]string, error) {
		return []string{"missing-new", "previous"}, nil
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
