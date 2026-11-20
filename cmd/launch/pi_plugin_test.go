package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/cmd/config"
	"github.com/spf13/cobra"
)

func TestPiLaunchArgsPreserveExplicitAndResumeArguments(t *testing.T) {
	for _, tc := range []struct {
		model string
		args  []string
		want  []string
	}{
		{"", nil, nil},
		{"qwen3.6:latest", []string{"-p", "Explain this project"}, []string{"--model", "ollama/qwen3.6:latest", "-p", "Explain this project"}},
		{"cloud/glm-5.3-flash", nil, []string{"--model", "ollama/cloud/glm-5.3-flash"}},
		{"ollama/local:latest", []string{"--continue"}, []string{"--model", "ollama/local:latest", "--continue"}},
		{"", []string{"--continue", "--session", "a path/session.jsonl"}, []string{"--continue", "--session", "a path/session.jsonl"}},
		{"", []string{"--model", "other/model"}, []string{"--model", "other/model"}},
		{"one", []string{"--model=two"}, []string{"--model", "ollama/one", "--model=two"}},
		{"one", []string{"--model", "two"}, []string{"--model", "ollama/one", "--model", "two"}},
		{"one", []string{"-m", "two"}, []string{"--model", "ollama/one", "-m", "two"}},
	} {
		if got := piLaunchArgs(tc.model, tc.args); !slices.Equal(got, tc.want) {
			t.Fatalf("piLaunchArgs(%q, %q) = %q; want %q", tc.model, tc.args, got, tc.want)
		}
	}
}

func TestPiLaunchPreservesDefaultsAndSessionOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX fake Pi executable")
	}
	root := t.TempDir()
	setTestHome(t, root)
	logPath := filepath.Join(root, "pi.log")
	script := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
if [ "$1" = "--version" ]; then echo 1.5.2; exit 0; fi
if [ "$1" = "--no-extensions" ]; then echo "unexpected setup" >&2; exit 1; fi
echo "color=$FORCE_COLOR" >> %q
if [ "$#" = 0 ]; then echo launched >> %q; echo "launch=$OLLAMA_LAUNCH_PI" >> %q; echo session-output; exit 0; fi
if [ "$1" = "--model" ]; then echo "launch=$OLLAMA_LAUNCH_PI" >> %q; exit 0; fi
exit 1
`, logPath, logPath, logPath, logPath, logPath)
	if err := os.WriteFile(filepath.Join(root, "pi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "npm"), []byte("#!/bin/sh\necho '{}'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root)
	t.Setenv("OLLAMA_LAUNCH_PI", "0")
	t.Setenv("FORCE_COLOR", "1")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("Pi launch must not contact the Ollama daemon: %s", r.URL.Path)
		http.NotFound(w, r)
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	agent := filepath.Join(root, ".pi", "agent")
	if err := os.MkdirAll(filepath.Join(agent, "sessions"), 0o755); err != nil {
		t.Fatal(err)
	}
	settings := `{"defaultProvider":"other","defaultModel":"keep","packages":["git:github.com/ollama/pi-ollama@pinned"]}`
	models := `{"providers":{"ollama":{"baseUrl":"http://old-server/v1","apiKey":"ollama","models":[{"id":"generated","_launch":true},{"id":"manual","contextWindow":8192}]},"other":{"baseUrl":"http://custom","models":[{"id":"keep"}]}}}`
	files := map[string]string{
		"settings.json":          settings,
		"models.json":            models,
		"auth.json":              `{"other":{"type":"api_key","key":"fixture"}}`,
		"sessions/fixture.jsonl": "existing conversation\n",
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(agent, name), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	stdout, err := os.CreateTemp(root, "stdout")
	if err != nil {
		t.Fatal(err)
	}
	previousStdout := os.Stdout
	os.Stdout = stdout
	t.Cleanup(func() {
		os.Stdout = previousStdout
		stdout.Close()
	})
	for range 2 {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "pi"}); err != nil {
			t.Fatal(err)
		}
	}
	if output, err := os.ReadFile(stdout.Name()); err != nil || string(output) != "session-output\nsession-output\n" {
		t.Fatalf("unexpected session stdout: %q (%v)", output, err)
	}
	log, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(log), "launched\n") != 2 {
		t.Fatalf("ordinary Pi was not launched twice:\n%s", log)
	}
	if strings.Count(string(log), "launch=1\n") != 2 {
		t.Fatalf("Pi must receive the Launch handoff:\n%s", log)
	}
	if strings.Count(string(log), "color=1\n") != 2 {
		t.Fatalf("interactive Pi must keep the user's color setting:\n%s", log)
	}
	for name, want := range files {
		got, err := os.ReadFile(filepath.Join(agent, name))
		if err != nil || string(got) != want {
			t.Fatalf("%s changed: %s (%v)", name, got, err)
		}
	}
	// The CLI's Pi path must not start or require the Ollama daemon.
	cmd := LaunchCmd(func(_ *cobra.Command, _ []string) error {
		t.Fatal("Pi must not require the daemon preflight")
		return nil
	}, func(_ *cobra.Command) {})
	cmd.SetArgs([]string{"pi", "--yes", "--model", "cloud/fixture", "--", "-p", "Reply READY"})
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	log, err = os.ReadFile(logPath)
	if err != nil || strings.Count(string(log), "launch=1\n") != 3 {
		t.Fatalf("explicit models must use the same Launch handoff: %s (%v)", log, err)
	}
}

func TestPiLauncherStateLeavesModelsToPi(t *testing.T) {
	root := t.TempDir()
	setTestHome(t, root)
	writeFakeBinary(t, root, "droid")
	t.Setenv("PATH", root)
	if err := config.SaveIntegration("pi", []string{"stale-model"}); err != nil {
		t.Fatal(err)
	}
	if err := config.SaveIntegration("droid", []string{"existing-model"}); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/status" {
			t.Errorf("unexpected request: %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"cloud":{"disabled":false}}`)
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	client, err := newLauncherClient(defaultLaunchPolicy(false, false))
	if err != nil {
		t.Fatal(err)
	}
	state, err := client.buildLauncherIntegrationState(t.Context(), IntegrationInfo{Name: "pi"})
	if err != nil {
		t.Fatal(err)
	}
	if !state.Selectable || state.Changeable || state.CurrentModel != "" || state.ModelUsable || state.Editor {
		t.Fatalf("Pi must launch without advertising launcher-owned model configuration: %+v", state)
	}
	if models := config.IntegrationModels("pi"); !slices.Equal(models, []string{"stale-model"}) {
		t.Fatalf("legacy settings must be preserved: %v", models)
	}
	state, err = client.buildLauncherIntegrationState(t.Context(), IntegrationInfo{Name: "droid"})
	if err != nil {
		t.Fatal(err)
	}
	if !state.Changeable || state.CurrentModel != "existing-model" || !state.ModelUsable || !state.Editor {
		t.Fatalf("other editor integrations must keep their configuration state: %+v", state)
	}
}

func TestPiOfflineSetup(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX fake Pi executable")
	}
	root := t.TempDir()
	t.Setenv("PATH", root)
	t.Setenv("PI_OFFLINE", "")
	if _, err := preparePi(IntegrationLaunchRequest{ExtraArgs: []string{"--offline"}}); err == nil || !strings.Contains(err.Error(), "Pi is missing") {
		t.Fatalf("expected an offline prerequisite error, got %v", err)
	}
}

func TestPiNativePluginSetup(t *testing.T) {
	bin, err := exec.LookPath("pi")
	if err != nil || checkPiVersion(bin) != nil {
		t.Skip("requires an installed Pi 1.5.2+")
	}
	root := t.TempDir()
	setTestHome(t, root)
	t.Chdir(root)
	agent := filepath.Join(root, "agent")
	t.Setenv("PI_CODING_AGENT_DIR", agent)
	plugin := filepath.Join(root, "plugin")
	for _, dir := range []string{agent, plugin, filepath.Join(root, ".pi")} {
		if err := os.Mkdir(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	entry := filepath.Join(plugin, "index.ts")
	for name, data := range map[string]string{
		entry:                                 "throw new Error('inventory must not execute user extensions')",
		filepath.Join(plugin, "package.json"): `{"name":"@ollama/pi","pi":{"extensions":["./index.ts"]}}`,
	} {
		if err := os.WriteFile(name, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		name     string
		settings any
		args     []string
	}{
		{"manual entry", map[string]any{"extensions": []string{entry}}, nil},
		{"local file package", map[string]any{"packages": []string{entry}}, nil},
		{"disabled package", map[string]any{"packages": []any{map[string]any{"source": plugin, "extensions": []string{}}}}, nil},
		{"disabled missing checkout", map[string]any{"packages": []any{map[string]any{"source": "git:github:ollama/pi-ollama@release", "autoload": false}}}, nil},
		{"explicit extension", map[string]any{}, []string{"-e", entry}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.settings)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(agent, "settings.json")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := configurePi(bin, IntegrationLaunchRequest{ConfigureOnly: true, ExtraArgs: tc.args}, true); err != nil {
				t.Fatal(err)
			}
			if after, err := os.ReadFile(path); err != nil || string(after) != string(data) {
				t.Fatalf("Pi settings changed: %s (%v)", after, err)
			}
		})
	}
	path := filepath.Join(agent, "settings.json")
	t.Run("configure saves the user default", func(t *testing.T) {
		const initial = `{"packages":["git:github.com/ollama/pi-ollama@release"],"defaultProvider":"other","defaultModel":"old","theme":"light","enabledModels":["other/old"]}`
		if err := os.WriteFile(path, []byte(initial), 0o600); err != nil {
			t.Fatal(err)
		}
		cmd := LaunchCmd(func(_ *cobra.Command, _ []string) error {
			t.Fatal("saving a Pi default must not require the Ollama daemon")
			return nil
		}, func(_ *cobra.Command) {})
		cmd.SetArgs([]string{"pi", "--config", "--model", "ollama/cloud/fixture", "--", "--offline"})
		for range 2 {
			if err := cmd.Execute(); err != nil {
				t.Fatal(err)
			}
		}
		var settings struct {
			DefaultProvider string
			DefaultModel    string
			Theme           string
			Packages        []string
			EnabledModels   []string
		}
		data, err := os.ReadFile(path)
		if err != nil || json.Unmarshal(data, &settings) != nil {
			t.Fatalf("read settings: %s (%v)", data, err)
		}
		if settings.DefaultProvider != "ollama" || settings.DefaultModel != "cloud/fixture" || settings.Theme != "light" ||
			!slices.Equal(settings.Packages, []string{piOllamaSource + "@release"}) ||
			!slices.Equal(settings.EnabledModels, []string{"other/old", "ollama/cloud/fixture"}) {
			t.Fatalf("unexpected saved settings: %s", data)
		}
	})
	t.Run("malformed settings are preserved", func(t *testing.T) {
		const malformed = `{"theme":`
		if err := os.WriteFile(path, []byte(malformed), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := configurePi(bin, IntegrationLaunchRequest{ConfigureOnly: true, ModelOverride: "local:latest"}, true); err == nil {
			t.Fatal("saved a model to invalid Pi settings")
		}
		if after, err := os.ReadFile(path); err != nil || string(after) != malformed {
			t.Fatalf("malformed settings changed: %s (%v)", after, err)
		}
	})
	t.Run("local and custom model IDs", func(t *testing.T) {
		const models = `{"providers":{"ollama":{"baseUrl":"http://localhost:11434/v1","api":"openai-completions","apiKey":"ollama","models":[{"id":"manual"}]}}}`
		if err := os.WriteFile(filepath.Join(agent, "models.json"), []byte(models), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(`{"packages":["git:github.com/ollama/pi-ollama"]}`), 0o600); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct{ model, want string }{
			{"ollama/local", "local:latest"},
			{"local:7b", "local:7b"},
			{"manual", "manual"},
		} {
			if err := configurePi(bin, IntegrationLaunchRequest{ConfigureOnly: true, ModelOverride: tc.model}, true); err != nil {
				t.Fatal(err)
			}
			var settings struct{ DefaultModel string }
			data, err := os.ReadFile(path)
			if err != nil || json.Unmarshal(data, &settings) != nil || settings.DefaultModel != tc.want {
				t.Fatalf("model %q: expected %q in settings: %s (%v)", tc.model, tc.want, data, err)
			}
		}
	})
	t.Run("install in an existing custom profile", func(t *testing.T) {
		if _, err := exec.LookPath("git"); err != nil {
			t.Skip("requires Git")
		}
		for _, args := range [][]string{
			{"init", "-b", "release"},
			{"add", "package.json", "index.ts"},
			{"-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-m", "fixture"},
		} {
			cmd := exec.Command("git", args...)
			cmd.Dir = plugin
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("git %s: %s (%v)", args[0], out, err)
			}
		}
		t.Setenv("GIT_CONFIG_COUNT", "1")
		t.Setenv("GIT_CONFIG_KEY_0", "url."+filepath.ToSlash(plugin)+".insteadOf")
		t.Setenv("GIT_CONFIG_VALUE_0", "https://github.com/ollama/pi-ollama")
		t.Setenv("GIT_ALLOW_PROTOCOL", "file")
		t.Setenv("npm_config_offline", "true")
		const custom = `{"theme":"light","extensions":["./my-extension.ts"]}`
		if err := os.WriteFile(path, []byte(custom), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := preparePi(IntegrationLaunchRequest{}); err != nil {
			t.Fatal(err)
		}
		if !piPluginDeclared() {
			t.Fatal("Pi did not register the plugin")
		}
		cached := filepath.Join(agent, "git", "github.com", "ollama", "pi-ollama", "index.ts")
		if _, err := os.Stat(cached); err != nil {
			t.Fatalf("Pi did not install the plugin: %v", err)
		}
		data, err := os.ReadFile(path)
		if err != nil || !strings.Contains(string(data), "my-extension.ts") || !strings.Contains(string(data), "light") {
			t.Fatalf("existing configuration lost: %s (%v)", data, err)
		}
	})
}
