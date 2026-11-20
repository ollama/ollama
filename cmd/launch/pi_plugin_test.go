package launch

import (
	"context"
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

func TestPiPluginLaunchInstallsOnceUpdatesAndKeepsBareLaunchDefaults(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX fake Pi executable")
	}
	root := t.TempDir()
	setTestHome(t, root)
	plugin := filepath.Join(root, "plugin")
	logPath := filepath.Join(root, "pi.log")
	if err := os.MkdirAll(plugin, 0o755); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(plugin, "package.json")
	marker := filepath.Join(plugin, "installed")
	if err := os.WriteFile(manifest, []byte(`{"name":"@ollama/pi"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(plugin, "index.ts"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	script := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
if [ "$1" = "--version" ]; then echo 1.5.2; exit 0; fi
if [ "$1" = "list" ]; then
  if [ -f %q ]; then
    echo '  '%q
    if [ "$FORCE_COLOR" = "1" ]; then printf '\033[2m    %%s\033[22m\n' %q; else echo '    '%q; fi
  fi
  exit 0
fi
if [ "$1" = "install" ] && [ "$2" = %q ]; then : > %q; exit 0; fi
if [ "$1" = "update" ] && [ "$2" = %q ]; then exit 0; fi
echo "color=$FORCE_COLOR" >> %q
if [ "$#" = 0 ]; then echo launched >> %q; echo "launch=$OLLAMA_LAUNCH_PI" >> %q; exit 0; fi
if [ "$1" = "--model" ]; then echo "launch=$OLLAMA_LAUNCH_PI" >> %q; exit 0; fi
exit 1
`, logPath, marker, piOllamaSource, plugin, plugin, piOllamaSource, marker, piOllamaSource, logPath, logPath, logPath, logPath)
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
	settings := `{"defaultProvider":"other","defaultModel":"keep","packages":["existing"]}`
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
	for range 2 {
		if err := LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: "pi"}); err != nil {
			t.Fatal(err)
		}
	}
	log, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(string(log), "install "+piOllamaSource+"\n") != 1 {
		t.Fatalf("plugin was not installed exactly once:\n%s", log)
	}
	if strings.Count(string(log), "update "+piOllamaSource+"\n") != 1 {
		t.Fatalf("the existing plugin must be updated on the second launch:\n%s", log)
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
	if strings.Count(string(log), "update "+piOllamaSource+"\n") != 2 {
		t.Fatalf("explicit model launches must update the plugin too:\n%s", log)
	}
}

func TestPiPluginUpdateFailureKeepsInstalledPlugin(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX fake Pi executable")
	}
	root := t.TempDir()
	setTestHome(t, root)
	t.Setenv("PATH", root)
	plugin := filepath.Join(root, "plugin")
	if err := os.MkdirAll(plugin, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{"package.json": `{"name":"@ollama/pi"}`, "index.ts": ""} {
		if err := os.WriteFile(filepath.Join(plugin, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	logPath := filepath.Join(root, "pi.log")
	script := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
if [ "$1" = "--version" ]; then echo 1.5.2; exit 0; fi
if [ "$1" = "list" ]; then printf 'User packages:\n  %%s\n    %%s\n' %q %q; exit 0; fi
if [ "$1" = "update" ]; then echo 'network unavailable' >&2; exit 1; fi
if [ "$1" = "--continue" ]; then exit 0; fi
exit 1
`, logPath, piOllamaSource, plugin)
	if err := os.WriteFile(filepath.Join(root, "pi"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := (&Pi{}).Run("", nil, []string{"--continue"}); err != nil {
		t.Fatal(err)
	}
	log, err := os.ReadFile(logPath)
	if err != nil || string(log) != "--version\nlist\nupdate "+piOllamaSource+"\n--continue\n" {
		t.Fatalf("a failed update must still launch the installed plugin: %s (%v)", log, err)
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

func TestPiPluginRecognizesLocalInstallRegardlessOfVersion(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "index.ts"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		manifest string
		want     bool
	}{
		{`{"name":"@ollama/pi","version":"0.0.0"}`, true},
		{`{"name":"@ollama/pi","version":"0.1.0"}`, true},
		{`{"name":"@ollama/pi","version":"0.2.0"}`, true},
		{`{"name":"@ollama/pi"}`, true},
		{`{"name":"another-plugin","version":"0.2.0"}`, false},
	} {
		if err := os.WriteFile(filepath.Join(root, "package.json"), []byte(tc.manifest), 0o644); err != nil {
			t.Fatal(err)
		}
		if installed := piOllamaPackageSource("User packages:\n  "+root+"\n") != ""; installed != tc.want {
			t.Fatalf("manifest %s: installed %v, want %v", tc.manifest, installed, tc.want)
		}
	}
}

func TestPiPluginPackageSourcePreservesConfiguredRefsAndFilters(t *testing.T) {
	root := t.TempDir()
	for name, data := range map[string]string{"package.json": `{"name":"@ollama/pi"}`, "index.ts": ""} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, source := range []string{
		piOllamaSource,
		"https://github.com/ollama/pi-ollama",
		piOllamaSource + "@v0.2.0",
		piOllamaSource + " (filtered)",
		"../local-checkout",
	} {
		output := "User packages:\n  npm:unrelated\n    /missing-package\n  " + source + "\n    " + root + "\n"
		if got := piOllamaPackageSource(output); got != strings.TrimSuffix(source, " (filtered)") {
			t.Fatalf("package source = %q, want %q", got, source)
		}
	}
}
