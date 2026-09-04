package launch

import (
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

	"github.com/ollama/ollama/types/model"
	"gopkg.in/yaml.v3"
)

func TestDeepSeekHarnessRegistry(t *testing.T) {
	spec, err := LookupIntegrationSpec("deepseek-harness")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Name != deepSeekHarnessIntegrationName {
		t.Fatalf("canonical name = %q, want %q", spec.Name, deepSeekHarnessIntegrationName)
	}
	if spec.Runner.String() != "DeepSeek Harness" {
		t.Fatalf("display name = %q", spec.Runner.String())
	}
	if got := strings.Join(spec.Install.Command, " "); got != "npm install -g @deepseek-ai/dsh@latest" {
		t.Fatalf("install command = %q", got)
	}
}

func TestDeepSeekHarnessConfigurePreservesSettingsAndIsIdempotent(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:12345")

	settingsPath, err := deepSeekHarnessSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o700); err != nil {
		t.Fatal(err)
	}
	existing := []byte("# keep-comment\ndefaults: &defaults\n    mode: dark\ntheme: *defaults\nagent-default-model:\n    # keep-reasoning-comment\n    reasoningEffort: high\nllm-pi-ai:\n    providers:\n        custom:\n            api: openai-completions\n            baseURL: https://example.invalid/v1\n            models:\n                - id: custom-model\n        ollama:\n            # keep-retry-comment\n            retryPolicy:\n                maxAttempts: 2\nweb-search-deepseek:\n    maxUses: 3\n")
	if err := os.WriteFile(settingsPath, existing, 0o600); err != nil {
		t.Fatal(err)
	}

	models := []LaunchModel{
		{Name: "qwen3.5:latest", ContextLength: 262144, MaxOutputTokens: 32768, Capabilities: []model.Capability{model.CapabilityVision}},
		{Name: "kimi-k2.6:cloud", ContextLength: 262144, MaxOutputTokens: 262144},
	}
	dsh := &DeepSeekHarness{}
	if err := dsh.ConfigureWithModels("qwen3.5", models); err != nil {
		t.Fatal(err)
	}
	firstSettings, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	patchPath, err := deepSeekHarnessPatchPath()
	if err != nil {
		t.Fatal(err)
	}
	firstPatch, err := os.ReadFile(patchPath)
	if err != nil {
		t.Fatal(err)
	}

	if err := dsh.ConfigureWithModels("qwen3.5", models); err != nil {
		t.Fatal(err)
	}
	secondSettings, _ := os.ReadFile(settingsPath)
	secondPatch, _ := os.ReadFile(patchPath)
	if string(firstSettings) != string(secondSettings) || string(firstPatch) != string(secondPatch) {
		t.Fatal("repeated configuration changed Ollama-managed files")
	}
	for _, preserved := range []string{"# keep-comment", "&defaults", "*defaults", "# keep-reasoning-comment", "# keep-retry-comment"} {
		if !strings.Contains(string(firstSettings), preserved) {
			t.Fatalf("settings did not preserve %q:\n%s", preserved, firstSettings)
		}
	}

	var settings map[string]any
	if err := yaml.Unmarshal(firstSettings, &settings); err != nil {
		t.Fatal(err)
	}
	if theme, _ := settings["theme"].(map[string]any); theme["mode"] != "dark" {
		t.Fatalf("unrelated settings were not preserved: %#v", settings["theme"])
	}
	selected, _ := settings["agent-default-model"].(map[string]any)
	if selected["provider"] != deepSeekHarnessProvider || selected["model"] != "qwen3.5:latest" {
		t.Fatalf("default model = %#v", selected)
	}
	if selected["reasoningEffort"] != "high" {
		t.Fatalf("default model settings were not preserved: %#v", selected)
	}
	llm, _ := settings["llm-pi-ai"].(map[string]any)
	providers, _ := llm["providers"].(map[string]any)
	if providers["custom"] == nil {
		t.Fatal("custom provider was removed")
	}
	provider, _ := providers[deepSeekHarnessProvider].(map[string]any)
	if provider["baseURL"] != "http://127.0.0.1:12345/v1" || provider["apiKeyEnv"] != deepSeekHarnessAPIKeyEnv {
		t.Fatalf("Ollama provider = %#v", provider)
	}
	retryPolicy, _ := provider["retryPolicy"].(map[string]any)
	if retryPolicy["maxAttempts"] != 2 {
		t.Fatalf("Ollama provider settings were not preserved: %#v", provider)
	}
	configuredModels, _ := provider["models"].([]any)
	if len(configuredModels) != 2 {
		t.Fatalf("configured models = %#v", configuredModels)
	}
	local, _ := configuredModels[0].(map[string]any)
	if local["id"] != "qwen3.5:latest" || local["contextWindow"] != 262144 || local["maxTokens"] != 32768 {
		t.Fatalf("local model = %#v", local)
	}
	if got, _ := local["input"].([]any); !slices.Equal(got, []any{"text", "image"}) {
		t.Fatalf("local model input = %#v", local["input"])
	}
	cloud, _ := configuredModels[1].(map[string]any)
	if cloud["id"] != "kimi-k2.6:cloud" || cloud["contextWindow"] != 262144 || cloud["maxTokens"] != 262144 {
		t.Fatalf("cloud model = %#v", cloud)
	}
	web, _ := settings[deepSeekHarnessWebSettings].(map[string]any)
	if web["baseURL"] != "http://127.0.0.1:12345/v1" || web["apiKeyEnv"] != deepSeekHarnessAPIKeyEnv || web["model"] != "qwen3.5:latest" {
		t.Fatalf("Ollama web search provider = %#v", web)
	}
	if web["maxUses"] != 3 {
		t.Fatalf("existing web search settings were not preserved: %#v", web)
	}

	var patches []map[string]any
	if err := yaml.Unmarshal(firstPatch, &patches); err != nil {
		t.Fatal(err)
	}
	if len(patches) != 1 || patches[0]["id"] != "settings" {
		t.Fatalf("patches = %#v", patches)
	}
	patchConfig, _ := patches[0]["config"].(map[string]any)
	if patchConfig["path"] != settingsPath {
		t.Fatalf("settings patch path = %#v", patchConfig["path"])
	}
	if got := dsh.CurrentModel(); got != "qwen3.5:latest" {
		t.Fatalf("CurrentModel() = %q", got)
	}
}

func TestDeepSeekHarnessConfigureSkipsWebSearchWhenCloudDisabled(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/status" {
			fmt.Fprint(w, `{"cloud":{"disabled":true,"source":"config"}}`)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("OLLAMA_HOST", srv.URL)

	settingsPath, err := deepSeekHarnessSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o700); err != nil {
		t.Fatal(err)
	}
	existing := []byte("web-search-deepseek:\n    maxUses: 3\n")
	if err := os.WriteFile(settingsPath, existing, 0o600); err != nil {
		t.Fatal(err)
	}

	dsh := &DeepSeekHarness{}
	if err := dsh.ConfigureWithModels("qwen3.5", []LaunchModel{{Name: "qwen3.5:latest"}}); err != nil {
		t.Fatal(err)
	}

	settings, err := readDeepSeekHarnessYAML(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	web, _ := settings[deepSeekHarnessWebSettings].(map[string]any)
	if len(web) != 1 || web["maxUses"] != 3 {
		t.Fatalf("web search settings = %#v", web)
	}
	if got := dsh.CurrentModel(); got != "qwen3.5:latest" {
		t.Fatalf("CurrentModel() = %q", got)
	}
}

func TestDeepSeekHarnessCurrentModelRejectsDrift(t *testing.T) {
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	dsh := &DeepSeekHarness{}
	if err := dsh.Configure("qwen3.5"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
	if got := dsh.CurrentModel(); got != "" {
		t.Fatalf("CurrentModel() = %q for stale endpoint", got)
	}
}

func TestDeepSeekHarnessCurrentModelRejectsPatchDrift(t *testing.T) {
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	dsh := &DeepSeekHarness{}
	if err := dsh.Configure("qwen3.5"); err != nil {
		t.Fatal(err)
	}
	patchPath, err := deepSeekHarnessPatchPath()
	if err != nil {
		t.Fatal(err)
	}

	for name, data := range map[string][]byte{
		"missing":   nil,
		"malformed": []byte("["),
		"wrong path": []byte(`- id: settings
  config:
    path: /tmp/not-managed.yaml
`),
	} {
		t.Run(name, func(t *testing.T) {
			if err := dsh.Configure("qwen3.5"); err != nil {
				t.Fatal(err)
			}
			if data == nil {
				if err := os.Remove(patchPath); err != nil {
					t.Fatal(err)
				}
			} else if err := os.WriteFile(patchPath, data, 0o600); err != nil {
				t.Fatal(err)
			}
			if got := dsh.CurrentModel(); got != "" {
				t.Fatalf("CurrentModel() = %q", got)
			}
		})
	}
}

func TestDeepSeekHarnessConfigureRejectsMalformedSettingsWithoutOverwrite(t *testing.T) {
	setTestHome(t, t.TempDir())
	settingsPath, err := deepSeekHarnessSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o700); err != nil {
		t.Fatal(err)
	}
	malformed := []byte("llm-pi-ai: [")
	if err := os.WriteFile(settingsPath, malformed, 0o600); err != nil {
		t.Fatal(err)
	}

	err = (&DeepSeekHarness{}).Configure("qwen3.5")
	if err == nil || !strings.Contains(err.Error(), "parse deepseek harness launch settings") {
		t.Fatalf("Configure() error = %v", err)
	}
	got, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, malformed) {
		t.Fatalf("malformed settings were overwritten: %q", got)
	}
}

func TestDeepSeekHarnessConfigureAcceptsNullSettings(t *testing.T) {
	setTestHome(t, t.TempDir())
	settingsPath, err := deepSeekHarnessSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(settingsPath, []byte("null\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := (&DeepSeekHarness{}).Configure("qwen3.5"); err != nil {
		t.Fatal(err)
	}
}

func TestDeepSeekHarnessRunUsesManagedPatchAndCredential(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	home := t.TempDir()
	setTestHome(t, home)
	binDir := t.TempDir()
	logPath := filepath.Join(home, "dsh-invocation")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$DSH_TEST_LOG\"\nprintf '%s\\n' \"$OLLAMA_LAUNCH_DSH_API_KEY\" >> \"$DSH_TEST_LOG\"\n"
	bin := filepath.Join(binDir, "dsh")
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", strings.Join([]string{binDir, "/bin", "/usr/bin"}, string(os.PathListSeparator)))
	t.Setenv("DSH_TEST_LOG", logPath)
	t.Setenv(deepSeekHarnessAPIKeyEnv, "do-not-keep")

	dsh := &DeepSeekHarness{}
	if err := dsh.Configure("qwen3.5"); err != nil {
		t.Fatal(err)
	}
	if err := dsh.Run("qwen3.5", nil, []string{"--port", "0"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	patchPath, _ := deepSeekHarnessPatchPath()
	want := "web\n--patch\n" + patchPath + "\n--port\n0\nollama\n"
	if string(data) != want {
		t.Fatalf("invocation = %q, want %q", data, want)
	}
}

func TestDeepSeekHarnessRejectsManagedPatchArgument(t *testing.T) {
	for _, args := range [][]string{{"--patch", "other.yml"}, {"--patch=other.yml"}} {
		if err := (&DeepSeekHarness{}).Run("qwen3.5", nil, args); err == nil || !strings.Contains(err.Error(), "manages --patch") {
			t.Fatalf("Run(%v) error = %v", args, err)
		}
	}
}

func TestEnsureDeepSeekHarnessInstalledUsesPublicNpmPackage(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	home := t.TempDir()
	binDir := t.TempDir()
	logPath := filepath.Join(home, "npm-invocation")
	npm := filepath.Join(binDir, "npm")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$DSH_NPM_LOG\"\nprintf '#!/bin/sh\\nexit 0\\n' > \"$DSH_INSTALLED_BIN\"\nchmod +x \"$DSH_INSTALLED_BIN\"\n"
	if err := os.WriteFile(npm, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	dshBin := filepath.Join(binDir, "dsh")
	t.Setenv("PATH", strings.Join([]string{binDir, "/bin", "/usr/bin"}, string(os.PathListSeparator)))
	t.Setenv("DSH_NPM_LOG", logPath)
	t.Setenv("DSH_INSTALLED_BIN", dshBin)

	restore := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restore()
	path, err := ensureDeepSeekHarnessInstalled()
	if err != nil {
		t.Fatal(err)
	}
	if path != dshBin {
		t.Fatalf("installed path = %q, want %q", path, dshBin)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "install\n-g\n@deepseek-ai/dsh@latest\n" {
		t.Fatalf("npm invocation = %q", data)
	}
}

func TestDeepSeekHarnessFallsBackToNpxAfterGlobalInstallFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	home := t.TempDir()
	setTestHome(t, home)
	binDir := t.TempDir()
	logPath := filepath.Join(home, "npx-invocation")
	for name, script := range map[string]string{
		"npm": "#!/bin/sh\nexit 1\n",
		"npx": "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$DSH_NPX_LOG\"\nprintf '%s\\n' \"$OLLAMA_LAUNCH_DSH_API_KEY\" >> \"$DSH_NPX_LOG\"\n",
	} {
		if err := os.WriteFile(filepath.Join(binDir, name), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", strings.Join([]string{binDir, "/bin", "/usr/bin"}, string(os.PathListSeparator)))
	t.Setenv("DSH_NPX_LOG", logPath)

	restore := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restore()
	path, err := ensureDeepSeekHarnessInstalled()
	if err != nil {
		t.Fatal(err)
	}
	if path != "" {
		t.Fatalf("fallback path = %q, want empty", path)
	}

	dsh := &DeepSeekHarness{}
	if err := dsh.Run("qwen3.5", nil, []string{"--port", "0"}); err != nil {
		t.Fatal(err)
	}
	patchPath, err := deepSeekHarnessPatchPath()
	if err != nil {
		t.Fatal(err)
	}
	want := "--yes\n@deepseek-ai/dsh@latest\nweb\n--patch\n" + patchPath + "\n--port\n0\nollama\n"
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != want {
		t.Fatalf("npx invocation = %q, want %q", data, want)
	}
}

func TestEnsureDeepSeekHarnessInstalledPreservesGlobalFailureWhenNpxIsUnavailable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	binDir := t.TempDir()
	npm := filepath.Join(binDir, "npm")
	if err := os.WriteFile(npm, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	originalLookPath := deepSeekHarnessLookPath
	deepSeekHarnessLookPath = func(file string) (string, error) {
		if file == "npm" {
			return npm, nil
		}
		return "", exec.ErrNotFound
	}
	t.Cleanup(func() { deepSeekHarnessLookPath = originalLookPath })

	restore := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restore()
	_, err := ensureDeepSeekHarnessInstalled()
	if err == nil || !strings.Contains(err.Error(), "failed to install deepseek harness") {
		t.Fatalf("error = %v", err)
	}
	if strings.Contains(err.Error(), "npx") {
		t.Fatalf("error exposes fallback: %v", err)
	}
}

func TestDeepSeekHarnessLaunchArgs(t *testing.T) {
	got := deepSeekHarnessLaunchArgs("/tmp/ollama.cordis.yml", []string{"--port", "0"})
	want := []string{"web", "--patch", "/tmp/ollama.cordis.yml", "--port", "0"}
	if !slices.Equal(got, want) {
		t.Fatalf("launch args = %v, want %v", got, want)
	}
}

func TestDeepSeekHarnessWindowsNodeShims(t *testing.T) {
	root := t.TempDir()
	node := filepath.Join(root, "node.exe")
	dsh := filepath.Join(root, "npm", "dsh.cmd")
	npm := filepath.Join(root, "node", "npm.cmd")
	npx := filepath.Join(root, "node", "npx.cmd")
	dshEntrypoint := filepath.Join(filepath.Dir(dsh), "node_modules", "@deepseek-ai", "dsh", "lib", "bin.js")
	npmEntrypoint := filepath.Join(filepath.Dir(npm), "node_modules", "npm", "bin", "npm-cli.js")
	npxEntrypoint := filepath.Join(filepath.Dir(npx), "node_modules", "npm", "bin", "npx-cli.js")
	for _, path := range []string{node, dsh, npm, npx, dshEntrypoint, npmEntrypoint, npxEntrypoint} {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, nil, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	originalGOOS := deepSeekHarnessGOOS
	originalLookPath := deepSeekHarnessLookPath
	deepSeekHarnessGOOS = "windows"
	deepSeekHarnessLookPath = func(file string) (string, error) {
		switch file {
		case "node":
			return node, nil
		case "npx":
			return npx, nil
		}
		return "", exec.ErrNotFound
	}
	t.Cleanup(func() {
		deepSeekHarnessGOOS = originalGOOS
		deepSeekHarnessLookPath = originalLookPath
	})

	t.Run("dsh", func(t *testing.T) {
		cmd, err := deepSeekHarnessExecutableCommand(dsh, []string{"web", "--port", "0", "a&b"})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{node, dshEntrypoint, "web", "--port", "0", "a&b"}
		if !slices.Equal(cmd.Args, want) {
			t.Fatalf("command args = %v, want %v", cmd.Args, want)
		}
	})

	t.Run("npm", func(t *testing.T) {
		cmd, err := deepSeekHarnessNpmCommand(npm, []string{"install", "-g", deepSeekHarnessNpmPackage})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{node, npmEntrypoint, "install", "-g", deepSeekHarnessNpmPackage}
		if !slices.Equal(cmd.Args, want) {
			t.Fatalf("command args = %v, want %v", cmd.Args, want)
		}
	})

	t.Run("npx", func(t *testing.T) {
		cmd, err := deepSeekHarnessLaunchCommand([]string{"web"})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{node, npxEntrypoint, "--yes", deepSeekHarnessNpmPackage, "web"}
		if !slices.Equal(cmd.Args, want) {
			t.Fatalf("command args = %v, want %v", cmd.Args, want)
		}
	})
}

func TestDeepSeekHarnessWindowsNodeShimRequiresEntrypoint(t *testing.T) {
	originalGOOS := deepSeekHarnessGOOS
	originalLookPath := deepSeekHarnessLookPath
	deepSeekHarnessGOOS = "windows"
	deepSeekHarnessLookPath = func(file string) (string, error) {
		if file == "node" {
			return filepath.Join(t.TempDir(), "node.exe"), nil
		}
		return "", exec.ErrNotFound
	}
	t.Cleanup(func() {
		deepSeekHarnessGOOS = originalGOOS
		deepSeekHarnessLookPath = originalLookPath
	})

	_, err := deepSeekHarnessExecutableCommand(filepath.Join(t.TempDir(), "dsh.cmd"), nil)
	if err == nil || !strings.Contains(err.Error(), "resolve Windows entrypoint") {
		t.Fatalf("error = %v", err)
	}
}

func TestDeepSeekHarnessInstallDependencyError(t *testing.T) {
	originalLookPath := deepSeekHarnessLookPath
	deepSeekHarnessLookPath = func(file string) (string, error) { return "", exec.ErrNotFound }
	t.Cleanup(func() { deepSeekHarnessLookPath = originalLookPath })
	_, err := ensureDeepSeekHarnessInstalled()
	if err == nil || !strings.Contains(err.Error(), "npm (Node.js) is required") {
		t.Fatalf("error = %v", err)
	}
}
