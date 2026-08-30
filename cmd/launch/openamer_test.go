package launch

import (
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

func TestOpenAmerRegistry(t *testing.T) {
	spec, err := LookupIntegrationSpec("openamer")
	if err != nil {
		t.Fatal(err)
	}
	if spec.Name != openamerIntegrationName {
		t.Fatalf("canonical name = %q, want %q", spec.Name, openamerIntegrationName)
	}
	if spec.Runner.String() != "OpenAmer" {
		t.Fatalf("display name = %q", spec.Runner.String())
	}
	if got := strings.Join(spec.Install.Command, " "); got != "pip install --user openamer-agent" {
		t.Fatalf("install command = %q", got)
	}
	if spec.Install.URL != "https://github.com/openamer/openamer" {
		t.Fatalf("install URL = %q", spec.Install.URL)
	}
}

func TestOpenAmerConfigureWritesSettingsAndIsIdempotent(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:12345")

	settingsPath, err := openAmerSettingsPath()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o700); err != nil {
		t.Fatal(err)
	}
	existing := []byte("# keep-comment\ntheme: dark\n")
	if err := os.WriteFile(settingsPath, existing, 0o600); err != nil {
		t.Fatal(err)
	}

	models := []LaunchModel{
		{Name: "qwen3.5:latest", ContextLength: 262144, MaxOutputTokens: 32768, Capabilities: []model.Capability{model.CapabilityVision}},
		{Name: "kimi-k2.6:cloud", ContextLength: 262144, MaxOutputTokens: 262144},
	}
	oa := &OpenAmer{}
	if err := oa.ConfigureWithModels("qwen3.5", models); err != nil {
		t.Fatal(err)
	}
	first, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}

	if err := oa.ConfigureWithModels("qwen3.5", models); err != nil {
		t.Fatal(err)
	}
	second, _ := os.ReadFile(settingsPath)
	if string(first) != string(second) {
		t.Fatal("repeated configuration changed Ollama-managed files")
	}
	if !strings.Contains(string(first), "# keep-comment") {
		t.Fatalf("settings did not preserve existing content:\n%s", first)
	}

	var settings map[string]any
	if err := yaml.Unmarshal(first, &settings); err != nil {
		t.Fatal(err)
	}
	if theme, _ := settings["theme"].(string); theme != "dark" {
		t.Fatalf("unrelated settings were not preserved: %#v", settings["theme"])
	}
	provider, _ := settings["provider"].(map[string]any)
	if provider == nil {
		t.Fatal("provider settings were not written")
	}
	if provider["baseURL"] != "http://127.0.0.1:12345/v1" || provider["apiKeyEnv"] != openamerAPIKeyEnv {
		t.Fatalf("Ollama provider = %#v", provider)
	}
	if provider["model"] != "qwen3.5:latest" {
		t.Fatalf("provider model = %#v", provider["model"])
	}
	models_, _ := provider["models"].([]any)
	if len(models_) != 2 || models_[0] != "qwen3.5:latest" || models_[1] != "kimi-k2.6:cloud" {
		t.Fatalf("configured models = %#v", models_)
	}

	if got := oa.CurrentModel(); got != "qwen3.5:latest" {
		t.Fatalf("CurrentModel() = %q", got)
	}
}

func TestOpenAmerCurrentModelRejectsDrift(t *testing.T) {
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
	oa := &OpenAmer{}
	if err := oa.Configure("qwen3.5"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
	if got := oa.CurrentModel(); got != "" {
		t.Fatalf("CurrentModel() = %q for stale endpoint", got)
	}
}

func TestOpenAmerCurrentModelEmptyWithoutSettings(t *testing.T) {
	setTestHome(t, t.TempDir())
	if got := (&OpenAmer{}).CurrentModel(); got != "" {
		t.Fatalf("CurrentModel() = %q, want empty", got)
	}
}

func TestOpenAmerRunPassthrough(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	home := t.TempDir()
	setTestHome(t, home)
	binDir := t.TempDir()
	logPath := filepath.Join(home, "openamer-invocation")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$OA_TEST_LOG\"\nprintf '%s\\n' \"$OLLAMA_LAUNCH_OPENAMER_API_KEY\" >> \"$OA_TEST_LOG\"\n"
	bin := filepath.Join(binDir, "openamer")
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", strings.Join([]string{binDir, "/bin", "/usr/bin"}, string(os.PathListSeparator)))
	t.Setenv("OA_TEST_LOG", logPath)
	t.Setenv(openamerAPIKeyEnv, "do-not-keep")

	oa := &OpenAmer{}
	if err := oa.Run("qwen3.5", nil, []string{"--help"}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	want := "--help\nollama\n"
	if string(data) != want {
		t.Fatalf("invocation = %q, want %q", data, want)
	}
}

func TestEnsureOpenAmerInstalledUsesPip(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}

	home := t.TempDir()
	binDir := t.TempDir()
	logPath := filepath.Join(home, "pip-invocation")
	pip := filepath.Join(binDir, "pip")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$OA_PIP_LOG\"\nprintf '#!/bin/sh\\nexit 0\\n' > \"$OA_INSTALLED_BIN\"\nchmod +x \"$OA_INSTALLED_BIN\"\n"
	if err := os.WriteFile(pip, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	oaBin := filepath.Join(binDir, "openamer")
	t.Setenv("PATH", strings.Join([]string{binDir, "/bin", "/usr/bin"}, string(os.PathListSeparator)))
	t.Setenv("OA_PIP_LOG", logPath)
	t.Setenv("OA_INSTALLED_BIN", oaBin)

	restore := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restore()
	path, err := ensureOpenAmerInstalled()
	if err != nil {
		t.Fatal(err)
	}
	if path != oaBin {
		t.Fatalf("installed path = %q, want %q", path, oaBin)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "install\n--user\nopenamer-agent\n" {
		t.Fatalf("pip invocation = %q", data)
	}
}

func TestOpenAmerShimHandling(t *testing.T) {
	t.Run("passthrough on posix", func(t *testing.T) {
		cmd, err := openAmerShimCommand("/usr/local/bin/openamer", []string{"run"})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(cmd.Args, []string{"/usr/local/bin/openamer", "run"}) {
			t.Fatalf("command args = %v", cmd.Args)
		}
	})
}

func TestOpenAmerUpsertEnv(t *testing.T) {
	got := openAmerUpsertEnv([]string{"A=1", openamerAPIKeyEnv + "=old"}, openamerAPIKeyEnv, "ollama")
	want := []string{"A=1", openamerAPIKeyEnv + "=ollama"}
	if !slices.Equal(got, want) {
		t.Fatalf("env = %v, want %v", got, want)
	}
}

func TestOpenAmerLaunchArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses a POSIX shell test binary")
	}
	// Validate the shim detection helper on representative names.
	if !openAmerIsCommandShim("C:\\Python311\\Scripts\\openamer.cmd") {
		t.Fatal("expected .cmd to be detected as shim")
	}
	if openAmerIsCommandShim("/usr/bin/openamer") {
		t.Fatal("expected plain path to not be a shim")
	}
}
