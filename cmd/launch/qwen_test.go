package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
)

func setQwenTestHome(t *testing.T, home string) {
	t.Helper()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
}

func TestQwenConfigure(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	q := &Qwen{}
	if err := q.Configure("gemma4"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(filepath.Join(tmpDir, ".qwen", "settings.json"))
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse config: %v", err)
	}

	envCfg := cfg["env"].(map[string]any)
	if envCfg[qwenOllamaEnvKey] != "ollama" {
		t.Fatalf("expected env[%q] to be ollama, got %v", qwenOllamaEnvKey, envCfg[qwenOllamaEnvKey])
	}

	modelCfg := cfg["model"].(map[string]any)
	if modelCfg["name"] != "gemma4" {
		t.Fatalf("expected model.name gemma4, got %v", modelCfg["name"])
	}

	security := cfg["security"].(map[string]any)
	auth := security["auth"].(map[string]any)
	if auth["selectedType"] != "openai" {
		t.Fatalf("expected auth.selectedType openai, got %v", auth["selectedType"])
	}
	if auth["baseUrl"] != qwenBaseURL() {
		t.Fatalf("expected auth.baseUrl %q, got %v", qwenBaseURL(), auth["baseUrl"])
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	openai := modelProviders["openai"].([]any)
	if len(openai) != 1 {
		t.Fatalf("expected one openai provider, got %d", len(openai))
	}

	provider := openai[0].(map[string]any)
	if provider["id"] != "gemma4" {
		t.Fatalf("expected provider id gemma4, got %v", provider["id"])
	}
	if provider["name"] != "gemma4 (Ollama)" {
		t.Fatalf("expected provider name %q, got %v", "gemma4 (Ollama)", provider["name"])
	}
	if provider["baseUrl"] != qwenBaseURL() {
		t.Fatalf("expected provider baseUrl %q, got %v", qwenBaseURL(), provider["baseUrl"])
	}
	if provider["envKey"] != qwenOllamaEnvKey {
		t.Fatalf("expected provider envKey %q, got %v", qwenOllamaEnvKey, provider["envKey"])
	}
}

func TestQwenConfigureBacksUpUnderIntegrationDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}
	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, []byte(`{"original":true}`), 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	if err := (&Qwen{}).Configure("gemma4"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	backups, err := filepath.Glob(filepath.Join(fileutil.BackupDir(), "qwen", "settings.json.*"))
	if err != nil {
		t.Fatalf("failed to glob backups: %v", err)
	}
	for _, backup := range backups {
		data, err := os.ReadFile(backup)
		if err != nil {
			t.Fatalf("failed to read backup: %v", err)
		}
		if string(data) == `{"original":true}` {
			return
		}
	}
	t.Fatalf("backup with original content not found in %v", backups)
}

func TestQwenConfigureMergesWithExistingSettings(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}
	configPath := filepath.Join(configDir, "settings.json")
	initialConfig := []byte(`{
  "theme": "dark",
  "env": {
    "OPENROUTER_API_KEY": "openrouter-key",
    "OLLAMA_API_KEY": "old-ollama-key"
  },
  "modelProviders": {
    "openai": [
      {
        "id": "old-ollama",
        "name": "old-ollama (Ollama)",
        "envKey": "OLLAMA_API_KEY",
        "baseUrl": "` + qwenBaseURL() + `"
      },
      {
        "id": "openrouter/model",
        "name": "OpenRouter Model",
        "envKey": "OPENROUTER_API_KEY",
        "baseUrl": "https://openrouter.ai/api/v1",
        "customField": "preserved"
      },
      {
        "id": "remote-ollama",
        "name": "Remote Ollama",
        "envKey": "OLLAMA_API_KEY",
        "baseUrl": "http://10.0.0.20:11434/v1"
      }
    ],
    "gemini": [
      {
        "id": "gemini-2.5-pro",
        "envKey": "GEMINI_API_KEY"
      }
    ]
  },
  "security": {
    "auth": {
      "selectedType": "qwen-oauth",
      "baseUrl": "https://old.example/v1",
      "customAuthField": "preserved"
    },
    "trustedFolders": ["/tmp/project"]
  },
  "model": {
    "name": "old-ollama",
    "generationConfig": {
      "temperature": 0.2
    }
  }
}`)
	if err := os.WriteFile(configPath, initialConfig, 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	if err := (&Qwen{}).Configure("gemma4"); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config: %v", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("failed to parse config: %v", err)
	}

	if cfg["theme"] != "dark" {
		t.Fatalf("expected top-level theme to be preserved, got %v", cfg["theme"])
	}

	envCfg := cfg["env"].(map[string]any)
	if envCfg["OPENROUTER_API_KEY"] != "openrouter-key" {
		t.Fatalf("expected OPENROUTER_API_KEY to be preserved, got %v", envCfg["OPENROUTER_API_KEY"])
	}
	if envCfg[qwenOllamaEnvKey] != "ollama" {
		t.Fatalf("expected %s to be updated, got %v", qwenOllamaEnvKey, envCfg[qwenOllamaEnvKey])
	}

	modelProviders := cfg["modelProviders"].(map[string]any)
	gemini := modelProviders["gemini"].([]any)
	if len(gemini) != 1 {
		t.Fatalf("expected gemini providers to be preserved, got %v", gemini)
	}
	openai := modelProviders["openai"].([]any)
	if len(openai) != 3 {
		t.Fatalf("expected new Ollama provider plus preserved OpenRouter and remote Ollama providers, got %v", openai)
	}
	ollamaProvider := openai[0].(map[string]any)
	if ollamaProvider["id"] != "gemma4" {
		t.Fatalf("expected Ollama provider to update to gemma4, got %v", ollamaProvider["id"])
	}
	openRouterProvider := openai[1].(map[string]any)
	if openRouterProvider["id"] != "openrouter/model" {
		t.Fatalf("expected OpenRouter provider to be preserved, got %v", openRouterProvider["id"])
	}
	if openRouterProvider["customField"] != "preserved" {
		t.Fatalf("expected OpenRouter custom field to be preserved, got %v", openRouterProvider["customField"])
	}
	remoteOllamaProvider := openai[2].(map[string]any)
	if remoteOllamaProvider["id"] != "remote-ollama" {
		t.Fatalf("expected remote Ollama provider to be preserved, got %v", remoteOllamaProvider["id"])
	}

	security := cfg["security"].(map[string]any)
	auth := security["auth"].(map[string]any)
	if auth["selectedType"] != "openai" {
		t.Fatalf("expected selectedType openai, got %v", auth["selectedType"])
	}
	if auth["baseUrl"] != qwenBaseURL() {
		t.Fatalf("expected auth.baseUrl %q, got %v", qwenBaseURL(), auth["baseUrl"])
	}
	if auth["customAuthField"] != "preserved" {
		t.Fatalf("expected custom auth field to be preserved, got %v", auth["customAuthField"])
	}
	if len(security["trustedFolders"].([]any)) != 1 {
		t.Fatalf("expected security.trustedFolders to be preserved, got %v", security["trustedFolders"])
	}

	modelCfg := cfg["model"].(map[string]any)
	if modelCfg["name"] != "gemma4" {
		t.Fatalf("expected model.name gemma4, got %v", modelCfg["name"])
	}
	generationConfig := modelCfg["generationConfig"].(map[string]any)
	if generationConfig["temperature"] != 0.2 {
		t.Fatalf("expected model generationConfig to be preserved, got %v", generationConfig)
	}
}

func TestQwenCurrentModel(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	q := &Qwen{}
	if got := q.CurrentModel(); got != "" {
		t.Fatalf("expected empty model without config, got %q", got)
	}

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, []byte(`{"model":{"name":"llama3.2"}}`), 0o644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	if got := q.CurrentModel(); got != "llama3.2" {
		t.Fatalf("expected current model llama3.2, got %q", got)
	}
}

func TestQwenCurrentModelFallsBackToProvider(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, []byte(`{"modelProviders":{"openai":[{"id":"mistral"}]}}`), 0o644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	if got := (&Qwen{}).CurrentModel(); got != "mistral" {
		t.Fatalf("expected provider fallback mistral, got %q", got)
	}
}

func TestQwenOnboard(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)

	if err := (&Qwen{}).Onboard(); err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	saved, err := config.LoadIntegration("qwen")
	if err != nil {
		t.Fatalf("failed to load integration config: %v", err)
	}
	if !saved.Onboarded {
		t.Fatal("expected qwen integration to be marked onboarded")
	}
}

func TestQwenIntegration(t *testing.T) {
	q := &Qwen{}

	t.Run("String", func(t *testing.T) {
		if got := q.String(); got != "Qwen Code" {
			t.Fatalf("String() = %q, want %q", got, "Qwen Code")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = q
	})

	t.Run("implements ManagedSingleModel", func(t *testing.T) {
		var _ ManagedSingleModel = q
	})

	t.Run("implements ManagedInteractiveOnboarding", func(t *testing.T) {
		var _ ManagedInteractiveOnboarding = q
	})
}

func TestQwenFindPath(t *testing.T) {
	q := &Qwen{}
	path, err := q.findPath()
	if err != nil {
		t.Skipf("qwen binary not found, skipping: %v", err)
	}
	if path == "" {
		t.Fatal("expected non-empty path")
	}
}

func TestQwenPaths(t *testing.T) {
	testDir := filepath.Join(t.TempDir(), "qwen-paths-test")
	setQwenTestHome(t, testDir)

	q := &Qwen{}
	os.MkdirAll(filepath.Join(testDir, ".qwen"), 0o755)
	os.WriteFile(filepath.Join(testDir, ".qwen", "settings.json"), []byte("{}"), 0o644)

	paths := q.Paths()
	if len(paths) != 1 {
		t.Fatalf("expected 1 path, got %v", paths)
	}

	want, err := filepath.EvalSymlinks(filepath.Join(testDir, ".qwen", "settings.json"))
	if err != nil {
		t.Fatalf("failed to resolve expected path: %v", err)
	}
	got, err := filepath.EvalSymlinks(paths[0])
	if err != nil {
		t.Fatalf("failed to resolve returned path: %v", err)
	}
	if got != want {
		t.Fatalf("expected user config path %s, got %s", want, got)
	}
}

func TestQwenLaunchArgs(t *testing.T) {
	got := qwenLaunchArgs("llama3.2", nil)
	want := []string{"--model", "llama3.2", "--auth-type", "openai"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}

	got = qwenLaunchArgs("llama3.2", []string{"--auth-type", "openai"})
	want = []string{"--model", "llama3.2", "--auth-type", "openai"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}

	got = qwenLaunchArgs("llama3.2", []string{"-m", "gemma4"})
	want = []string{"--auth-type", "openai", "-m", "gemma4"}
	if !slices.Equal(got, want) {
		t.Fatalf("expected %v, got %v", want, got)
	}
}

func TestQwenLaunchEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("OPENAI_MODEL", "")

	env := qwenLaunchEnv("llama3.2")
	if !slices.Contains(env, "OPENAI_API_KEY=ollama") {
		t.Fatalf("expected OPENAI_API_KEY override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_BASE_URL="+qwenBaseURL()) {
		t.Fatalf("expected OPENAI_BASE_URL override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_MODEL=llama3.2") {
		t.Fatalf("expected OPENAI_MODEL override, got %v", env)
	}
}

func TestQwenLaunchEnvOverridesExistingOpenAIEnv(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "real-key")
	t.Setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
	t.Setenv("OPENAI_MODEL", "gpt-4.1")

	env := qwenLaunchEnv("llama3.2")
	if !slices.Contains(env, "OPENAI_API_KEY=ollama") {
		t.Fatalf("expected OPENAI_API_KEY override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_BASE_URL="+qwenBaseURL()) {
		t.Fatalf("expected OPENAI_BASE_URL override, got %v", env)
	}
	if !slices.Contains(env, "OPENAI_MODEL=llama3.2") {
		t.Fatalf("expected OPENAI_MODEL override, got %v", env)
	}
}

func TestQwenRunDoesNotRewriteConfig(t *testing.T) {
	tmpDir := t.TempDir()
	setQwenTestHome(t, tmpDir)
	t.Chdir(tmpDir)

	qwenBinDir := filepath.Join(tmpDir, "bin")
	if err := os.MkdirAll(qwenBinDir, 0o755); err != nil {
		t.Fatalf("failed to create bin dir: %v", err)
	}

	qwenBin := filepath.Join(qwenBinDir, "qwen")
	qwenScript := "#!/bin/sh\nexit 0\n"
	if runtime.GOOS == "windows" {
		qwenBin = filepath.Join(qwenBinDir, "qwen.bat")
		qwenScript = "@echo off\r\nexit /b 0\r\n"
	}
	if err := os.WriteFile(qwenBin, []byte(qwenScript), 0o755); err != nil {
		t.Fatalf("failed to write fake qwen binary: %v", err)
	}
	if runtime.GOOS != "windows" {
		if err := os.Chmod(qwenBin, 0o755); err != nil {
			t.Fatalf("failed to chmod fake qwen binary: %v", err)
		}
	}

	t.Setenv("PATH", qwenBinDir+string(os.PathListSeparator)+os.Getenv("PATH"))

	configDir := filepath.Join(tmpDir, ".qwen")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("failed to create config dir: %v", err)
	}

	initialConfig := []byte(`{"model":{"name":"qwen3:32b"}}`)
	configPath := filepath.Join(configDir, "settings.json")
	if err := os.WriteFile(configPath, initialConfig, 0o644); err != nil {
		t.Fatalf("failed to write initial config: %v", err)
	}

	if err := (&Qwen{}).Run("qwen3:32b", nil, nil); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("failed to read config after run: %v", err)
	}
	if string(data) != string(initialConfig) {
		t.Fatalf("expected run not to rewrite config, got %s", string(data))
	}
}

func TestEnsureQwenInstalled(t *testing.T) {
	oldGOOS := qwenGOOS
	t.Cleanup(func() { qwenGOOS = oldGOOS })

	withConfirm := func(t *testing.T, fn func(prompt string) (bool, error)) {
		t.Helper()
		oldConfirm := DefaultConfirmPrompt
		DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
			return fn(prompt)
		}
		t.Cleanup(func() { DefaultConfirmPrompt = oldConfirm })
	}

	t.Run("already installed", func(t *testing.T) {
		setQwenTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		writeFakeBinary(t, tmpDir, "qwen")
		qwenGOOS = runtime.GOOS

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		bin, err := ensureQwenInstalled()
		if err != nil {
			t.Fatalf("ensureQwenInstalled() error = %v", err)
		}
		if filepath.Base(bin) == "" {
			t.Fatalf("expected qwen binary path, got %q", bin)
		}
	})

	t.Run("missing dependencies", func(t *testing.T) {
		setQwenTestHome(t, t.TempDir())
		t.Setenv("PATH", t.TempDir())
		qwenGOOS = "linux"

		withConfirm(t, func(prompt string) (bool, error) {
			t.Fatalf("did not expect prompt, got %q", prompt)
			return false, nil
		})

		_, err := ensureQwenInstalled()
		if err == nil || !strings.Contains(err.Error(), "required dependencies are missing") {
			t.Fatalf("expected missing dependency error, got %v", err)
		}
	})

	t.Run("missing and user declines install", func(t *testing.T) {
		setQwenTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		writeFakeBinary(t, tmpDir, "curl")
		writeFakeBinary(t, tmpDir, "bash")
		qwenGOOS = "linux"

		withConfirm(t, func(prompt string) (bool, error) {
			if !strings.Contains(prompt, "Qwen Code is not installed.") {
				t.Fatalf("unexpected prompt: %q", prompt)
			}
			return false, nil
		})

		_, err := ensureQwenInstalled()
		if err == nil || !strings.Contains(err.Error(), "installation cancelled") {
			t.Fatalf("expected cancellation error, got %v", err)
		}
	})

	t.Run("missing and user confirms unix install succeeds", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		homeDir := t.TempDir()
		setQwenTestHome(t, homeDir)
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		qwenGOOS = "linux"
		writeFakeBinary(t, tmpDir, "curl")

		installLog := filepath.Join(tmpDir, "bash.log")
		qwenPath := filepath.Join(homeDir, ".npm-global", "bin", "qwen")
		bashScript := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
if [ "$1" = "-c" ]; then
  /bin/mkdir -p %q
  /bin/cat > %q <<'EOS'
#!/bin/sh
exit 0
EOS
  /bin/chmod +x %q
fi
exit 0
`, installLog, filepath.Dir(qwenPath), qwenPath, qwenPath)
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte(bashScript), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		bin, err := ensureQwenInstalled()
		if err != nil {
			t.Fatalf("ensureQwenInstalled() error = %v", err)
		}
		if bin != qwenPath {
			t.Fatalf("bin = %q, want %q", bin, qwenPath)
		}

		logData, err := os.ReadFile(installLog)
		if err != nil {
			t.Fatalf("failed to read install log: %v", err)
		}
		if !strings.Contains(string(logData), "install-qwen.sh") {
			t.Fatalf("expected install-qwen.sh command in log, got:\n%s", string(logData))
		}
		if !strings.Contains(string(logData), "exec qwen/d") {
			t.Fatalf("expected command to remove installer auto-start block, got:\n%s", string(logData))
		}
	})

	t.Run("missing and user confirms windows install succeeds", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		homeDir := t.TempDir()
		setQwenTestHome(t, homeDir)
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		appData := filepath.Join(homeDir, "AppData", "Roaming")
		t.Setenv("APPDATA", appData)
		t.Setenv("LOCALAPPDATA", filepath.Join(homeDir, "AppData", "Local"))
		qwenGOOS = "windows"

		installLog := filepath.Join(tmpDir, "powershell.log")
		qwenPath := filepath.Join(appData, "npm", "qwen.cmd")
		powershellScript := fmt.Sprintf(`#!/bin/sh
echo "$@" >> %q
/bin/mkdir -p %q
/bin/cat > %q <<'EOS'
@echo off
exit /b 0
EOS
/bin/chmod +x %q
exit 0
`, installLog, filepath.Dir(qwenPath), qwenPath, qwenPath)
		if err := os.WriteFile(filepath.Join(tmpDir, "powershell"), []byte(powershellScript), 0o755); err != nil {
			t.Fatalf("failed to write fake powershell: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		bin, err := ensureQwenInstalled()
		if err != nil {
			t.Fatalf("ensureQwenInstalled() error = %v", err)
		}
		if bin != qwenPath {
			t.Fatalf("bin = %q, want %q", bin, qwenPath)
		}

		logData, err := os.ReadFile(installLog)
		if err != nil {
			t.Fatalf("failed to read install log: %v", err)
		}
		if !strings.Contains(string(logData), "install-qwen.bat") {
			t.Fatalf("expected install-qwen.bat command in log, got:\n%s", string(logData))
		}
		if !strings.Contains(string(logData), "REM call qwen") {
			t.Fatalf("expected command to replace installer auto-start call, got:\n%s", string(logData))
		}
	})

	t.Run("install command fails", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("uses POSIX shell fake binaries")
		}

		setQwenTestHome(t, t.TempDir())
		tmpDir := t.TempDir()
		t.Setenv("PATH", tmpDir)
		qwenGOOS = "linux"
		writeFakeBinary(t, tmpDir, "curl")
		if err := os.WriteFile(filepath.Join(tmpDir, "bash"), []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
			t.Fatalf("failed to write fake bash: %v", err)
		}

		withConfirm(t, func(prompt string) (bool, error) {
			return true, nil
		})

		_, err := ensureQwenInstalled()
		if err == nil || !strings.Contains(err.Error(), "failed to install qwen") {
			t.Fatalf("expected install failure error, got %v", err)
		}
	})
}

func TestQwenFindPathFallbacks(t *testing.T) {
	oldGOOS := qwenGOOS
	t.Cleanup(func() { qwenGOOS = oldGOOS })

	t.Run("unix npm global bin", func(t *testing.T) {
		homeDir := t.TempDir()
		setQwenTestHome(t, homeDir)
		t.Setenv("PATH", t.TempDir())
		qwenGOOS = "linux"

		target := filepath.Join(homeDir, ".npm-global", "bin", "qwen")
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			t.Fatalf("failed to create qwen dir: %v", err)
		}
		if err := os.WriteFile(target, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatalf("failed to write qwen binary: %v", err)
		}

		got, err := (&Qwen{}).findPath()
		if err != nil {
			t.Fatalf("findPath() error = %v", err)
		}
		if got != target {
			t.Fatalf("findPath() = %q, want %q", got, target)
		}
	})

	t.Run("windows appdata npm shim", func(t *testing.T) {
		homeDir := t.TempDir()
		setQwenTestHome(t, homeDir)
		t.Setenv("PATH", t.TempDir())
		appData := filepath.Join(homeDir, "AppData", "Roaming")
		t.Setenv("APPDATA", appData)
		t.Setenv("LOCALAPPDATA", filepath.Join(homeDir, "AppData", "Local"))
		qwenGOOS = "windows"

		target := filepath.Join(appData, "npm", "qwen.cmd")
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			t.Fatalf("failed to create qwen dir: %v", err)
		}
		if err := os.WriteFile(target, []byte("@echo off\r\nexit /b 0\r\n"), 0o755); err != nil {
			t.Fatalf("failed to write qwen shim: %v", err)
		}

		got, err := (&Qwen{}).findPath()
		if err != nil {
			t.Fatalf("findPath() error = %v", err)
		}
		if got != target {
			t.Fatalf("findPath() = %q, want %q", got, target)
		}
	})

	t.Run("unix nvm npm bin", func(t *testing.T) {
		homeDir := t.TempDir()
		setQwenTestHome(t, homeDir)
		t.Setenv("PATH", t.TempDir())
		qwenGOOS = "linux"

		target := filepath.Join(homeDir, ".nvm", "versions", "node", "v20.18.1", "bin", "qwen")
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			t.Fatalf("failed to create qwen dir: %v", err)
		}
		if err := os.WriteFile(target, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
			t.Fatalf("failed to write qwen binary: %v", err)
		}

		got, err := (&Qwen{}).findPath()
		if err != nil {
			t.Fatalf("findPath() error = %v", err)
		}
		if got != target {
			t.Fatalf("findPath() = %q, want %q", got, target)
		}
	})
}

func TestQwenInstallShimDir(t *testing.T) {
	oldGOOS := qwenGOOS
	t.Cleanup(func() { qwenGOOS = oldGOOS })

	t.Run("unix shim", func(t *testing.T) {
		qwenGOOS = "linux"
		dir, cleanup, err := qwenInstallShimDir()
		if err != nil {
			t.Fatalf("qwenInstallShimDir() error = %v", err)
		}
		defer cleanup()

		if _, err := os.Stat(filepath.Join(dir, "qwen")); err != nil {
			t.Fatalf("expected qwen shim: %v", err)
		}
	})

	t.Run("windows shim", func(t *testing.T) {
		qwenGOOS = "windows"
		dir, cleanup, err := qwenInstallShimDir()
		if err != nil {
			t.Fatalf("qwenInstallShimDir() error = %v", err)
		}
		defer cleanup()

		for _, name := range []string{"qwen.cmd", "qwen.bat"} {
			if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
				t.Fatalf("expected %s shim: %v", name, err)
			}
		}
	})
}

func TestQwenInstallerEnvPrependsShimPath(t *testing.T) {
	env := qwenInstallerEnv([]string{"FOO=bar", "PATH=/usr/bin"}, "/tmp/qwen-shim")
	if !slices.Contains(env, "FOO=bar") {
		t.Fatalf("expected unrelated env to be preserved, got %v", env)
	}
	if !slices.Contains(env, "PATH=/tmp/qwen-shim"+string(os.PathListSeparator)+"/usr/bin") {
		t.Fatalf("expected shim path to be prepended, got %v", env)
	}

	env = qwenInstallerEnv([]string{"Path=C:\\Windows"}, "C:\\qwen-shim")
	if !slices.Contains(env, "Path=C:\\qwen-shim"+string(os.PathListSeparator)+"C:\\Windows") {
		t.Fatalf("expected existing Path casing to be preserved, got %v", env)
	}
}

func TestQwenInstallerCommand(t *testing.T) {
	tests := []struct {
		name      string
		goos      string
		wantBin   string
		wantParts []string
		wantErr   bool
	}{
		{
			name:      "linux",
			goos:      "linux",
			wantBin:   "bash",
			wantParts: []string{"-c", "install-qwen.sh", "sed", "exec qwen/d"},
		},
		{
			name:      "darwin",
			goos:      "darwin",
			wantBin:   "bash",
			wantParts: []string{"-c", "install-qwen.sh", "sed", "exec qwen/d"},
		},
		{
			name:      "windows",
			goos:      "windows",
			wantBin:   "powershell",
			wantParts: []string{"-Command", "-UseBasicParsing", "-OutFile", "Get-Content -Raw", "install-qwen.bat", "REM call qwen"},
		},
		{
			name:    "unsupported",
			goos:    "freebsd",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin, args, err := qwenInstallerCommand(tt.goos)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("qwenInstallerCommand() error = %v", err)
			}
			if bin != tt.wantBin {
				t.Fatalf("bin = %q, want %q", bin, tt.wantBin)
			}
			joined := strings.Join(args, " ")
			for _, part := range tt.wantParts {
				if !strings.Contains(joined, part) {
					t.Fatalf("args %q missing %q", joined, part)
				}
			}
		})
	}
}
