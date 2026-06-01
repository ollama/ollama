package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const qwenOllamaEnvKey = "OLLAMA_API_KEY"

var qwenGOOS = runtime.GOOS

type Qwen struct{}

func (q *Qwen) String() string { return "Qwen Code" }

func (q *Qwen) findPath() (string, error) {
	if p, err := exec.LookPath("qwen"); err == nil {
		return p, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	var candidates []string
	switch qwenGOOS {
	case "darwin":
		candidates = []string{
			"/opt/homebrew/bin/qwen",
			"/usr/local/bin/qwen",
			filepath.Join(home, ".npm-global", "bin", "qwen"),
			filepath.Join(home, ".local", "bin", "qwen"),
			filepath.Join(home, "Library", "Application Support", "qwen", "bin", "qwen"),
		}
		candidates = append(candidates, qwenNVMCandidatePaths(home)...)
	case "windows":
		candidates = []string{
			filepath.Join(qwenWindowsAppData(home), "npm", "qwen.cmd"),
			filepath.Join(qwenWindowsAppData(home), "npm", "qwen.exe"),
			filepath.Join(qwenWindowsLocalAppData(home), "npm", "qwen.cmd"),
			filepath.Join(qwenWindowsLocalAppData(home), "npm", "qwen.exe"),
			filepath.Join(home, "AppData", "Local", "Programs", "qwen", "qwen.exe"),
			filepath.Join(home, "AppData", "Roaming", "qwen", "bin", "qwen.exe"),
		}
	default:
		candidates = []string{
			filepath.Join(home, ".npm-global", "bin", "qwen"),
			filepath.Join(home, ".local", "bin", "qwen"),
			filepath.Join(home, ".cargo", "bin", "qwen"),
			"/usr/local/bin/qwen",
		}
		candidates = append(candidates, qwenNVMCandidatePaths(home)...)
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("qwen binary not found (checked PATH and common npm install locations)")
}

func qwenNVMCandidatePaths(home string) []string {
	matches, err := filepath.Glob(filepath.Join(home, ".nvm", "versions", "node", "*", "bin", "qwen"))
	if err != nil {
		return nil
	}
	return matches
}

func qwenWindowsAppData(home string) string {
	if appData := os.Getenv("APPDATA"); appData != "" {
		return appData
	}
	return filepath.Join(home, "AppData", "Roaming")
}

func qwenWindowsLocalAppData(home string) string {
	if localAppData := os.Getenv("LOCALAPPDATA"); localAppData != "" {
		return localAppData
	}
	return filepath.Join(home, "AppData", "Local")
}

func ensureQwenInstalled() (string, error) {
	if path, err := (&Qwen{}).findPath(); err == nil {
		return path, nil
	}

	if err := checkQwenInstallerDependencies(); err != nil {
		return "", err
	}

	ok, err := ConfirmPrompt("Qwen Code is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("qwen installation cancelled")
	}

	bin, args, err := qwenInstallerCommand(qwenGOOS)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Qwen Code...\n")
	shimDir, cleanup, err := qwenInstallShimDir()
	if err != nil {
		return "", err
	}
	defer cleanup()

	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = qwenInstallerEnv(os.Environ(), shimDir)
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install qwen: %w", err)
	}

	path, err := (&Qwen{}).findPath()
	if err != nil {
		return "", fmt.Errorf("qwen was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sQwen Code installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}

func qwenInstallShimDir() (string, func(), error) {
	dir, err := os.MkdirTemp("", "ollama-qwen-install-*")
	if err != nil {
		return "", nil, err
	}

	cleanup := func() {
		_ = os.RemoveAll(dir)
	}

	if qwenGOOS == "windows" {
		for _, name := range []string{"qwen.cmd", "qwen.bat"} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte("@echo off\r\nexit /b 0\r\n"), 0o755); err != nil {
				cleanup()
				return "", nil, err
			}
		}
		return dir, cleanup, nil
	}

	if err := os.WriteFile(filepath.Join(dir, "qwen"), []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		cleanup()
		return "", nil, err
	}
	return dir, cleanup, nil
}

func qwenInstallerEnv(env []string, shimDir string) []string {
	out := make([]string, 0, len(env)+1)
	pathEntry := "PATH=" + shimDir
	for _, entry := range env {
		key, value, ok := strings.Cut(entry, "=")
		if ok && strings.EqualFold(key, "PATH") {
			pathEntry = key + "=" + shimDir + string(os.PathListSeparator) + value
			continue
		}
		out = append(out, entry)
	}
	return append(out, pathEntry)
}

func checkQwenInstallerDependencies() error {
	switch qwenGOOS {
	case "windows":
		if _, err := exec.LookPath("powershell"); err != nil {
			return fmt.Errorf("qwen is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch qwen")
		}
	default:
		var missing []string
		if _, err := exec.LookPath("curl"); err != nil {
			missing = append(missing, "curl: https://curl.se/")
		}
		if _, err := exec.LookPath("bash"); err != nil {
			missing = append(missing, "bash: https://www.gnu.org/software/bash/")
		}
		if len(missing) > 0 {
			return fmt.Errorf("qwen is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch qwen", strings.Join(missing, "\n  "))
		}
	}
	return nil
}

func qwenInstallerCommand(goos string) (string, []string, error) {
	switch goos {
	case "windows":
		return "powershell", []string{
			"-NoProfile",
			"-ExecutionPolicy",
			"Bypass",
			"-Command",
			"$installer = Join-Path $env:TEMP 'install-qwen.bat'; Invoke-WebRequest -UseBasicParsing -Uri 'https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen.bat' -OutFile $installer; $content = Get-Content -Raw -Path $installer; $content = $content -replace '(?m)^\\s*call qwen\\s*$', 'REM call qwen'; Set-Content -Path $installer -Value $content -Encoding ASCII; & $installer",
		}, nil
	case "darwin", "linux":
		return "bash", []string{
			"-c",
			"set -o pipefail; curl -fsSL https://qwen-code-assets.oss-cn-hangzhou.aliyuncs.com/installation/install-qwen.sh | sed '/log_info \"Starting Qwen Code...\"/,/exec qwen/d' | bash",
		}, nil
	default:
		return "", nil, fmt.Errorf("unsupported platform for qwen install: %s", goos)
	}
}

func (q *Qwen) Run(model string, _ []LaunchModel, args []string) error {
	qwenPath, err := q.findPath()
	if err != nil {
		return fmt.Errorf("qwen is not installed: %w", err)
	}

	cmd := exec.Command(qwenPath, qwenLaunchArgs(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = qwenLaunchEnv(model)
	return cmd.Run()
}

func (q *Qwen) Paths() []string {
	path, err := q.configPath()
	if err != nil {
		return nil
	}
	return []string{path}
}

func (q *Qwen) Configure(model string) error {
	if model == "" {
		return nil
	}

	configPath, err := q.configPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	cfg, err := q.readConfig()
	if err != nil {
		return err
	}

	applyQwenOllamaConfig(cfg, model)

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return fileutil.WriteWithBackup(configPath, data, "qwen")
}

func applyQwenOllamaConfig(cfg map[string]any, model string) {
	envCfg := qwenMap(cfg["env"])
	envCfg[qwenOllamaEnvKey] = "ollama"
	cfg["env"] = envCfg

	modelProviders := qwenMap(cfg["modelProviders"])
	modelProviders["openai"] = qwenMergeOpenAIProviders(modelProviders["openai"], qwenProvider(model))
	cfg["modelProviders"] = modelProviders

	security := qwenMap(cfg["security"])
	auth := qwenMap(security["auth"])
	auth["selectedType"] = "openai"
	auth["baseUrl"] = qwenBaseURL()
	security["auth"] = auth
	cfg["security"] = security

	modelCfg := qwenMap(cfg["model"])
	modelCfg["name"] = model
	cfg["model"] = modelCfg
}

func qwenMap(value any) map[string]any {
	if m, ok := value.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}

func qwenMergeOpenAIProviders(value any, provider map[string]any) []any {
	merged := []any{provider}
	for _, existing := range qwenProviderList(value) {
		if qwenIsOllamaProvider(existing) {
			continue
		}
		merged = append(merged, existing)
	}
	return merged
}

func qwenProviderList(value any) []any {
	switch providers := value.(type) {
	case []any:
		return providers
	case []map[string]any:
		out := make([]any, 0, len(providers))
		for _, provider := range providers {
			out = append(out, provider)
		}
		return out
	default:
		return nil
	}
}

func qwenIsOllamaProvider(value any) bool {
	provider, ok := value.(map[string]any)
	if !ok {
		return false
	}
	envKey, _ := provider["envKey"].(string)
	baseURL, _ := provider["baseUrl"].(string)
	return envKey == qwenOllamaEnvKey && strings.TrimRight(baseURL, "/") == qwenBaseURL()
}

func (q *Qwen) CurrentModel() string {
	cfg, err := q.readConfig()
	if err != nil {
		return ""
	}

	if modelCfg, ok := cfg["model"].(map[string]any); ok {
		if name, ok := modelCfg["name"].(string); ok {
			return strings.TrimSpace(name)
		}
	}

	modelProviders, ok := cfg["modelProviders"].(map[string]any)
	if !ok {
		return ""
	}

	providers, ok := modelProviders["openai"].([]any)
	if !ok || len(providers) == 0 {
		return ""
	}

	provider, ok := providers[0].(map[string]any)
	if !ok {
		return ""
	}

	name, _ := provider["id"].(string)
	return strings.TrimSpace(name)
}

func (q *Qwen) Onboard() error {
	return config.MarkIntegrationOnboarded("qwen")
}

func (q *Qwen) RequiresInteractiveOnboarding() bool { return false }

func (q *Qwen) configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("could not determine config path")
	}
	return filepath.Join(home, ".qwen", "settings.json"), nil
}

func (q *Qwen) readConfig() (map[string]any, error) {
	configPath, err := q.configPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]any{}, nil
		}
		return nil, err
	}

	cfg := map[string]any{}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse qwen config: %w", err)
	}

	return cfg, nil
}

func qwenBaseURL() string {
	return strings.TrimRight(envconfig.Host().String(), "/") + "/v1"
}

func qwenProvider(model string) map[string]any {
	return map[string]any{
		"id":      model,
		"name":    fmt.Sprintf("%s (Ollama)", model),
		"baseUrl": qwenBaseURL(),
		"envKey":  qwenOllamaEnvKey,
	}
}

func qwenLaunchArgs(model string, args []string) []string {
	launchArgs := append([]string{}, args...)
	if !qwenHasFlag(launchArgs, "--auth-type") {
		launchArgs = append([]string{"--auth-type", "openai"}, launchArgs...)
	}
	if model != "" && !qwenHasFlag(launchArgs, "--model", "-m") {
		launchArgs = append([]string{"--model", model}, launchArgs...)
	}
	return launchArgs
}

func qwenLaunchEnv(model string) []string {
	env := os.Environ()
	env = qwenUpsertEnv(env, "OPENAI_API_KEY", "ollama")
	env = qwenUpsertEnv(env, "OPENAI_BASE_URL", qwenBaseURL())
	if model != "" {
		env = qwenUpsertEnv(env, "OPENAI_MODEL", model)
	}
	return env
}

func qwenUpsertEnv(env []string, key, value string) []string {
	prefix := key + "="
	filtered := env[:0]
	for _, entry := range env {
		if strings.HasPrefix(entry, prefix) {
			continue
		}
		filtered = append(filtered, entry)
	}
	return append(filtered, prefix+value)
}

func qwenHasFlag(args []string, names ...string) bool {
	for _, arg := range args {
		for _, name := range names {
			if arg == name || strings.HasPrefix(arg, name+"=") {
				return true
			}
		}
	}
	return false
}
