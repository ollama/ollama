package launch

import (
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const (
	qwenNpmPackage   = "@qwen-code/qwen-code"
	qwenOllamaEnvKey = "OLLAMA_API_KEY"
)

var (
	qwenPendingConfigPath    string
	qwenPendingConfigCreated bool
)

type Qwen struct{}

func (q *Qwen) String() string { return "Qwen Code CLI" }

func (q *Qwen) findPath() (string, error) {
	if p, err := exec.LookPath("qwen"); err == nil {
		return p, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"/opt/homebrew/bin/qwen",
			"/usr/local/bin/qwen",
			filepath.Join(home, ".local", "bin", "qwen"),
			filepath.Join(home, "Library", "Application Support", "qwen", "bin", "qwen"),
		}
	case "windows":
		candidates = []string{
			filepath.Join(home, "AppData", "Local", "Programs", "qwen", "qwen.exe"),
			filepath.Join(home, "AppData", "Roaming", "qwen", "bin", "qwen.exe"),
		}
	default:
		candidates = []string{
			filepath.Join(home, ".local", "bin", "qwen"),
			filepath.Join(home, ".cargo", "bin", "qwen"),
			"/usr/local/bin/qwen",
		}
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("qwen binary not found (checked PATH, ~/.local/bin, ~/.cargo/bin, /usr/local/bin)")
}

func (q *Qwen) ensureInstalled() error {
	if _, err := q.findPath(); err == nil {
		return nil
	}

	if err := qwenCheckInstallerDependencies(); err != nil {
		return err
	}

	ok, err := ConfirmPrompt("Qwen Code is not installed. Install now?")
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("qwen installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Qwen Code...\n")
	cmd := exec.Command("npm", "install", "-g", qwenNpmPackage+"@latest")
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		qwenCleanupPendingConfig()
		return fmt.Errorf("failed to install qwen: %w", err)
	}

	if _, err := q.findPath(); err != nil {
		return fmt.Errorf("qwen was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sQwen Code installed successfully%s\n\n", ansiGreen, ansiReset)
	qwenPendingConfigPath = ""
	qwenPendingConfigCreated = false
	return nil
}

func (q *Qwen) Run(model string, args []string) error {
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

func (q *Qwen) configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("could not determine config path")
	}
	return filepath.Join(home, ".qwen", "settings.json"), nil
}

func (q *Qwen) Paths() []string {
	path, err := q.configPath()
	if err != nil {
		return nil
	}
	return []string{path}
}

func (q *Qwen) Edit(models []string) error {
	if len(models) == 0 {
		return nil
	}

	configPath, err := q.configPath()
	if err != nil {
		return err
	}
	_, statErr := os.Stat(configPath)
	configExisted := statErr == nil

	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	var existingConfig map[string]any
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &existingConfig); err != nil {
			existingConfig = make(map[string]any)
		}
	} else if !os.IsNotExist(err) {
		return err
	}

	if existingConfig == nil {
		existingConfig = make(map[string]any)
	}

	envCfg, ok := existingConfig["env"].(map[string]any)
	if !ok {
		envCfg = make(map[string]any)
	}
	envCfg[qwenAPIKeyEnvKey()] = "ollama"
	existingConfig["env"] = envCfg

	modelProviders, ok := existingConfig["modelProviders"].(map[string]any)
	if !ok {
		modelProviders = make(map[string]any)
	}
	modelProviders["openai"] = qwenUpsertOpenAIProviders(modelProviders["openai"], models)
	existingConfig["modelProviders"] = modelProviders

	security, ok := existingConfig["security"].(map[string]any)
	if !ok {
		security = make(map[string]any)
		existingConfig["security"] = security
	}
	auth, ok := security["auth"].(map[string]any)
	if !ok {
		auth = make(map[string]any)
		security["auth"] = auth
	}
	auth["selectedType"] = "openai"
	auth["baseUrl"] = qwenBaseURL()

	if len(models) > 0 {
		modelCfg, ok := existingConfig["model"].(map[string]any)
		if !ok {
			modelCfg = make(map[string]any)
			existingConfig["model"] = modelCfg
		}
		modelCfg["name"] = models[0]
	}

	data, err := json.MarshalIndent(existingConfig, "", "  ")
	if err != nil {
		return err
	}

	if err := fileutil.WriteWithBackup(configPath, data); err != nil {
		return err
	}
	qwenPendingConfigPath = configPath
	qwenPendingConfigCreated = !configExisted
	return nil
}

func (q *Qwen) Models() []string {
	configPath, err := q.configPath()
	if err != nil {
		return nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil
	}

	var models []string
	seen := make(map[string]bool)

	modelCfg, ok := cfg["model"].(map[string]any)
	if ok {
		if name, ok := modelCfg["name"].(string); ok && name != "" {
			models = append(models, name)
			seen[name] = true
		}
	}

	modelProviders, ok := cfg["modelProviders"].(map[string]any)
	if !ok {
		return modelsOrNil(models)
	}

	for _, provider := range qwenOpenAIProviders(modelProviders["openai"]) {
		id, _ := provider["id"].(string)
		baseURL, _ := provider["baseUrl"].(string)
		if id == "" || !qwenIsOllamaProviderBaseURL(baseURL) || seen[id] {
			continue
		}
		models = append(models, id)
		seen[id] = true
	}

	return modelsOrNil(models)
}

func qwenBaseURL() string {
	return strings.TrimRight(envconfig.Host().String(), "/") + "/v1"
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

func qwenCheckInstallerDependencies() error {
	if _, err := exec.LookPath("npm"); err != nil {
		return fmt.Errorf("qwen is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch qwen")
	}
	return nil
}

func qwenCleanupPendingConfig() {
	if !qwenPendingConfigCreated || qwenPendingConfigPath == "" {
		return
	}
	_ = os.Remove(qwenPendingConfigPath)
	qwenPendingConfigPath = ""
	qwenPendingConfigCreated = false
}

func qwenAPIKeyEnvKey() string {
	return qwenOllamaEnvKey
}

func qwenOpenAIProviders(raw any) []map[string]any {
	switch v := raw.(type) {
	case []any:
		providers := make([]map[string]any, 0, len(v))
		for _, item := range v {
			provider, ok := item.(map[string]any)
			if !ok {
				continue
			}
			providers = append(providers, provider)
		}
		return providers
	case map[string]any:
		return []map[string]any{v}
	default:
		return nil
	}
}

func qwenUpsertOpenAIProviders(existing any, models []string) []any {
	providers := qwenOpenAIProviders(existing)
	selected := make(map[string]bool, len(models))
	for _, model := range models {
		selected[model] = true
	}

	baseURL := qwenBaseURL()
	envKey := qwenAPIKeyEnvKey()
	result := make([]any, 0, len(models)+len(providers))

	for _, model := range models {
		var provider map[string]any
		for _, existingProvider := range providers {
			id, _ := existingProvider["id"].(string)
			if id == model {
				provider = existingProvider
				break
			}
		}
		if provider == nil {
			provider = map[string]any{
				"id": model,
			}
		}
		provider["id"] = model
		provider["baseUrl"] = baseURL
		provider["envKey"] = envKey
		if _, ok := provider["name"].(string); !ok || provider["name"] == "" {
			provider["name"] = modelLabel(model)
		}
		if _, ok := provider["description"].(string); !ok || provider["description"] == "" {
			if description := qwenModelDescription(model); description != "" {
				provider["description"] = description
			}
		}
		result = append(result, provider)
	}

	for _, provider := range providers {
		id, _ := provider["id"].(string)
		rawBaseURL, _ := provider["baseUrl"].(string)
		if selected[id] || qwenIsOllamaProviderBaseURL(rawBaseURL) {
			continue
		}
		result = append(result, provider)
	}

	return result
}

func modelLabel(name string) string {
	return fmt.Sprintf("%s (Ollama)", name)
}

func qwenModelDescription(name string) string {
	for _, item := range recommendedModels {
		if item.Name == name {
			return item.Description
		}
	}
	return ""
}

func modelsOrNil(models []string) []string {
	if len(models) == 0 {
		return nil
	}
	return models
}

func qwenIsOllamaProviderBaseURL(baseURL string) bool {
	want, err := url.Parse(qwenBaseURL())
	if err != nil {
		return false
	}
	got, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return false
	}
	if !strings.EqualFold(got.Scheme, want.Scheme) {
		return false
	}
	if strings.TrimRight(got.EscapedPath(), "/") != strings.TrimRight(want.EscapedPath(), "/") {
		return false
	}
	if got.Port() != want.Port() {
		return false
	}
	return qwenEquivalentHosts(got.Hostname(), want.Hostname())
}

func qwenEquivalentHosts(a, b string) bool {
	if strings.EqualFold(a, b) {
		return true
	}
	return qwenIsLoopbackHost(a) && qwenIsLoopbackHost(b)
}

func qwenIsLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
