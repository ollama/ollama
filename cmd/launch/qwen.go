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

	cfg["env"] = map[string]any{
		qwenOllamaEnvKey: "ollama",
	}
	cfg["modelProviders"] = map[string]any{
		"openai": []map[string]any{qwenProvider(model)},
	}
	cfg["security"] = map[string]any{
		"auth": map[string]any{
			"selectedType": "openai",
			"baseUrl":      qwenBaseURL(),
		},
	}
	cfg["model"] = map[string]any{
		"name": model,
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return fileutil.WriteWithBackup(configPath, data)
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
		return map[string]any{}, nil
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
