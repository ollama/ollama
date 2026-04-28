package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
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

func (q *Qwen) buildEnv(model string) []string {
	host := strings.TrimRight(envconfig.Host().String(), "/")
	base := host + "/v1"

	env := os.Environ()

	// Filter out existing OPENAI_* vars to force Ollama-specific values
	filteredEnv := make([]string, 0, len(env))
	for _, e := range env {
		if !strings.HasPrefix(e, "OPENAI_API_KEY=") &&
			!strings.HasPrefix(e, "OPENAI_BASE_URL=") &&
			!strings.HasPrefix(e, "OPENAI_MODEL=") {
			filteredEnv = append(filteredEnv, e)
		}
	}
	env = filteredEnv

	// Force Ollama-specific values for the child process
	env = append(env, "OPENAI_API_KEY=dummy")
	env = append(env, "OPENAI_BASE_URL="+base)
	if model != "" {
		env = append(env, "OPENAI_MODEL="+model)
	}

	return env
}

func (q *Qwen) Run(model string, args []string) error {
	qwenPath, err := q.findPath()
	if err != nil {
		return fmt.Errorf("qwen is not installed: %w", err)
	}

	cmd := exec.Command(qwenPath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	cmd.Env = q.buildEnv(model)
	return cmd.Run()
}

func (q *Qwen) configPath() (string, error) {
	// Check for existing project-local config first
	if cwd, err := os.Getwd(); err == nil {
		projectPath := filepath.Join(cwd, ".qwen", "settings.json")
		if _, err := os.Stat(projectPath); err == nil {
			return projectPath, nil
		}
	}

	// Check for existing user config next
	home, err := os.UserHomeDir()
	if err == nil {
		userPath := filepath.Join(home, ".qwen", "settings.json")
		if _, err := os.Stat(userPath); err == nil {
			return userPath, nil
		}
	}

	// Default to project-local config (will be created by Edit())
	if cwd, err := os.Getwd(); err == nil {
		return filepath.Join(cwd, ".qwen", "settings.json"), nil
	}

	// Fall back to user scope if cwd fails
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".qwen", "settings.json"), nil
	}

	return "", fmt.Errorf("could not determine config path")
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

	host := strings.TrimRight(envconfig.Host().String(), "/")
	base := host + "/v1"
	providerID := "ollama"
	if len(models) > 0 {
		providerID = models[0]
	}

	modelProviders, ok := existingConfig["modelProviders"].(map[string]any)
	if !ok {
		modelProviders = make(map[string]any)
	}
	if existingConfig["modelProviders"] == nil {
		existingConfig["modelProviders"] = modelProviders
	}

	var openaiProviders []any
	switch v := modelProviders["openai"].(type) {
	case []any:
		openaiProviders = v
	case map[string]any:
		openaiProviders = []any{v}
	default:
		openaiProviders = []any{}
	}

	found := false
	for i, item := range openaiProviders {
		if provider, ok := item.(map[string]any); ok {
			if id, ok := provider["id"].(string); ok && id == providerID {
				provider["baseUrl"] = base
				provider["envKey"] = "OPENAI_API_KEY"
				openaiProviders[i] = provider
				found = true
				break
			}
		}
	}

	if !found {
		newProvider := map[string]any{
			"id":      providerID,
			"envKey":  "OPENAI_API_KEY",
			"baseUrl": base,
		}
		openaiProviders = append(openaiProviders, newProvider)
	}

	modelProviders["openai"] = openaiProviders
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

	return fileutil.WriteWithBackup(configPath, data)
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

	modelCfg, ok := cfg["model"].(map[string]any)
	if !ok {
		return nil
	}

	name, ok := modelCfg["name"].(string)
	if !ok || name == "" {
		return nil
	}

	return []string{name}
}
