package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"gopkg.in/yaml.v3"
)

const (
	openamerIntegrationName = "openamer"
	openamerProvider        = "ollama"
	openamerAPIKeyEnv       = "OLLAMA_LAUNCH_OPENAMER_API_KEY"
	openamerBinary          = "openamer"
	openamerPipPackage      = "openamer-agent"
)

var (
	openamerLookPath = exec.LookPath
	openamerCommand  = exec.Command
	openamerGOOS     = runtime.GOOS
)

// OpenAmer is the Ollama-managed OpenAmer integration. OpenAmer is a personal
// AI agent that learns across sessions. Ollama redirects only the launch-owned
// settings document for this invocation; the user's normal config, memories,
// skills, and credentials remain available and untouched.
type OpenAmer struct{}

func (o *OpenAmer) String() string { return "OpenAmer" }

func (o *OpenAmer) Run(_ string, _ []LaunchModel, args []string) error {
	bin, err := openAmerLaunchCommand(args)
	if err != nil {
		return err
	}
	bin.Env = openAmerLaunchEnv(os.Environ())
	bin.Stdin = os.Stdin
	bin.Stdout = os.Stdout
	bin.Stderr = os.Stderr
	return bin.Run()
}

func openAmerLaunchCommand(args []string) (*exec.Cmd, error) {
	path, err := openamerLookPath(openamerBinary)
	if err != nil {
		return nil, fmt.Errorf("openamer is not installed: %w", err)
	}
	return openAmerShimCommand(path, args)
}

// Windows pip console scripts are .exe launchers but some users install via
// wrappers (.cmd/.bat shims), which cannot be passed safely to CreateProcess
// with an argv. Route through the underlying Python interpreter when a shim is
// detected so passthrough arguments remain data rather than cmd.exe syntax.
func openAmerShimCommand(shim string, args []string) (*exec.Cmd, error) {
	if openamerGOOS != "windows" || !openAmerIsCommandShim(shim) {
		return openamerCommand(shim, args...), nil
	}

	script := strings.TrimSuffix(shim, filepath.Ext(shim))
	// pip console scripts on Windows ship a shimless companion script
	// without a file extension (e.g. ...\Scripts\openamer) that Python can
	// execute directly.
	entrypoint := script
	if _, err := os.Stat(entrypoint); err != nil {
		return nil, fmt.Errorf("resolve Windows entrypoint for %s: %w", filepath.Base(shim), err)
	}

	python, err := openamerLookPath("python")
	if err != nil {
		return nil, fmt.Errorf("python is required to run %s on Windows: %w", filepath.Base(shim), err)
	}
	return openamerCommand(python, append([]string{entrypoint}, args...)...), nil
}

func openAmerIsCommandShim(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".cmd" || ext == ".bat"
}

func openAmerUpsertEnv(env []string, key, value string) []string {
	prefix := key + "="
	out := make([]string, 0, len(env)+1)
	for _, entry := range env {
		if strings.HasPrefix(entry, prefix) {
			continue
		}
		out = append(out, entry)
	}
	return append(out, prefix+value)
}

func openAmerLaunchEnv(env []string) []string {
	return openAmerUpsertEnv(env, openamerAPIKeyEnv, "ollama")
}

func ensureOpenAmerInstalled() (string, error) {
	if path, err := openamerLookPath(openamerBinary); err == nil {
		return path, nil
	}
	if _, err := openamerLookPath("python"); err != nil {
		if _, err := openamerLookPath("python3"); err != nil {
			return "", fmt.Errorf("openamer is not installed and Python (pip) is required\n\nInstall Python first:\n  https://www.python.org/downloads/\n\nOr use the installer:\n  curl -fsSL https://openamer.dev/scripts/install.sh | bash\n\nThen re-run:\n  ollama launch openamer")
		}
	}

	ok, err := ConfirmPrompt("OpenAmer is not installed. Install with pip?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("openamer installation cancelled")
	}

	fmt.Fprintln(os.Stderr, "\nInstalling OpenAmer...")
	pip := "pip"
	if _, err := openamerLookPath("pip"); err != nil {
		pip = "pip3"
		if _, err := openamerLookPath(pip); err != nil {
			pip = "python"
		}
	}
	pipArgs := []string{"install", "--user", openamerPipPackage}
	if pip == "python" {
		pipArgs = append([]string{"-m", "pip"}, pipArgs...)
	}
	cmd := openamerCommand(pip, pipArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install openamer: %w", err)
	}

	path, err := openamerLookPath(openamerBinary)
	if err != nil {
		return "", fmt.Errorf("openamer was installed but %s was not found on PATH\n\nYou may need to restart your shell", openamerBinary)
	}
	fmt.Fprintf(os.Stderr, "%sOpenAmer installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}

func (o *OpenAmer) Paths() []string {
	settingsPath, err := openAmerSettingsPath()
	if err != nil {
		return nil
	}
	return []string{settingsPath}
}

func (o *OpenAmer) Configure(modelName string) error {
	return o.ConfigureWithModels(modelName, []LaunchModel{fallbackLaunchModel(modelName)})
}

func (o *OpenAmer) ConfigureWithModels(primary string, models []LaunchModel) error {
	if strings.TrimSpace(primary) == "" {
		return nil
	}
	if len(models) == 0 {
		models = []LaunchModel{fallbackLaunchModel(primary)}
	}
	if selected, ok := findLaunchModel(models, primary); ok {
		primary = selected.Name
	}

	settingsPath, err := openAmerSettingsPath()
	if err != nil {
		return err
	}
	settings, err := readOpenAmerYAML(settingsPath)
	if err != nil {
		return fmt.Errorf("parse openamer launch settings: %w", err)
	}
	if err := applyOpenAmerSettings(settings, primary, models); err != nil {
		return err
	}

	data, err := yaml.Marshal(settings)
	if err != nil {
		return err
	}
	return writeOpenAmerFile(settingsPath, data)
}

func readOpenAmerYAML(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]any), nil
		}
		return nil, err
	}
	settings := make(map[string]any)
	if err := yaml.Unmarshal(data, &settings); err != nil {
		return nil, err
	}
	if settings == nil {
		settings = make(map[string]any)
	}
	return settings, nil
}

func applyOpenAmerSettings(settings map[string]any, primary string, models []LaunchModel) error {
	settings["provider"] = map[string]any{
		"name":     openamerProvider,
		"baseURL":  openAmerBaseURL(),
		"apiKeyEnv": openamerAPIKeyEnv,
		"model":    primary,
		"models":   openAmerModelNames(primary, models),
	}
	return nil
}

func openAmerModelConfigs(primary string, models []LaunchModel) []any {
	ordered := append([]LaunchModel(nil), models...)
	if selected, ok := findLaunchModel(ordered, primary); ok {
		ordered = append([]LaunchModel{selected}, removeLaunchModel(ordered, primary)...)
	} else {
		ordered = append([]LaunchModel{fallbackLaunchModel(primary)}, ordered...)
	}

	configs := make([]any, 0, len(ordered))
	seen := make(map[string]bool, len(ordered))
	for _, item := range ordered {
		if item.Name == "" || seen[item.Name] {
			continue
		}
		seen[item.Name] = true
		configs = append(configs, item.Name)
	}
	return configs
}

func (o *OpenAmer) CurrentModel() string {
	settingsPath, err := openAmerSettingsPath()
	if err != nil {
		return ""
	}
	settings, err := readOpenAmerYAML(settingsPath)
	if err != nil {
		return ""
	}
	provider, _ := settings["provider"].(map[string]any)
	if provider == nil {
		return ""
	}
	if name, _ := provider["name"].(string); name != openamerProvider {
		return ""
	}
	if !openAmerProviderHealthy(provider) {
		return ""
	}
	modelName, _ := provider["model"].(string)
	return modelName
}

func openAmerProviderHealthy(provider map[string]any) bool {
	baseURL, _ := provider["baseURL"].(string)
	if strings.TrimRight(baseURL, "/") != strings.TrimRight(openAmerBaseURL(), "/") {
		return false
	}
	apiKeyEnv, _ := provider["apiKeyEnv"].(string)
	return apiKeyEnv == openamerAPIKeyEnv
}

func (o *OpenAmer) Onboard() error {
	return config.MarkIntegrationOnboarded(openamerIntegrationName)
}

func (o *OpenAmer) RequiresInteractiveOnboarding() bool { return false }

func openAmerBaseURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/") + "/v1"
}

func openAmerConfigDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "launch", "openamer"), nil
}

func openAmerSettingsPath() (string, error) {
	dir, err := openAmerConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "settings.yaml"), nil
}

func writeOpenAmerFile(path string, data []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(path, data, openamerIntegrationName); err != nil {
		return err
	}
	return os.Chmod(path, 0o600)
}
