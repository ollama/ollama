package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
	"gopkg.in/yaml.v3"
)

const (
	ompIntegrationName = "omp"
	ompProviderName    = "ollama"
	ompSetupVersion    = 1
	ompWebSearchPlugin = "@ollama/pi-web-search"
)

// OMP implements Runner for the OMP coding-agent integration.
type OMP struct{}

func (o *OMP) String() string { return "OMP" }

func (o *OMP) Paths() []string {
	var paths []string
	for _, pathFn := range []func() (string, error){ompModelsPath, ompConfigPath} {
		path, err := pathFn()
		if err != nil {
			continue
		}
		if _, err := os.Stat(path); err == nil {
			paths = append(paths, path)
		}
	}
	return paths
}

func (o *OMP) Configure(model string) error {
	return o.ConfigureWithModels(model, []LaunchModel{fallbackLaunchModel(model)})
}

func (o *OMP) ConfigureWithModels(primary string, models []LaunchModel) error {
	if primary == "" {
		return nil
	}
	if len(models) == 0 {
		models = []LaunchModel{fallbackLaunchModel(primary)}
	}
	if err := writeOMPModelsConfig(primary, models); err != nil {
		return err
	}
	return writeOMPAgentConfig()
}

func (o *OMP) CurrentModel() string {
	cfg, err := readOMPModelsConfig()
	if err != nil {
		return ""
	}
	provider, ok := ompProvider(cfg)
	if !ok {
		return ""
	}
	if !ompProviderHealthy(provider) {
		return ""
	}
	models, _ := provider["models"].([]any)
	for _, raw := range models {
		entry, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if id, _ := entry["id"].(string); id != "" {
			return id
		}
	}
	return ""
}

func (o *OMP) Onboard() error {
	return config.MarkIntegrationOnboarded(ompIntegrationName)
}

func (o *OMP) RequiresInteractiveOnboarding() bool { return false }

func (o *OMP) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", ompModelName(model))
	}
	args = append(args, extra...)
	return args
}

func ompModelName(model string) string {
	if strings.HasPrefix(model, "ollama/") {
		return model
	}
	return "ollama/" + model
}

func (o *OMP) findPath() (string, error) {
	if p, err := exec.LookPath("omp"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	for _, dir := range []string{
		filepath.Join(home, ".local", "bin"),
		filepath.Join(home, ".bun", "bin"),
	} {
		for _, name := range ompExecutableNames() {
			fallback := filepath.Join(dir, name)
			if _, err := os.Stat(fallback); err == nil {
				return fallback, nil
			}
		}
	}
	return "", exec.ErrNotFound
}

func ompExecutableNames() []string {
	if runtime.GOOS == "windows" {
		return []string{"omp.exe", "omp.cmd", "omp.bat"}
	}
	return []string{"omp"}
}

func (o *OMP) Run(model string, _ []LaunchModel, args []string) error {
	ompPath, err := o.findPath()
	if err != nil {
		return fmt.Errorf("omp is not installed, install from https://omp.sh")
	}

	ensureOMPWebSearchPlugin(ompPath)

	cmd := exec.Command(ompPath, o.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	return cmd.Run()
}

func ensureOMPWebSearchPlugin(bin string) {
	if !shouldManageOllamaWebSearch() {
		fmt.Fprintf(os.Stderr, "%sCloud is disabled; skipping %s setup.%s\n", ansiGray, ompWebSearchPlugin, ansiReset)
		return
	}

	fmt.Fprintf(os.Stderr, "%sChecking OMP web search plugin...%s\n", ansiGray, ansiReset)

	installed, err := ompPluginInstalled(bin, ompWebSearchPlugin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not check %s installation: %v%s\n", ansiYellow, ompWebSearchPlugin, err, ansiReset)
		return
	}

	verb := "Installing"
	warnVerb := "install"
	doneVerb := "Installed"
	if installed {
		verb = "Updating"
		warnVerb = "update"
		doneVerb = "Updated"
	}

	fmt.Fprintf(os.Stderr, "%s%s %s...%s\n", ansiGray, verb, ompWebSearchPlugin, ansiReset)
	cmd := exec.Command(bin, "plugin", "install", ompWebSearchPlugin)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not %s %s: %v%s\n", ansiYellow, warnVerb, ompWebSearchPlugin, err, ansiReset)
		return
	}

	fmt.Fprintf(os.Stderr, "%s  ✓ %s %s%s\n", ansiGreen, doneVerb, ompWebSearchPlugin, ansiReset)
}

func ompPluginInstalled(bin, plugin string) (bool, error) {
	cmd := exec.Command(bin, "plugin", "list")
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg == "" {
			return false, err
		}
		return false, fmt.Errorf("%w: %s", err, msg)
	}

	versioned := plugin + "@"
	for _, line := range strings.Split(string(out), "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.Contains(trimmed, versioned) || trimmed == plugin {
			return true, nil
		}
	}
	return false, nil
}

func ompModelsPath() (string, error) {
	dir, err := ompAgentDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "models.yml"), nil
}

func ompConfigPath() (string, error) {
	dir, err := ompAgentDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "config.yml"), nil
}

func ompAgentDir() (string, error) {
	if dir := strings.TrimSpace(os.Getenv("PI_CODING_AGENT_DIR")); dir != "" {
		return dir, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	configDir := strings.TrimSpace(os.Getenv("PI_CONFIG_DIR"))
	if configDir == "" {
		configDir = ".omp"
	}
	if filepath.IsAbs(configDir) {
		return filepath.Join(configDir, "agent"), nil
	}
	return filepath.Join(home, configDir, "agent"), nil
}

func readOMPModelsConfig() (map[string]any, error) {
	path, err := ompModelsPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg == nil {
		cfg = make(map[string]any)
	}
	return cfg, nil
}

func writeOMPModelsConfig(primary string, models []LaunchModel) error {
	path, err := ompModelsPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	cfg := make(map[string]any)
	if existing, err := readOMPModelsConfig(); err == nil {
		cfg = existing
	}

	provider := ensureOMPProvider(cfg)
	existingByID := ompModelEntriesByID(provider)
	ordered := append([]LaunchModel(nil), models...)
	if model, ok := findLaunchModel(ordered, primary); ok {
		ordered = append([]LaunchModel{model}, removeLaunchModel(ordered, primary)...)
	} else {
		ordered = append([]LaunchModel{fallbackLaunchModel(primary)}, ordered...)
	}

	var merged []any
	seen := make(map[string]bool, len(ordered))
	for _, model := range ordered {
		if model.Name == "" || seen[model.Name] {
			continue
		}
		seen[model.Name] = true
		entry := ompModelConfig(model)
		if existing, ok := existingByID[model.Name]; ok {
			for key, value := range existing {
				if _, overridden := entry[key]; !overridden {
					entry[key] = value
				}
			}
		}
		merged = append(merged, entry)
	}

	for _, raw := range ompProviderModels(provider) {
		entry, ok := raw.(map[string]any)
		if !ok {
			merged = append(merged, raw)
			continue
		}
		id, _ := entry["id"].(string)
		if id == "" || seen[id] {
			continue
		}
		merged = append(merged, entry)
	}
	provider["models"] = merged

	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, data, ompIntegrationName)
}

func writeOMPAgentConfig() error {
	path, err := ompConfigPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	cfg := make(map[string]any)
	if data, err := os.ReadFile(path); err == nil {
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return err
		}
		if cfg == nil {
			cfg = make(map[string]any)
		}
	}
	cfg["setupVersion"] = ompSetupVersion

	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, data, ompIntegrationName)
}

func ensureOMPProvider(cfg map[string]any) map[string]any {
	providers, _ := cfg["providers"].(map[string]any)
	if providers == nil {
		providers = make(map[string]any)
		cfg["providers"] = providers
	}
	provider, _ := providers[ompProviderName].(map[string]any)
	if provider == nil {
		provider = make(map[string]any)
		providers[ompProviderName] = provider
	}

	provider["baseUrl"] = ompBaseURL()
	provider["api"] = "openai-responses"
	provider["auth"] = "none"
	provider["discovery"] = map[string]any{"type": "ollama"}
	return provider
}

func ompBaseURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/") + "/v1"
}

func ompProviderHealthy(provider map[string]any) bool {
	baseURL, _ := provider["baseUrl"].(string)
	if strings.TrimRight(baseURL, "/") != strings.TrimRight(ompBaseURL(), "/") {
		return false
	}
	api, _ := provider["api"].(string)
	if api != "openai-responses" {
		return false
	}
	auth, _ := provider["auth"].(string)
	if auth != "none" {
		return false
	}
	discovery, _ := provider["discovery"].(map[string]any)
	if discovery == nil {
		return false
	}
	discoveryType, _ := discovery["type"].(string)
	return discoveryType == "ollama"
}

func ompProvider(cfg map[string]any) (map[string]any, bool) {
	providers, ok := cfg["providers"].(map[string]any)
	if !ok {
		return nil, false
	}
	provider, ok := providers[ompProviderName].(map[string]any)
	return provider, ok
}

func ompProviderModels(provider map[string]any) []any {
	models, _ := provider["models"].([]any)
	return models
}

func ompModelEntriesByID(provider map[string]any) map[string]map[string]any {
	out := make(map[string]map[string]any)
	for _, raw := range ompProviderModels(provider) {
		entry, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if id, _ := entry["id"].(string); id != "" {
			out[id] = entry
		}
	}
	return out
}

func ompModelConfig(modelInfo LaunchModel) map[string]any {
	entry := map[string]any{
		"id":   modelInfo.Name,
		"name": modelInfo.Name,
	}
	input := []string{"text"}
	if slices.Contains(modelInfo.Capabilities, model.CapabilityVision) {
		input = append(input, "image")
	}
	entry["input"] = input

	if modelInfo.ContextLength > 0 {
		entry["contextWindow"] = modelInfo.ContextLength
	}
	if modelInfo.MaxOutputTokens > 0 {
		entry["maxTokens"] = modelInfo.MaxOutputTokens
	}
	return entry
}

func removeLaunchModel(models []LaunchModel, name string) []LaunchModel {
	out := make([]LaunchModel, 0, len(models))
	for _, model := range models {
		if launchModelMatches(model.Name, name) || launchModelMatches(name, model.Name) {
			continue
		}
		out = append(out, model)
	}
	return out
}
