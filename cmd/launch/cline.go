package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const clineLaunchProvider = "ollama"

// Cline implements Runner and Editor for the Cline CLI integration
type Cline struct{}

func (c *Cline) String() string { return "Cline" }

func (c *Cline) Run(model string, _ []LaunchModel, args []string) error {
	if _, err := exec.LookPath("cline"); err != nil {
		return fmt.Errorf("cline is not installed, install with: npm install -g cline")
	}

	launchArgs := clineLaunchArgs(model, args)
	cmd := exec.Command("cline", launchArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func clineLaunchArgs(model string, extra []string) []string {
	return extra
}

func (c *Cline) Paths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	var paths []string
	for _, p := range []string{
		clineProvidersPath(home),
		clineLegacyGlobalStatePath(home),
	} {
		if _, err := os.Stat(p); err == nil {
			paths = append(paths, p)
		}
	}
	return paths
}

func (c *Cline) Edit(models []LaunchModel) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	providersPath := clineProvidersPath(home)
	legacyPath := clineLegacyGlobalStatePath(home)

	providersConfig, err := readClineConfig(providersPath)
	if err != nil {
		return err
	}
	legacyConfig, err := readClineConfig(legacyPath)
	if err != nil {
		return err
	}

	if err := writeClineProvidersConfig(providersPath, providersConfig, models[0].Name); err != nil {
		return err
	}
	return writeClineLegacyGlobalState(legacyPath, legacyConfig, models[0].Name)
}

func clineProvidersPath(home string) string {
	return filepath.Join(home, ".cline", "data", "settings", "providers.json")
}

func clineLegacyGlobalStatePath(home string) string {
	return filepath.Join(home, ".cline", "data", "globalState.json")
}

func clineOllamaRootURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/")
}

func clineProviderBaseURL() string {
	return clineOllamaRootURL() + "/v1"
}

func readClineConfig(configPath string) (map[string]any, error) {
	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("failed to parse config: %w, at: %s", err, configPath)
		}
	} else if !os.IsNotExist(err) {
		return nil, err
	}
	return config, nil
}

func writeClineProvidersConfig(configPath string, config map[string]any, model string) error {
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	providers, _ := config["providers"].(map[string]any)
	if providers == nil {
		providers = make(map[string]any)
	}

	provider, _ := providers[clineLaunchProvider].(map[string]any)
	if provider == nil {
		provider = make(map[string]any)
	}
	settings, _ := provider["settings"].(map[string]any)
	if settings == nil {
		settings = make(map[string]any)
	}

	baseURL := clineProviderBaseURL()
	previousModel, _ := settings["model"].(string)
	previousBaseURL, _ := settings["baseUrl"].(string)
	previousTokenSource, _ := provider["tokenSource"].(string)

	settings["provider"] = clineLaunchProvider
	settings["model"] = model
	settings["baseUrl"] = baseURL
	delete(settings, "apiKey")
	provider["settings"] = settings

	if previousModel != model || previousBaseURL != baseURL || previousTokenSource != "manual" {
		provider["updatedAt"] = time.Now().UTC().Format(time.RFC3339Nano)
	} else if _, ok := provider["updatedAt"].(string); !ok {
		provider["updatedAt"] = time.Now().UTC().Format(time.RFC3339Nano)
	}
	provider["tokenSource"] = "manual"
	providers[clineLaunchProvider] = provider

	config["version"] = float64(1)
	config["lastUsedProvider"] = clineLaunchProvider
	config["providers"] = providers

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(configPath, data, "cline")
}

func writeClineLegacyGlobalState(configPath string, config map[string]any, model string) error {
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	baseURL := clineOllamaRootURL()
	config["ollamaBaseUrl"] = baseURL
	config["actModeApiProvider"] = clineLaunchProvider
	config["actModeOllamaModelId"] = model
	config["actModeOllamaBaseUrl"] = baseURL
	config["planModeApiProvider"] = clineLaunchProvider
	config["planModeOllamaModelId"] = model
	config["planModeOllamaBaseUrl"] = baseURL

	config["welcomeViewCompleted"] = true

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(configPath, data, "cline")
}

func (c *Cline) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	if model := clineProviderModel(home); model != "" {
		return []string{model}
	}

	config, err := fileutil.ReadJSON(clineLegacyGlobalStatePath(home))
	if err != nil {
		return nil
	}

	switch config["actModeApiProvider"] {
	case "ollama":
	default:
		return nil
	}

	modelID, _ := config["actModeOllamaModelId"].(string)
	if modelID == "" {
		return nil
	}
	return []string{modelID}
}

func clineProviderModel(home string) string {
	config, err := fileutil.ReadJSON(clineProvidersPath(home))
	if err != nil {
		return ""
	}
	if config["lastUsedProvider"] != clineLaunchProvider {
		return ""
	}
	providers, _ := config["providers"].(map[string]any)
	provider, _ := providers[clineLaunchProvider].(map[string]any)
	settings, _ := provider["settings"].(map[string]any)
	model, _ := settings["model"].(string)
	return model
}
