package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// Pi implements Runner and Editor for Pi (Pi Coding Agent) integration
type Pi struct{}

const (
	piNpmPackage      = "@mariozechner/pi-coding-agent"
	piWebSearchSource = "npm:@ollama/pi-web-search"
	piWebSearchPkg    = "@ollama/pi-web-search"
)

func (p *Pi) String() string { return "Pi" }

func (p *Pi) Run(model string, args []string) error {
	fmt.Fprintf(os.Stderr, "\n%sPreparing Pi...%s\n", ansiGray, ansiReset)
	if err := ensureNpmInstalled(); err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "%sChecking Pi installation...%s\n", ansiGray, ansiReset)
	bin, err := ensurePiInstalled()
	if err != nil {
		return err
	}

	ensurePiWebSearchPackage(bin)

	fmt.Fprintf(os.Stderr, "\n%sLaunching Pi...%s\n\n", ansiGray, ansiReset)

	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func ensureNpmInstalled() error {
	if _, err := exec.LookPath("npm"); err != nil {
		return fmt.Errorf("npm (Node.js) is required to launch pi\n\nInstall it first:\n  https://nodejs.org/\n\nThen re-run:\n  ollama launch pi")
	}
	return nil
}

func ensurePiInstalled() (string, error) {
	if _, err := exec.LookPath("pi"); err == nil {
		return "pi", nil
	}

	if _, err := exec.LookPath("npm"); err != nil {
		return "", fmt.Errorf("pi is not installed and required dependencies are missing\n\nInstall the following first:\n  npm (Node.js): https://nodejs.org/\n\nThen re-run:\n  ollama launch pi")
	}

	ok, err := ConfirmPrompt("Pi is not installed. Install with npm?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("pi installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Pi...\n")
	cmd := exec.Command("npm", "install", "-g", piNpmPackage+"@latest")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install pi: %w", err)
	}

	if _, err := exec.LookPath("pi"); err != nil {
		return "", fmt.Errorf("pi was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sPi installed successfully%s\n\n", ansiGreen, ansiReset)
	return "pi", nil
}

func ensurePiWebSearchPackage(bin string) {
	if !shouldManagePiWebSearch() {
		fmt.Fprintf(os.Stderr, "%sCloud is disabled; skipping %s setup.%s\n", ansiGray, piWebSearchPkg, ansiReset)
		return
	}

	fmt.Fprintf(os.Stderr, "%sChecking Pi web search package...%s\n", ansiGray, ansiReset)

	installed, err := piPackageInstalled(bin, piWebSearchSource)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not check %s installation: %v%s\n", ansiYellow, piWebSearchPkg, err, ansiReset)
		return
	}

	if !installed {
		fmt.Fprintf(os.Stderr, "%sInstalling %s...%s\n", ansiGray, piWebSearchPkg, ansiReset)
		cmd := exec.Command(bin, "install", piWebSearchSource)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "%s  Warning: could not install %s: %v%s\n", ansiYellow, piWebSearchPkg, err, ansiReset)
			return
		}

		fmt.Fprintf(os.Stderr, "%s  ✓ Installed %s%s\n", ansiGreen, piWebSearchPkg, ansiReset)
		return
	}

	fmt.Fprintf(os.Stderr, "%sUpdating %s...%s\n", ansiGray, piWebSearchPkg, ansiReset)
	cmd := exec.Command(bin, "update", piWebSearchSource)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not update %s: %v%s\n", ansiYellow, piWebSearchPkg, err, ansiReset)
		return
	}

	fmt.Fprintf(os.Stderr, "%s  ✓ Updated %s%s\n", ansiGreen, piWebSearchPkg, ansiReset)
}

func shouldManagePiWebSearch() bool {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return true
	}

	disabled, known := cloudStatusDisabled(context.Background(), client)
	if known && disabled {
		return false
	}
	return true
}

func piPackageInstalled(bin, source string) (bool, error) {
	cmd := exec.Command(bin, "list")
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg == "" {
			return false, err
		}
		return false, fmt.Errorf("%w: %s", err, msg)
	}

	for _, line := range strings.Split(string(out), "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, source) {
			return true, nil
		}
	}

	return false, nil
}

func (p *Pi) Paths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	var paths []string
	modelsPath := filepath.Join(home, ".pi", "agent", "models.json")
	if _, err := os.Stat(modelsPath); err == nil {
		paths = append(paths, modelsPath)
	}
	settingsPath := filepath.Join(home, ".pi", "agent", "settings.json")
	if _, err := os.Stat(settingsPath); err == nil {
		paths = append(paths, settingsPath)
	}
	return paths
}

func (p *Pi) Edit(models []string) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".pi", "agent", "models.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(data, &config)
	}

	providers, ok := config["providers"].(map[string]any)
	if !ok {
		providers = make(map[string]any)
	}

	ollama, ok := providers["ollama"].(map[string]any)
	if !ok {
		ollama = map[string]any{
			"baseUrl": envconfig.Host().String() + "/v1",
			"api":     "openai-completions",
			"apiKey":  "ollama",
		}
	}

	existingModels, ok := ollama["models"].([]any)
	if !ok {
		existingModels = make([]any, 0)
	}

	// Build set of selected models to track which need to be added
	selectedSet := make(map[string]bool, len(models))
	for _, m := range models {
		selectedSet[m] = true
	}

	// Build new models list:
	// 1. Keep user-managed models (no _launch marker) - untouched
	// 2. Keep ollama-managed models (_launch marker) that are still selected,
	//    except stale cloud entries that should be rebuilt below
	// 3. Add new ollama-managed models
	var newModels []any
	for _, m := range existingModels {
		if modelObj, ok := m.(map[string]any); ok {
			if id, ok := modelObj["id"].(string); ok {
				// User-managed model (no _launch marker) - always preserve
				if !isPiOllamaModel(modelObj) {
					newModels = append(newModels, m)
				} else if selectedSet[id] {
					// Rebuild stale managed cloud entries so createConfig refreshes
					// the whole entry instead of patching it in place.
					if !hasContextWindow(modelObj) {
						if _, ok := lookupCloudModelLimit(id); ok {
							continue
						}
					}
					newModels = append(newModels, m)
					selectedSet[id] = false
				}
			}
		}
	}

	// Add newly selected models that weren't already in the list
	client := api.NewClient(envconfig.Host(), http.DefaultClient)
	ctx := context.Background()
	for _, model := range models {
		if selectedSet[model] {
			newModels = append(newModels, createConfig(ctx, client, model))
		}
	}

	ollama["models"] = newModels
	providers["ollama"] = ollama
	config["providers"] = providers

	configData, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(configPath, configData); err != nil {
		return err
	}

	// Update settings.json with default provider and model
	settingsPath := filepath.Join(home, ".pi", "agent", "settings.json")
	settings := make(map[string]any)
	if data, err := os.ReadFile(settingsPath); err == nil {
		_ = json.Unmarshal(data, &settings)
	}

	settings["defaultProvider"] = "ollama"
	settings["defaultModel"] = models[0]

	settingsData, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return err
	}
	return fileutil.WriteWithBackup(settingsPath, settingsData)
}

func (p *Pi) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	configPath := filepath.Join(home, ".pi", "agent", "models.json")
	config, err := fileutil.ReadJSON(configPath)
	if err != nil {
		return nil
	}

	providers, _ := config["providers"].(map[string]any)
	ollama, _ := providers["ollama"].(map[string]any)
	models, _ := ollama["models"].([]any)

	var result []string
	for _, m := range models {
		if modelObj, ok := m.(map[string]any); ok {
			if id, ok := modelObj["id"].(string); ok {
				result = append(result, id)
			}
		}
	}
	slices.Sort(result)
	return result
}

// isPiOllamaModel reports whether a model config entry is managed by ollama launch
func isPiOllamaModel(cfg map[string]any) bool {
	if v, ok := cfg["_launch"].(bool); ok && v {
		return true
	}
	return false
}

func hasContextWindow(cfg map[string]any) bool {
	switch v := cfg["contextWindow"].(type) {
	case float64:
		return v > 0
	case int:
		return v > 0
	case int64:
		return v > 0
	default:
		return false
	}
}

// createConfig builds Pi model config with capability detection
func createConfig(ctx context.Context, client *api.Client, modelID string) map[string]any {
	cfg := map[string]any{
		"id":      modelID,
		"_launch": true,
	}
	if l, ok := lookupCloudModelLimit(modelID); ok {
		cfg["contextWindow"] = l.Context
	}

	applyCloudContextFallback := func() {
		if l, ok := lookupCloudModelLimit(modelID); ok {
			cfg["contextWindow"] = l.Context
		}
	}

	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelID})
	if err != nil {
		applyCloudContextFallback()
		return cfg
	}

	// Set input types based on vision capability
	if slices.Contains(resp.Capabilities, model.CapabilityVision) {
		cfg["input"] = []string{"text", "image"}
	} else {
		cfg["input"] = []string{"text"}
	}

	// Set reasoning based on thinking capability
	if slices.Contains(resp.Capabilities, model.CapabilityThinking) {
		cfg["reasoning"] = true
	}

	// Extract context window from ModelInfo. For known cloud models, the
	// pre-filled shared limit remains unless the server provides a positive value.
	hasContextWindow := false
	for key, val := range resp.ModelInfo {
		if strings.HasSuffix(key, ".context_length") {
			if ctxLen, ok := val.(float64); ok && ctxLen > 0 {
				cfg["contextWindow"] = int(ctxLen)
				hasContextWindow = true
			}
			break
		}
	}
	if !hasContextWindow {
		applyCloudContextFallback()
	}

	return cfg
}
