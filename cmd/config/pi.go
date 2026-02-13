package config

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
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// Pi implements Runner and Editor for Pi (Pi Coding Agent) integration
type Pi struct{}

func (p *Pi) String() string { return "Pi" }

func (p *Pi) Run(model string, args []string) error {
	if _, err := exec.LookPath("pi"); err != nil {
		return fmt.Errorf("pi is not installed, install with: npm install -g @mariozechner/pi-coding-agent")
	}

	// Call Edit() to ensure config is up-to-date before launch
	models := []string{model}
	if config, err := loadIntegration("pi"); err == nil && len(config.Models) > 0 {
		models = config.Models
	}
	if err := p.Edit(models); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	cmd := exec.Command("pi", args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
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
	// 2. Keep ollama-managed models (_launch marker) that are still selected
	// 3. Add new ollama-managed models
	var newModels []any
	for _, m := range existingModels {
		if modelObj, ok := m.(map[string]any); ok {
			if id, ok := modelObj["id"].(string); ok {
				// User-managed model (no _launch marker) - always preserve
				if !isPiOllamaModel(modelObj) {
					newModels = append(newModels, m)
				} else if selectedSet[id] {
					// Ollama-managed and still selected - keep it
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
	if err := writeWithBackup(configPath, configData); err != nil {
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
	return writeWithBackup(settingsPath, settingsData)
}

func (p *Pi) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	configPath := filepath.Join(home, ".pi", "agent", "models.json")
	config, err := readJSONFile(configPath)
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

// createConfig builds Pi model config with capability detection
func createConfig(ctx context.Context, client *api.Client, modelID string) map[string]any {
	cfg := map[string]any{
		"id":      modelID,
		"_launch": true,
	}

	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelID})
	if err != nil {
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

	// Extract context window from ModelInfo
	for key, val := range resp.ModelInfo {
		if strings.HasSuffix(key, ".context_length") {
			if ctxLen, ok := val.(float64); ok && ctxLen > 0 {
				cfg["contextWindow"] = int(ctxLen)
			}
			break
		}
	}

	return cfg
}
