package config

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// Droid implements Runner and Editor for Droid integration
type Droid struct{}

// droidSettings represents the Droid settings.json file (only fields we use)
type droidSettings struct {
	CustomModels           []modelEntry    `json:"customModels"`
	SessionDefaultSettings sessionSettings `json:"sessionDefaultSettings"`
}

type sessionSettings struct {
	Model           string `json:"model"`
	ReasoningEffort string `json:"reasoningEffort"`
}

type modelEntry struct {
	Model           string `json:"model"`
	DisplayName     string `json:"displayName"`
	BaseURL         string `json:"baseUrl"`
	APIKey          string `json:"apiKey"`
	Provider        string `json:"provider"`
	MaxOutputTokens int    `json:"maxOutputTokens"`
	SupportsImages  bool   `json:"supportsImages"`
	ID              string `json:"id"`
	Index           int    `json:"index"`
}

func (d *Droid) String() string { return "Droid" }

func (d *Droid) Run(model string, args []string) error {
	if _, err := exec.LookPath("droid"); err != nil {
		return fmt.Errorf("droid is not installed, install from https://docs.factory.ai/cli/getting-started/quickstart")
	}

	// Call Edit() to ensure config is up-to-date before launch
	models := []string{model}
	if config, err := loadIntegration("droid"); err == nil && len(config.Models) > 0 {
		models = config.Models
	}
	if err := d.Edit(models); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	cmd := exec.Command("droid", args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (d *Droid) Paths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	p := filepath.Join(home, ".factory", "settings.json")
	if _, err := os.Stat(p); err == nil {
		return []string{p}
	}
	return nil
}

func (d *Droid) Edit(models []string) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	settingsPath := filepath.Join(home, ".factory", "settings.json")
	if err := os.MkdirAll(filepath.Dir(settingsPath), 0o755); err != nil {
		return err
	}

	// Read file once, unmarshal twice:
	// map preserves unknown fields for writing back (including extra fields in model entries)
	settingsMap := make(map[string]any)
	var settings droidSettings
	if data, err := os.ReadFile(settingsPath); err == nil {
		if err := json.Unmarshal(data, &settingsMap); err != nil {
			return fmt.Errorf("failed to parse settings file: %w, at: %s", err, settingsPath)
		}
		json.Unmarshal(data, &settings) // ignore error, zero values are fine
	}

	// Keep only non-Ollama models from the raw map (preserves extra fields)
	// Rebuild Ollama models
	var nonOllamaModels []any
	if rawModels, ok := settingsMap["customModels"].([]any); ok {
		for _, raw := range rawModels {
			if m, ok := raw.(map[string]any); ok {
				if m["apiKey"] != "ollama" {
					nonOllamaModels = append(nonOllamaModels, raw)
				}
			}
		}
	}

	// Build new Ollama model entries with sequential indices (0, 1, 2, ...)
	client, _ := api.ClientFromEnvironment()

	var newModels []any
	var defaultModelID string
	for i, model := range models {
		maxOutput := 64000
		if isCloudModel(context.Background(), client, model) {
			if l, ok := lookupCloudModelLimit(model); ok {
				maxOutput = l.Output
			}
		}
		modelID := fmt.Sprintf("custom:%s-%d", model, i)
		newModels = append(newModels, modelEntry{
			Model:           model,
			DisplayName:     model,
			BaseURL:         envconfig.Host().String() + "/v1",
			APIKey:          "ollama",
			Provider:        "generic-chat-completion-api",
			MaxOutputTokens: maxOutput,
			SupportsImages:  false,
			ID:              modelID,
			Index:           i,
		})
		if i == 0 {
			defaultModelID = modelID
		}
	}

	settingsMap["customModels"] = append(newModels, nonOllamaModels...)

	// Update session default settings (preserve unknown fields in the nested object)
	sessionSettings, ok := settingsMap["sessionDefaultSettings"].(map[string]any)
	if !ok {
		sessionSettings = make(map[string]any)
	}
	sessionSettings["model"] = defaultModelID

	if !isValidReasoningEffort(settings.SessionDefaultSettings.ReasoningEffort) {
		sessionSettings["reasoningEffort"] = "none"
	}

	settingsMap["sessionDefaultSettings"] = sessionSettings

	data, err := json.MarshalIndent(settingsMap, "", "  ")
	if err != nil {
		return err
	}
	return writeWithBackup(settingsPath, data)
}

func (d *Droid) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	data, err := os.ReadFile(filepath.Join(home, ".factory", "settings.json"))
	if err != nil {
		return nil
	}

	var settings droidSettings
	if err := json.Unmarshal(data, &settings); err != nil {
		return nil
	}

	var result []string
	for _, m := range settings.CustomModels {
		if m.APIKey == "ollama" {
			result = append(result, m.Model)
		}
	}
	return result
}

var validReasoningEfforts = []string{"high", "medium", "low", "none"}

func isValidReasoningEffort(effort string) bool {
	return slices.Contains(validReasoningEfforts, effort)
}
