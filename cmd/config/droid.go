package config

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
)

// Droid implements Runner and Editor for Droid integration
type Droid struct{}

// droidModelEntry represents a custom model entry in Droid's settings.json
type droidModelEntry struct {
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

func (d *Droid) Run(model string) error {
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

	cmd := exec.Command("droid")
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

	settings := make(map[string]any)
	if data, err := os.ReadFile(settingsPath); err == nil {
		if err := json.Unmarshal(data, &settings); err != nil {
			return fmt.Errorf("failed to parse settings file: %w, at: %s", err, settingsPath)
		}
	}

	customModels, _ := settings["customModels"].([]any)

	// Keep only non-Ollama models (we'll rebuild Ollama models fresh)
	nonOllamaModels := slices.DeleteFunc(slices.Clone(customModels), isOllamaModelEntry)

	// Build new Ollama model entries with sequential indices (0, 1, 2, ...)
	var ollamaModels []any
	var defaultModelID string
	for i, model := range models {
		modelID := fmt.Sprintf("custom:%s-[Ollama]-%d", model, i)
		ollamaModels = append(ollamaModels, droidModelEntry{
			Model:           model,
			DisplayName:     model,
			BaseURL:         "http://localhost:11434/v1",
			APIKey:          "ollama",
			Provider:        "generic-chat-completion-api",
			MaxOutputTokens: 64000,
			SupportsImages:  false,
			ID:              modelID,
			Index:           i,
		})
		if i == 0 {
			defaultModelID = modelID
		}
	}

	settings["customModels"] = append(ollamaModels, nonOllamaModels...)

	sessionSettings, ok := settings["sessionDefaultSettings"].(map[string]any)
	if !ok {
		sessionSettings = make(map[string]any)
	}
	sessionSettings["model"] = defaultModelID

	if effort, ok := sessionSettings["reasoningEffort"].(string); !ok || !isValidReasoningEffort(effort) {
		sessionSettings["reasoningEffort"] = "none"
	}

	settings["sessionDefaultSettings"] = sessionSettings

	data, err := json.MarshalIndent(settings, "", "  ")
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
	settings, err := readJSONFile(filepath.Join(home, ".factory", "settings.json"))
	if err != nil {
		return nil
	}

	customModels, _ := settings["customModels"].([]any)

	var result []string
	for _, m := range customModels {
		if !isOllamaModelEntry(m) {
			continue
		}
		entry, ok := m.(map[string]any)
		if !ok {
			continue
		}
		if model, _ := entry["model"].(string); model != "" {
			result = append(result, model)
		}
	}
	return result
}

var validReasoningEfforts = []string{"high", "medium", "low", "none"}

func isValidReasoningEffort(effort string) bool {
	return slices.Contains(validReasoningEfforts, effort)
}

func isOllamaModelEntry(m any) bool {
	entry, ok := m.(map[string]any)
	if !ok {
		return false
	}
	id, _ := entry["id"].(string)
	return strings.Contains(id, "-[Ollama]-")
}
