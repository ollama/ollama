package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

var droidIntegration = &integration{
	Name:             "Droid",
	DisplayName:      "Droid",
	Command:          "droid",
	EnvVars:          func(model string) []envVar { return nil },
	Args:             func(model string) []string { return nil },
	Setup:            setupDroidSettings,
	CheckInstall:     checkCommand("droid", "install from https://docs.factory.ai/cli/getting-started/quickstart"),
	ConfigPaths:      droidConfigPaths,
	ConfiguredModels: droidModels,
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
	displayName, _ := entry["displayName"].(string)
	return strings.HasSuffix(displayName, "[Ollama]")
}

func setupDroidSettings(models []string) error {
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
		_ = json.Unmarshal(data, &settings) // ignore parse errors; treat as empty
	}

	customModels, _ := settings["customModels"].([]any)

	// Keep only non-Ollama models (we'll rebuild Ollama models fresh)
	nonOllamaModels := slices.DeleteFunc(slices.Clone(customModels), isOllamaModelEntry)

	// Build new Ollama model entries with sequential indices (0, 1, 2, ...)
	var ollamaModels []any
	var defaultModelID string
	for i, model := range models {
		modelID := fmt.Sprintf("custom:%s-[Ollama]-%d", model, i)
		newEntry := map[string]any{
			"model":           model,
			"displayName":     fmt.Sprintf("%s [Ollama]", model),
			"baseUrl":         "http://localhost:11434/v1",
			"apiKey":          "ollama",
			"provider":        "generic-chat-completion-api",
			"maxOutputTokens": modelContextLength(model),
			"supportsImages":  modelSupportsImages(model),
			"id":              modelID,
			"index":           i,
		}
		ollamaModels = append(ollamaModels, newEntry)

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
	return atomicWrite(settingsPath, data)
}

func droidSettings() (map[string]any, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(home, ".factory", "settings.json"))
	if err != nil {
		return nil, err
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func droidModels() []string {
	settings, err := droidSettings()
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

func droidConfigPaths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	var paths []string
	p := filepath.Join(home, ".factory", "settings.json")
	if _, err := os.Stat(p); err == nil {
		paths = append(paths, p)
	}
	return paths
}
