package integrations

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var openCodeIntegration = &integrationDef{
	Name:         "OpenCode",
	DisplayName:  "OpenCode",
	Command:      "opencode",
	EnvVars:      func(model string) []envVar { return nil },
	Args:         func(model string) []string { return nil },
	Setup:        setupOpenCodeSettings,
	CheckInstall: checkCommand("opencode", "Install from: https://opencode.ai"),
}

func setupOpenCodeSettings(modelList []string) error {
	if len(modelList) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".config", "opencode", "opencode.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		json.Unmarshal(data, &config)
	}

	config["$schema"] = "https://opencode.ai/config.json"

	provider, ok := config["provider"].(map[string]any)
	if !ok {
		provider = make(map[string]any)
	}

	ollama, ok := provider["ollama"].(map[string]any)
	if !ok {
		ollama = map[string]any{
			"npm":  "@ai-sdk/openai-compatible",
			"name": "Ollama (local)",
			"options": map[string]any{
				"baseURL": "http://localhost:11434/v1",
			},
		}
	}

	models, ok := ollama["models"].(map[string]any)
	if !ok {
		models = make(map[string]any)
	}

	selectedSet := make(map[string]bool)
	for _, m := range modelList {
		selectedSet[m] = true
	}

	for name, cfg := range models {
		if cfgMap, ok := cfg.(map[string]any); ok {
			if displayName, ok := cfgMap["name"].(string); ok {
				if strings.HasSuffix(displayName, "[Ollama]") && !selectedSet[name] {
					delete(models, name)
				}
			}
		}
	}

	for _, model := range modelList {
		models[model] = map[string]any{
			"name": fmt.Sprintf("%s [Ollama]", model),
		}
	}

	ollama["models"] = models
	provider["ollama"] = ollama
	config["provider"] = provider

	if err := atomicWriteJSON(configPath, config); err != nil {
		return err
	}

	statePath := filepath.Join(home, ".local", "state", "opencode", "model.json")
	if err := os.MkdirAll(filepath.Dir(statePath), 0o755); err != nil {
		return err
	}

	state := map[string]any{
		"recent":   []any{},
		"favorite": []any{},
		"variant":  map[string]any{},
	}
	if data, err := os.ReadFile(statePath); err == nil {
		json.Unmarshal(data, &state)
	}

	recent, _ := state["recent"].([]any)

	modelSet := make(map[string]bool)
	for _, m := range modelList {
		modelSet[m] = true
	}

	newRecent := []any{}
	for _, entry := range recent {
		if e, ok := entry.(map[string]any); ok {
			if e["providerID"] == "ollama" {
				if modelID, ok := e["modelID"].(string); ok && modelSet[modelID] {
					continue
				}
			}
		}
		newRecent = append(newRecent, entry)
	}

	for i := len(modelList) - 1; i >= 0; i-- {
		newRecent = append([]any{map[string]any{
			"providerID": "ollama",
			"modelID":    modelList[i],
		}}, newRecent...)
	}

	if len(newRecent) > 10 {
		newRecent = newRecent[:10]
	}

	state["recent"] = newRecent

	return atomicWriteJSON(statePath, state)
}

func getOpenCodeOllamaModels() (map[string]any, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	config, err := readJSONFile(filepath.Join(home, ".config", "opencode", "opencode.json"))
	if err != nil {
		return nil, err
	}

	provider, _ := config["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	models, _ := ollama["models"].(map[string]any)
	return models, nil
}

func getOpenCodeConfiguredModels() []string {
	models, err := getOpenCodeOllamaModels()
	if err != nil || models == nil {
		return nil
	}

	var result []string
	for name := range models {
		result = append(result, name)
	}
	return result
}

func getOpenCodeExistingConfigPaths() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	var paths []string
	p := filepath.Join(home, ".config", "opencode", "opencode.json")
	if _, err := os.Stat(p); err == nil {
		paths = append(paths, p)
	}
	sp := filepath.Join(home, ".local", "state", "opencode", "model.json")
	if _, err := os.Stat(sp); err == nil {
		paths = append(paths, sp)
	}
	return paths
}
