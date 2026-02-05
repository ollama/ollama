package config

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// OpenCode implements Runner and Editor for OpenCode integration
type OpenCode struct{}

func (o *OpenCode) String() string { return "OpenCode" }

func (o *OpenCode) Run(model string, args []string) error {
	if _, err := exec.LookPath("opencode"); err != nil {
		return fmt.Errorf("opencode is not installed, install from https://opencode.ai")
	}

	// Call Edit() to ensure config is up-to-date before launch
	models := []string{model}
	if config, err := loadIntegration("opencode"); err == nil && len(config.Models) > 0 {
		models = config.Models
	}
	if err := o.Edit(models); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	cmd := exec.Command("opencode", args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (o *OpenCode) Paths() []string {
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

func (o *OpenCode) Edit(modelList []string) error {
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
		_ = json.Unmarshal(data, &config) // Ignore parse errors; treat missing/corrupt files as empty
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
				"baseURL": envconfig.Host().String() + "/v1",
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
			if isOllamaModel(cfgMap) && !selectedSet[name] {
				delete(models, name)
			}
		}
	}

	for _, model := range modelList {
		if existing, ok := models[model].(map[string]any); ok {
			// migrate existing models without _launch marker
			if isOllamaModel(existing) {
				existing["_launch"] = true
				if name, ok := existing["name"].(string); ok {
					existing["name"] = strings.TrimSuffix(name, " [Ollama]")
				}
			}
			continue
		}
		models[model] = map[string]any{
			"name":    model,
			"_launch": true,
		}
	}

	ollama["models"] = models
	provider["ollama"] = ollama
	config["provider"] = provider

	configData, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := writeWithBackup(configPath, configData); err != nil {
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
		_ = json.Unmarshal(data, &state) // Ignore parse errors; use defaults
	}

	recent, _ := state["recent"].([]any)

	modelSet := make(map[string]bool)
	for _, m := range modelList {
		modelSet[m] = true
	}

	// Filter out existing Ollama models we're about to re-add
	newRecent := slices.DeleteFunc(slices.Clone(recent), func(entry any) bool {
		e, ok := entry.(map[string]any)
		if !ok || e["providerID"] != "ollama" {
			return false
		}
		modelID, _ := e["modelID"].(string)
		return modelSet[modelID]
	})

	// Prepend models in reverse order so first model ends up first
	for _, model := range slices.Backward(modelList) {
		newRecent = slices.Insert(newRecent, 0, any(map[string]any{
			"providerID": "ollama",
			"modelID":    model,
		}))
	}

	const maxRecentModels = 10
	newRecent = newRecent[:min(len(newRecent), maxRecentModels)]

	state["recent"] = newRecent

	stateData, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return writeWithBackup(statePath, stateData)
}

func (o *OpenCode) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	config, err := readJSONFile(filepath.Join(home, ".config", "opencode", "opencode.json"))
	if err != nil {
		return nil
	}
	provider, _ := config["provider"].(map[string]any)
	ollama, _ := provider["ollama"].(map[string]any)
	models, _ := ollama["models"].(map[string]any)
	if len(models) == 0 {
		return nil
	}
	keys := slices.Collect(maps.Keys(models))
	slices.Sort(keys)
	return keys
}

// isOllamaModel reports whether a model config entry is managed by us
func isOllamaModel(cfg map[string]any) bool {
	if v, ok := cfg["_launch"].(bool); ok && v {
		return true
	}
	// previously used [Ollama] as a suffix for the model managed by ollama launch
	if name, ok := cfg["name"].(string); ok {
		return strings.HasSuffix(name, "[Ollama]")
	}
	return false
}
