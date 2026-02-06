package config

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// OpenCode implements Runner and Editor for OpenCode integration
type OpenCode struct{}

// cloudModelLimit holds context and output token limits for a cloud model.
type cloudModelLimit struct {
	Context int
	Output  int
}

// cloudModelLimits maps cloud model base names to their token limits.
// TODO(parthsareen): grab context/output limits from model info instead of hardcoding
var cloudModelLimits = map[string]cloudModelLimit{
	"cogito-2.1:671b":     {Context: 163_840, Output: 65_536},
	"deepseek-v3.1:671b":  {Context: 163_840, Output: 163_840},
	"deepseek-v3.2":       {Context: 163_840, Output: 65_536},
	"glm-4.6":             {Context: 202_752, Output: 131_072},
	"glm-4.7":             {Context: 202_752, Output: 131_072},
	"gpt-oss:120b":        {Context: 131_072, Output: 131_072},
	"gpt-oss:20b":         {Context: 131_072, Output: 131_072},
	"kimi-k2:1t":          {Context: 262_144, Output: 262_144},
	"kimi-k2.5":           {Context: 262_144, Output: 262_144},
	"kimi-k2-thinking":    {Context: 262_144, Output: 262_144},
	"nemotron-3-nano:30b": {Context: 1_048_576, Output: 131_072},
	"qwen3-coder:480b":    {Context: 262_144, Output: 65_536},
	"qwen3-coder-next":    {Context: 262_144, Output: 32_768},
	"qwen3-next:80b":      {Context: 262_144, Output: 32_768},
}

// lookupCloudModelLimit returns the token limits for a cloud model.
// It tries the exact name first, then strips the ":cloud" suffix.
func lookupCloudModelLimit(name string) (cloudModelLimit, bool) {
	if l, ok := cloudModelLimits[name]; ok {
		return l, true
	}
	base := strings.TrimSuffix(name, ":cloud")
	if base != name {
		if l, ok := cloudModelLimits[base]; ok {
			return l, true
		}
	}
	return cloudModelLimit{}, false
}

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

	client, _ := api.ClientFromEnvironment()

	for _, model := range modelList {
		if existing, ok := models[model].(map[string]any); ok {
			// migrate existing models without _launch marker
			if isOllamaModel(existing) {
				existing["_launch"] = true
				if name, ok := existing["name"].(string); ok {
					existing["name"] = strings.TrimSuffix(name, " [Ollama]")
				}
			}
			if isCloudModel(context.Background(), client, model) {
				if l, ok := lookupCloudModelLimit(model); ok {
					existing["limit"] = map[string]any{
						"context": l.Context,
						"output":  l.Output,
					}
				}
			}
			continue
		}
		entry := map[string]any{
			"name":    model,
			"_launch": true,
		}
		if isCloudModel(context.Background(), client, model) {
			if l, ok := lookupCloudModelLimit(model); ok {
				entry["limit"] = map[string]any{
					"context": l.Context,
					"output":  l.Output,
				}
			}
		}
		models[model] = entry
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
