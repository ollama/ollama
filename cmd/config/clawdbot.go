package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

type Clawdbot struct{}

func (c *Clawdbot) String() string { return "Clawdbot" }

const ansiGreen = "\033[32m"

func (c *Clawdbot) Run(model string) error {
	if _, err := exec.LookPath("clawdbot"); err != nil {
		return fmt.Errorf("clawdbot is not installed, install from https://docs.clawd.bot")
	}

	models := []string{model}
	if config, err := loadIntegration("clawdbot"); err == nil && len(config.Models) > 0 {
		models = config.Models
	}
	if err := c.Edit(models); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}

	cmd := exec.Command("clawdbot", "gateway")
	cmd.Stdin = os.Stdin

	// Capture output to detect "already running" message
	var outputBuf bytes.Buffer
	cmd.Stdout = io.MultiWriter(os.Stdout, &outputBuf)
	cmd.Stderr = io.MultiWriter(os.Stderr, &outputBuf)

	err := cmd.Run()
	if err != nil && strings.Contains(outputBuf.String(), "Gateway already running") {
		fmt.Fprintf(os.Stderr, "%sClawdbot has been configured with Ollama. Gateway is already running.%s\n", ansiGreen, ansiReset)
		return nil
	}
	return err
}

func (c *Clawdbot) Paths() []string {
	home, _ := os.UserHomeDir()
	p := filepath.Join(home, ".clawdbot", "clawdbot.json")
	if _, err := os.Stat(p); err == nil {
		return []string{p}
	}
	return nil
}

func (c *Clawdbot) Edit(models []string) error {
	if len(models) == 0 {
		return nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configPath := filepath.Join(home, ".clawdbot", "clawdbot.json")
	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}

	// Read into map[string]any to preserve unknown fields
	config := make(map[string]any)
	if data, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(data, &config)
	}

	// Navigate/create: models.providers.ollama (preserving other providers)
	modelsSection, _ := config["models"].(map[string]any)
	if modelsSection == nil {
		modelsSection = make(map[string]any)
	}
	providers, _ := modelsSection["providers"].(map[string]any)
	if providers == nil {
		providers = make(map[string]any)
	}
	ollama, _ := providers["ollama"].(map[string]any)
	if ollama == nil {
		ollama = make(map[string]any)
	}

	ollama["baseUrl"] = envconfig.Host().String() + "/v1"
	// needed to register provider
	ollama["apiKey"] = "ollama-local"
	// TODO(parthsareen): potentially move to responses
	ollama["api"] = "openai-completions"

	// Build map of existing models to preserve user customizations
	existingModels, _ := ollama["models"].([]any)
	existingByID := make(map[string]map[string]any)
	for _, m := range existingModels {
		if entry, ok := m.(map[string]any); ok {
			if id, ok := entry["id"].(string); ok {
				existingByID[id] = entry
			}
		}
	}

	var newModels []any
	for _, model := range models {
		entry := map[string]any{
			"id":        model,
			"name":      model,
			"reasoning": false,
			"input":     []any{"text"},
			"cost": map[string]any{
				"input":      0,
				"output":     0,
				"cacheRead":  0,
				"cacheWrite": 0,
			},
			// TODO(parthsareen): get these values from API
			"contextWindow": 131072,
			"maxTokens":     16384,
		}
		// Merge existing fields (user customizations)
		if existing, ok := existingByID[model]; ok {
			for k, v := range existing {
				if _, isNew := entry[k]; !isNew {
					entry[k] = v
				}
			}
		}
		newModels = append(newModels, entry)
	}
	ollama["models"] = newModels

	providers["ollama"] = ollama
	modelsSection["providers"] = providers
	config["models"] = modelsSection

	// Update agents.defaults.model.primary (preserving other agent settings)
	agents, _ := config["agents"].(map[string]any)
	if agents == nil {
		agents = make(map[string]any)
	}
	defaults, _ := agents["defaults"].(map[string]any)
	if defaults == nil {
		defaults = make(map[string]any)
	}
	modelConfig, _ := defaults["model"].(map[string]any)
	if modelConfig == nil {
		modelConfig = make(map[string]any)
	}
	modelConfig["primary"] = "ollama/" + models[0]
	defaults["model"] = modelConfig
	agents["defaults"] = defaults
	config["agents"] = agents

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	return writeWithBackup(configPath, data)
}

func (c *Clawdbot) Models() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	config, err := readJSONFile(filepath.Join(home, ".clawdbot", "clawdbot.json"))
	if err != nil {
		return nil
	}

	modelsSection, _ := config["models"].(map[string]any)
	providers, _ := modelsSection["providers"].(map[string]any)
	ollama, _ := providers["ollama"].(map[string]any)
	modelList, _ := ollama["models"].([]any)

	var result []string
	for _, m := range modelList {
		if entry, ok := m.(map[string]any); ok {
			if id, ok := entry["id"].(string); ok {
				result = append(result, id)
			}
		}
	}
	return result
}
