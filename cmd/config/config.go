// Package config provides integration configuration for external coding tools
// (Claude Code, Codex, Droid, OpenCode) to use Ollama models.
package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type integrationConfig struct {
	Models       []string  `json:"models"`
	ConfiguredAt time.Time `json:"configured_at"`
}

// defaultModel returns the first (default) model, or empty string if none.
func (c *integrationConfig) defaultModel() string {
	if len(c.Models) > 0 {
		return c.Models[0]
	}
	return ""
}

func integrationsPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "config", "integrations.json"), nil
}

func loadIntegrationsFile() (map[string]*integrationConfig, error) {
	path, err := integrationsPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]*integrationConfig), nil
		}
		return nil, err
	}

	var configs map[string]*integrationConfig
	_ = json.Unmarshal(data, &configs) // ignore parse errors; treat as empty
	if configs == nil {
		configs = make(map[string]*integrationConfig)
	}
	return configs, nil
}

func saveIntegrationsFile(configs map[string]*integrationConfig) error {
	path, err := integrationsPath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(configs, "", "  ")
	if err != nil {
		return err
	}

	return atomicWrite(path, data)
}

func saveIntegration(appName string, models []string) error {
	if appName == "" {
		return errEmptyAppName
	}

	configs, err := loadIntegrationsFile()
	if err != nil {
		return err
	}

	configs[strings.ToLower(appName)] = &integrationConfig{
		Models:       models,
		ConfiguredAt: time.Now(),
	}

	return saveIntegrationsFile(configs)
}

func loadIntegration(appName string) (*integrationConfig, error) {
	configs, err := loadIntegrationsFile()
	if err != nil {
		return nil, err
	}

	config, ok := configs[strings.ToLower(appName)]
	if !ok {
		return nil, os.ErrNotExist
	}

	return config, nil
}

func listIntegrations() ([]integrationConfig, error) {
	configs, err := loadIntegrationsFile()
	if err != nil {
		return nil, err
	}

	result := make([]integrationConfig, 0, len(configs))
	for _, config := range configs {
		result = append(result, *config)
	}

	return result, nil
}

var errEmptyAppName = errors.New("app name cannot be empty")
