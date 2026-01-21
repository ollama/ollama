package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type IntegrationConfig struct {
	App          string    `json:"app"`
	Models       []string  `json:"models"`
	ConfiguredAt time.Time `json:"configured_at"`
}

// DefaultModel returns the first (default) model, or empty string if none
func (c *IntegrationConfig) DefaultModel() string {
	if len(c.Models) > 0 {
		return c.Models[0]
	}
	return ""
}

func integrationsDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "config", "integrations"), nil
}

func configPath(appName string) (string, error) {
	dir, err := integrationsDir()
	if err != nil {
		return "", err
	}
	// Normalize to lowercase for consistent file naming across case-sensitive filesystems
	return filepath.Join(dir, strings.ToLower(appName)+".json"), nil
}

func SaveIntegration(appName string, models []string) error {
	path, err := configPath(appName)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	return atomicWriteJSON(path, IntegrationConfig{
		App:          appName,
		Models:       models,
		ConfiguredAt: time.Now(),
	})
}

func LoadIntegration(appName string) (*IntegrationConfig, error) {
	path, err := configPath(appName)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config IntegrationConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func ListIntegrations() ([]IntegrationConfig, error) {
	dir, err := integrationsDir()
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var configs []IntegrationConfig
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			continue
		}

		var config IntegrationConfig
		if err := json.Unmarshal(data, &config); err != nil {
			continue
		}
		configs = append(configs, config)
	}

	return configs, nil
}
