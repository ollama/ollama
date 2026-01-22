package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type integrationConfig struct {
	App          string    `json:"app"`
	Models       []string  `json:"models"`
	ConfiguredAt time.Time `json:"configured_at"`
}

// defaultModel returns the first (default) model, or empty string if none
func (c *integrationConfig) defaultModel() string {
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

func saveIntegration(appName string, models []string) error {
	if appName == "" {
		return fmt.Errorf("app name cannot be empty")
	}

	path, err := configPath(appName)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(integrationConfig{
		App:          appName,
		Models:       models,
		ConfiguredAt: time.Now(),
	}, "", "  ")
	if err != nil {
		return err
	}

	return atomicWrite(path, data)
}

func loadIntegration(appName string) (*integrationConfig, error) {
	path, err := configPath(appName)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config integrationConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func listIntegrations() ([]integrationConfig, error) {
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

	var configs []integrationConfig
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			continue
		}

		var config integrationConfig
		if err := json.Unmarshal(data, &config); err != nil {
			continue
		}
		configs = append(configs, config)
	}

	return configs, nil
}
