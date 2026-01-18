package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

type ConnectionConfig struct {
	App          string    `json:"app"`
	Model        string    `json:"model"`
	ConfiguredAt time.Time `json:"configured_at"`
}

func configPath(appName string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "connections", appName+".json"), nil
}

func SaveConnection(appName, model string) error {
	path, err := configPath(appName)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(ConnectionConfig{
		App:          appName,
		Model:        model,
		ConfiguredAt: time.Now(),
	}, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

func LoadConnection(appName string) (*ConnectionConfig, error) {
	path, err := configPath(appName)
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config ConnectionConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func ListConnections() ([]ConnectionConfig, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	dir := filepath.Join(home, ".ollama", "connections")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var configs []ConnectionConfig
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			continue
		}

		var config ConnectionConfig
		if err := json.Unmarshal(data, &config); err != nil {
			continue
		}
		configs = append(configs, config)
	}

	return configs, nil
}
