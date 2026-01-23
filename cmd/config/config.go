// Package config provides integration configuration for external coding tools
// (Claude Code, Codex, Droid, OpenCode) to use Ollama models.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type integration struct {
	Models []string `json:"models"`
}

type config struct {
	Integrations map[string]*integration `json:"integrations"`
}

func integrationsPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "config", "config.json"), nil
}

func loadIntegrationsFile() (*config, error) {
	path, err := integrationsPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &config{Integrations: make(map[string]*integration)}, nil
		}
		return nil, err
	}

	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse integrations file: %w, at: %s", err, path)
	}
	if cfg.Integrations == nil {
		cfg.Integrations = make(map[string]*integration)
	}
	return &cfg, nil
}

func saveIntegrationsFile(cfg *config) error {
	path, err := integrationsPath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}

	return writeWithBackup(path, data)
}

func saveIntegration(appName string, models []string) error {
	if appName == "" {
		return errors.New("app name cannot be empty")
	}

	cfg, err := loadIntegrationsFile()
	if err != nil {
		return err
	}

	cfg.Integrations[strings.ToLower(appName)] = &integration{
		Models: models,
	}

	return saveIntegrationsFile(cfg)
}

func loadIntegration(appName string) (*integration, error) {
	cfg, err := loadIntegrationsFile()
	if err != nil {
		return nil, err
	}

	ic, ok := cfg.Integrations[strings.ToLower(appName)]
	if !ok {
		return nil, os.ErrNotExist
	}

	return ic, nil
}

func listIntegrations() ([]integration, error) {
	cfg, err := loadIntegrationsFile()
	if err != nil {
		return nil, err
	}

	result := make([]integration, 0, len(cfg.Integrations))
	for _, ic := range cfg.Integrations {
		result = append(result, *ic)
	}

	return result, nil
}
