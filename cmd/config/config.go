// Package config provides integration configuration for external coding tools
// (Claude Code, Codex, Droid, OpenCode) to use Ollama models.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

type integration struct {
	Models  []string          `json:"models"`
	Aliases map[string]string `json:"aliases,omitempty"`
}

type config struct {
	Integrations map[string]*integration `json:"integrations"`
}

func configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "config.json"), nil
}

func legacyConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "config", "config.json"), nil
}

// migrateConfig moves the config from the legacy path to ~/.ollama/config.json
func migrateConfig() (bool, error) {
	oldPath, err := legacyConfigPath()
	if err != nil {
		return false, err
	}

	oldData, err := os.ReadFile(oldPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}

	var js json.RawMessage
	if err := json.Unmarshal(oldData, &js); err != nil {
		slog.Warn("legacy config has invalid JSON, skipping migration", "path", oldPath, "error", err)
		return false, nil
	}

	newPath, err := configPath()
	if err != nil {
		return false, err
	}

	if err := os.MkdirAll(filepath.Dir(newPath), 0o755); err != nil {
		return false, err
	}
	if err := os.WriteFile(newPath, oldData, 0o644); err != nil {
		return false, fmt.Errorf("write new config: %w", err)
	}

	_ = os.Remove(oldPath)
	_ = os.Remove(filepath.Dir(oldPath)) // clean up empty directory

	slog.Info("migrated config", "from", oldPath, "to", newPath)
	return true, nil
}

func load() (*config, error) {
	path, err := configPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil && os.IsNotExist(err) {
		if migrated, merr := migrateConfig(); merr == nil && migrated {
			data, err = os.ReadFile(path)
		}
	}
	if err != nil {
		if os.IsNotExist(err) {
			return &config{Integrations: make(map[string]*integration)}, nil
		}
		return nil, err
	}

	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w, at: %s", err, path)
	}
	if cfg.Integrations == nil {
		cfg.Integrations = make(map[string]*integration)
	}
	return &cfg, nil
}

func save(cfg *config) error {
	path, err := configPath()
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

	cfg, err := load()
	if err != nil {
		return err
	}

	key := strings.ToLower(appName)
	existing := cfg.Integrations[key]
	var aliases map[string]string
	if existing != nil && existing.Aliases != nil {
		aliases = existing.Aliases
	}

	cfg.Integrations[key] = &integration{
		Models:  models,
		Aliases: aliases,
	}

	return save(cfg)
}

func loadIntegration(appName string) (*integration, error) {
	cfg, err := load()
	if err != nil {
		return nil, err
	}

	ic, ok := cfg.Integrations[strings.ToLower(appName)]
	if !ok {
		return nil, os.ErrNotExist
	}

	return ic, nil
}

func saveAliases(appName string, aliases map[string]string) error {
	if appName == "" {
		return errors.New("app name cannot be empty")
	}

	cfg, err := load()
	if err != nil {
		return err
	}

	key := strings.ToLower(appName)
	existing := cfg.Integrations[key]
	if existing == nil {
		existing = &integration{}
	}

	if existing.Aliases == nil {
		existing.Aliases = make(map[string]string)
	}
	for k, v := range aliases {
		existing.Aliases[k] = v
	}

	cfg.Integrations[key] = existing
	return save(cfg)
}

func listIntegrations() ([]integration, error) {
	cfg, err := load()
	if err != nil {
		return nil, err
	}

	result := make([]integration, 0, len(cfg.Integrations))
	for _, ic := range cfg.Integrations {
		result = append(result, *ic)
	}

	return result, nil
}
