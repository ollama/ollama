// Package config provides integration configuration for external coding tools
// (Claude Code, Codex, Droid, OpenCode) to use Ollama models.
package config

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
)

type integration struct {
	Models  []string          `json:"models"`
	Aliases map[string]string `json:"aliases,omitempty"`
}

type config struct {
	Integrations  map[string]*integration `json:"integrations"`
	LastModel     string                  `json:"last_model,omitempty"`
	LastSelection string                  `json:"last_selection,omitempty"` // "run" or integration name
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

// IntegrationModel returns the first configured model for an integration, or empty string if not configured.
func IntegrationModel(appName string) string {
	ic, err := loadIntegration(appName)
	if err != nil || len(ic.Models) == 0 {
		return ""
	}
	return ic.Models[0]
}

// IntegrationModels returns all configured models for an integration, or nil.
func IntegrationModels(appName string) []string {
	ic, err := loadIntegration(appName)
	if err != nil || len(ic.Models) == 0 {
		return nil
	}
	return ic.Models
}

// LastModel returns the last model that was run, or empty string if none.
func LastModel() string {
	cfg, err := load()
	if err != nil {
		return ""
	}
	return cfg.LastModel
}

// SetLastModel saves the last model that was run.
func SetLastModel(model string) error {
	cfg, err := load()
	if err != nil {
		return err
	}
	cfg.LastModel = model
	return save(cfg)
}

// LastSelection returns the last menu selection ("run" or integration name), or empty string if none.
func LastSelection() string {
	cfg, err := load()
	if err != nil {
		return ""
	}
	return cfg.LastSelection
}

// SetLastSelection saves the last menu selection ("run" or integration name).
func SetLastSelection(selection string) error {
	cfg, err := load()
	if err != nil {
		return err
	}
	cfg.LastSelection = selection
	return save(cfg)
}

// ModelExists checks if a model exists on the Ollama server.
func ModelExists(ctx context.Context, name string) bool {
	if name == "" {
		return false
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return false
	}
	models, err := client.List(ctx)
	if err != nil {
		return false
	}
	for _, m := range models.Models {
		if m.Name == name || strings.HasPrefix(m.Name, name+":") {
			return true
		}
	}
	return false
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

	// Replace aliases entirely (not merge) so deletions are persisted
	existing.Aliases = aliases

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
