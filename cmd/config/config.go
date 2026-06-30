// Package config provides shared Ollama CLI configuration, including launch
// integration settings and scoped onboarding state.
package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

type integration struct {
	Models    []string          `json:"models"`
	Aliases   map[string]string `json:"aliases,omitempty"`
	Onboarded bool              `json:"onboarded,omitempty"`
}

// IntegrationConfig is the persisted config for one integration.
type IntegrationConfig = integration

type config struct {
	Integrations  map[string]*integration    `json:"integrations"`
	Onboarding    map[string]map[string]bool `json:"onboarding,omitempty"`
	LastModel     string                     `json:"last_model,omitempty"`
	LastSelection string                     `json:"last_selection,omitempty"` // "run" or integration name
}

const (
	OnboardingSectionApp        = "app"
	OnboardingKeyTerminalPrompt = "terminal_prompt"
)

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

	// Ignore legacy files with invalid JSON and continue startup.
	if !json.Valid(oldData) {
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

	var raw struct {
		Integrations  map[string]*integration `json:"integrations"`
		Onboarding    json.RawMessage         `json:"onboarding,omitempty"`
		LastModel     string                  `json:"last_model,omitempty"`
		LastSelection string                  `json:"last_selection,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w, at: %s", err, path)
	}
	cfg := config{
		Integrations:  raw.Integrations,
		Onboarding:    parseOnboarding(raw.Onboarding),
		LastModel:     raw.LastModel,
		LastSelection: raw.LastSelection,
	}
	if cfg.Integrations == nil {
		cfg.Integrations = make(map[string]*integration)
	}
	if cfg.Onboarding == nil {
		cfg.Onboarding = make(map[string]map[string]bool)
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

	return fileutil.WriteWithBackup(path, data)
}

func SaveIntegration(appName string, models []string) error {
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
	var onboarded bool
	if existing != nil {
		aliases = existing.Aliases
		onboarded = existing.Onboarded
	}

	cfg.Integrations[key] = &integration{
		Models:    models,
		Aliases:   aliases,
		Onboarded: onboarded,
	}

	return save(cfg)
}

// MarkIntegrationOnboarded marks an integration as onboarded in Ollama's config.
func MarkIntegrationOnboarded(appName string) error {
	cfg, err := load()
	if err != nil {
		return err
	}

	key := strings.ToLower(appName)
	existing := cfg.Integrations[key]
	if existing == nil {
		existing = &integration{}
	}
	existing.Onboarded = true
	cfg.Integrations[key] = existing
	return save(cfg)
}

// IntegrationModel returns the first configured model for an integration, or empty string if not configured.
func IntegrationModel(appName string) string {
	integrationConfig, err := LoadIntegration(appName)
	if err != nil || len(integrationConfig.Models) == 0 {
		return ""
	}
	return integrationConfig.Models[0]
}

// IntegrationModels returns all configured models for an integration, or nil.
func IntegrationModels(appName string) []string {
	integrationConfig, err := LoadIntegration(appName)
	if err != nil || len(integrationConfig.Models) == 0 {
		return nil
	}
	return integrationConfig.Models
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

// OnboardingCompleted returns true when the named onboarding item has been completed.
func OnboardingCompleted(section string, key string) bool {
	completed, err := GetOnboardingCompleted(section, key)
	return err == nil && completed
}

// GetOnboardingCompleted returns whether the named onboarding item has been completed.
func GetOnboardingCompleted(section string, key string) (bool, error) {
	completed, _, err := LookupOnboardingCompleted(section, key)
	return completed, err
}

// LookupOnboardingCompleted returns whether the named onboarding item exists
// and, if so, whether it has been completed.
func LookupOnboardingCompleted(section string, key string) (bool, bool, error) {
	section, key, err := normalizeOnboardingPath(section, key)
	if err != nil {
		return false, false, err
	}

	cfg, err := load()
	if err != nil {
		return false, false, err
	}

	values, ok := cfg.Onboarding[section]
	if !ok {
		return false, false, nil
	}
	completed, ok := values[key]
	return completed, ok, nil
}

// MarkOnboardingCompleted marks the named onboarding item as completed.
func MarkOnboardingCompleted(section string, key string) error {
	return SetOnboardingCompleted(section, key, true)
}

// SetOnboardingCompleted records whether the named onboarding item has been completed.
func SetOnboardingCompleted(section string, key string, completed bool) error {
	section, key, err := normalizeOnboardingPath(section, key)
	if err != nil {
		return err
	}

	cfg, err := load()
	if err != nil {
		return err
	}
	if cfg.Onboarding == nil {
		cfg.Onboarding = make(map[string]map[string]bool)
	}
	if cfg.Onboarding[section] == nil {
		cfg.Onboarding[section] = make(map[string]bool)
	}

	cfg.Onboarding[section][key] = completed
	return save(cfg)
}

func normalizeOnboardingPath(section string, key string) (string, string, error) {
	section = strings.ToLower(strings.TrimSpace(section))
	key = strings.ToLower(strings.TrimSpace(key))
	if section == "" {
		return "", "", errors.New("onboarding section cannot be empty")
	}
	if key == "" {
		return "", "", errors.New("onboarding key cannot be empty")
	}
	return section, key, nil
}

func parseOnboarding(raw json.RawMessage) map[string]map[string]bool {
	onboarding := make(map[string]map[string]bool)
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return onboarding
	}

	var nested map[string]map[string]bool
	if err := json.Unmarshal(raw, &nested); err == nil {
		for section, values := range nested {
			section = strings.ToLower(strings.TrimSpace(section))
			if section == "" {
				continue
			}
			if onboarding[section] == nil {
				onboarding[section] = make(map[string]bool)
			}
			for key, completed := range values {
				key = strings.ToLower(strings.TrimSpace(key))
				if key == "" {
					continue
				}
				onboarding[section][key] = completed
			}
		}
		return onboarding
	}

	var flat map[string]bool
	if err := json.Unmarshal(raw, &flat); err == nil {
		for scope, completed := range flat {
			section, key := splitLegacyOnboardingScope(scope)
			if section == "" || key == "" {
				continue
			}
			if onboarding[section] == nil {
				onboarding[section] = make(map[string]bool)
			}
			onboarding[section][key] = completed
		}
	}

	return onboarding
}

func splitLegacyOnboardingScope(scope string) (string, string) {
	scope = strings.ToLower(strings.TrimSpace(scope))
	if scope == "" {
		return "", ""
	}
	if scope == "app_terminal_prompt" {
		return OnboardingSectionApp, OnboardingKeyTerminalPrompt
	}
	if section, key, ok := strings.Cut(scope, "."); ok {
		section = strings.TrimSpace(section)
		key = strings.TrimSpace(key)
		return section, key
	}
	return "legacy", scope
}

// LoadIntegration returns the saved config for one integration.
func LoadIntegration(appName string) (*integration, error) {
	cfg, err := load()
	if err != nil {
		return nil, err
	}

	integrationConfig, ok := cfg.Integrations[strings.ToLower(appName)]
	if !ok {
		return nil, os.ErrNotExist
	}

	return integrationConfig, nil
}

// SaveAliases replaces the saved aliases for one integration.
func SaveAliases(appName string, aliases map[string]string) error {
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
	for _, integrationConfig := range cfg.Integrations {
		result = append(result, *integrationConfig)
	}

	return result, nil
}
