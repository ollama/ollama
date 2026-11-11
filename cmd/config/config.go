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
	"runtime"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/internal/onboarding"
)

type integration struct {
	Models    []string          `json:"models"`
	Primary   string            `json:"primary,omitempty"`
	Aliases   map[string]string `json:"aliases,omitempty"`
	Onboarded bool              `json:"onboarded,omitempty"`
	AutoMode  *bool             `json:"automode,omitempty"`
}

// IntegrationConfig is the persisted config for one integration.
type IntegrationConfig = integration

type config struct {
	Integrations      map[string]*integration `json:"integrations"`
	LastModel         string                  `json:"last_model,omitempty"`
	LastSelection     string                  `json:"last_selection,omitempty"` // "run" or integration name
	OnboardingVersion int                     `json:"onboarding_version,omitempty"`
}

func NeedsWelcome() (bool, error) {
	if runtime.GOOS == "linux" {
		return needsWelcomeInConfig()
	}
	state := onboarding.State{}
	completed, err := state.Completed()
	if completed {
		return false, err
	}
	// Older apps recorded completion only in SQLite.
	if onboarding.CompletedInApp() {
		if err := state.Complete(); err != nil {
			slog.Warn("could not share onboarding completion", "error", err)
		}
		return false, nil
	}
	return true, err
}

func CompleteWelcome() error {
	if runtime.GOOS == "linux" {
		return completeWelcomeInConfig()
	}
	return (onboarding.State{}).Complete()
}

// Linux keeps completion with CLI preferences; desktop platforms share the marker.
func needsWelcomeInConfig() (bool, error) {
	cfg, err := load()
	if err != nil {
		return false, err
	}
	return cfg.OnboardingVersion < onboarding.CurrentVersion, nil
}

func completeWelcomeInConfig() error {
	return updateConfigFields(func(document map[string]json.RawMessage) error {
		var version int
		if raw := document["onboarding_version"]; len(raw) > 0 {
			if err := json.Unmarshal(raw, &version); err != nil {
				return err
			}
		}
		if version >= onboarding.CurrentVersion {
			return nil
		}
		value, err := json.Marshal(onboarding.CurrentVersion)
		if err != nil {
			return err
		}
		document["onboarding_version"] = value
		return nil
	})
}

func configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "config.json"), nil
}

// updateIntegrationFields keeps one integration's changes under the same lock
// from read through write, without decoding unrelated config fields.
func updateIntegrationFields(name string, update func(map[string]json.RawMessage, json.RawMessage) error) error {
	if strings.TrimSpace(name) == "" {
		return errors.New("app name cannot be empty")
	}
	return updateConfigFields(func(document map[string]json.RawMessage) error {
		integrations := make(map[string]json.RawMessage)
		if raw := document["integrations"]; len(raw) > 0 {
			if err := json.Unmarshal(raw, &integrations); err != nil {
				return err
			}
			if integrations == nil {
				integrations = make(map[string]json.RawMessage)
			}
		}
		key := strings.ToLower(name)
		fields := make(map[string]json.RawMessage)
		if raw := integrations[key]; len(raw) > 0 {
			if err := json.Unmarshal(raw, &fields); err != nil {
				return err
			}
			if fields == nil {
				return fmt.Errorf("%s config must be an object", name)
			}
		}
		if err := update(fields, integrations[key]); err != nil {
			return err
		}
		var err error
		if len(fields) == 0 {
			delete(integrations, key)
		} else {
			integrations[key], err = json.Marshal(fields)
			if err != nil {
				return err
			}
		}
		document["integrations"], err = json.Marshal(integrations)
		return err
	})
}

func updateConfigFields(update func(map[string]json.RawMessage) error) error {
	path, err := configPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	unlock, err := fileutil.Lock(path + ".lock")
	if err != nil {
		return err
	}
	defer unlock()
	data, err := os.ReadFile(path)
	migratedFrom := ""
	if os.IsNotExist(err) {
		legacy, pathErr := legacyConfigPath()
		if pathErr != nil {
			return pathErr
		}
		data, err = os.ReadFile(legacy)
		if err == nil {
			migratedFrom = legacy
		}
	}
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	document := make(map[string]json.RawMessage)
	if len(data) > 0 {
		if err := json.Unmarshal(data, &document); err != nil {
			return err
		}
		if document == nil {
			return errors.New("launcher config must be an object")
		}
	}
	if err := update(document); err != nil {
		return err
	}
	data, err = json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(path, append(data, '\n')); err != nil {
		return err
	}
	if migratedFrom != "" {
		_ = os.Remove(migratedFrom)
		_ = os.Remove(filepath.Dir(migratedFrom))
	}
	return nil
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

	unlock, err := fileutil.Lock(newPath + ".lock")
	if err != nil {
		return false, err
	}
	defer unlock()
	// A scoped writer may have created the primary file while migration waited.
	if _, err := os.Stat(newPath); err == nil {
		return false, nil
	} else if !os.IsNotExist(err) {
		return false, err
	}
	if err := fileutil.WriteWithBackup(newPath, oldData); err != nil {
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
		if _, merr := migrateConfig(); merr == nil {
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

func SaveIntegration(appName string, models []string) error {
	value, err := json.Marshal(models)
	if err != nil {
		return err
	}
	return updateIntegrationFields(appName, func(fields map[string]json.RawMessage, _ json.RawMessage) error {
		fields["models"] = value
		return nil
	})
}

// SaveIntegrationAutoMode changes only an integration's auto mode preference,
// preserving unrelated fields and serializing with model selection updates.
func SaveIntegrationAutoMode(appName string, enabled bool) error {
	value, err := json.Marshal(enabled)
	if err != nil {
		return err
	}
	return updateIntegrationFields(appName, func(fields map[string]json.RawMessage, _ json.RawMessage) error {
		fields["automode"] = value
		return nil
	})
}

// MarkIntegrationOnboarded marks an integration as onboarded in Ollama's config.
func MarkIntegrationOnboarded(appName string) error {
	return updateIntegrationFields(appName, func(fields map[string]json.RawMessage, _ json.RawMessage) error {
		fields["onboarded"] = json.RawMessage(`true`)
		return nil
	})
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
	return saveStringConfigField("last_model", model)
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
	return saveStringConfigField("last_selection", selection)
}

func saveStringConfigField(key, value string) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return updateConfigFields(func(document map[string]json.RawMessage) error {
		if value == "" {
			delete(document, key)
		} else {
			document[key] = encoded
		}
		return nil
	})
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
	return updateIntegrationFields(appName, func(fields map[string]json.RawMessage, _ json.RawMessage) error {
		if _, exists := fields["models"]; !exists {
			fields["models"] = json.RawMessage(`null`)
		}
		if len(aliases) == 0 {
			delete(fields, "aliases")
			return nil
		}
		value, err := json.Marshal(aliases)
		if err != nil {
			return err
		}
		fields["aliases"] = value
		return nil
	})
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
