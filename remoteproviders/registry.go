package remoteproviders

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Channel represents a configured upstream provider.
type Channel struct {
	ID           string            `json:"id"`
	Type         string            `json:"type"` // e.g. "openai"
	BaseURL      string            `json:"base_url"`
	APIKey       string            `json:"api_key"`
	DefaultModel string            `json:"default_model,omitempty"`
	Headers      map[string]string `json:"headers,omitempty"`
}

// Config is the root configuration for remote providers.
type Config struct {
	Channels []Channel `json:"channels"`
}

var (
	mu           sync.RWMutex
	cached       Config
	cachedPath   string
	cachedMod    time.Time
	cachedLoaded bool
)

// ListChannels returns all configured channels.
func ListChannels(path string) ([]Channel, error) {
	cfg, err := loadConfig(path)
	if err != nil {
		return nil, err
	}

	return append([]Channel(nil), cfg.Channels...), nil
}

// GetChannel loads the config (with caching) and returns the channel by id.
func GetChannel(path, id string) (Channel, error) {
	cfg, err := loadConfig(path)
	if err != nil {
		return Channel{}, err
	}

	for _, ch := range cfg.Channels {
		if ch.ID == id {
			return ch, nil
		}
	}

	return Channel{}, fmt.Errorf("remote channel not found: %s", id)
}

// UpsertChannel creates or updates a channel. If APIKey is empty, it will be preserved
// from the existing channel (if any).
func UpsertChannel(path string, input Channel) (Channel, error) {
	mu.Lock()
	defer mu.Unlock()

	cfg, _, err := readConfig(path)
	if err != nil {
		return Channel{}, err
	}

	found := false
	for i, ch := range cfg.Channels {
		if ch.ID == input.ID {
			found = true
			if strings.TrimSpace(input.APIKey) == "" {
				input.APIKey = ch.APIKey
			}
			cfg.Channels[i] = input
			break
		}
	}
	if !found {
		if strings.TrimSpace(input.APIKey) == "" {
			return Channel{}, errors.New("api_key is required for new channel")
		}
		cfg.Channels = append(cfg.Channels, input)
	}

	if err := validateConfig(cfg); err != nil {
		return Channel{}, err
	}

	if err := writeConfig(path, cfg); err != nil {
		return Channel{}, err
	}

	updateCache(path, cfg)
	return input, nil
}

// DeleteChannel removes a channel by id.
func DeleteChannel(path, id string) error {
	mu.Lock()
	defer mu.Unlock()

	cfg, _, err := readConfig(path)
	if err != nil {
		return err
	}

	out := cfg.Channels[:0]
	found := false
	for _, ch := range cfg.Channels {
		if ch.ID == id {
			found = true
			continue
		}
		out = append(out, ch)
	}
	cfg.Channels = out

	if !found {
		return fmt.Errorf("remote channel not found: %s", id)
	}

	if err := writeConfig(path, cfg); err != nil {
		return err
	}
	updateCache(path, cfg)
	return nil
}

// RedactChannel returns a copy of the channel with masked API key.
func RedactChannel(ch Channel) Channel {
	out := ch
	if out.APIKey != "" {
		out.APIKey = maskKey(out.APIKey)
	}
	return out
}

// loadConfig reads the config file if it has changed.
func loadConfig(path string) (Config, error) {
	mu.RLock()
	if cachedLoaded && cachedPath == path {
		info, err := os.Stat(path)
		if err == nil && !info.ModTime().After(cachedMod) {
			cfg := cached
			mu.RUnlock()
			return cfg, nil
		}
	}
	mu.RUnlock()

	mu.Lock()
	defer mu.Unlock()

	// Re-check inside write lock
	if cachedLoaded && cachedPath == path {
		info, err := os.Stat(path)
		if err == nil && !info.ModTime().After(cachedMod) {
			return cached, nil
		}
	}

	cfg, mod, err := readConfig(path)
	if err != nil {
		return Config{}, err
	}

	cached = cfg
	cachedMod = mod
	cachedPath = path
	cachedLoaded = true

	return cfg, nil
}

func updateCache(path string, cfg Config) {
	cached = cfg
	cachedPath = path
	cachedLoaded = true
	if info, err := os.Stat(path); err == nil {
		cachedMod = info.ModTime()
	} else {
		cachedMod = time.Now()
	}
}

func readConfig(path string) (Config, time.Time, error) {
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Config{}, time.Time{}, nil
		}
		return Config{}, time.Time{}, fmt.Errorf("stat remote providers: %w", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, time.Time{}, fmt.Errorf("read remote providers: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, time.Time{}, fmt.Errorf("parse remote providers: %w", err)
	}

	if err := validateConfig(cfg); err != nil {
		return Config{}, time.Time{}, err
	}

	return cfg, info.ModTime(), nil
}

func writeConfig(path string, cfg Config) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir config dir: %w", err)
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("encode remote providers: %w", err)
	}

	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write remote providers: %w", err)
	}

	return nil
}

func validateConfig(cfg Config) error {
	seen := map[string]struct{}{}
	for i, ch := range cfg.Channels {
		if err := validateChannel(ch); err != nil {
			return fmt.Errorf("channel[%d]: %w", i, err)
		}
		if _, ok := seen[ch.ID]; ok {
			return fmt.Errorf("duplicate channel id: %s", ch.ID)
		}
		seen[ch.ID] = struct{}{}
	}
	return nil
}

func validateChannel(ch Channel) error {
	if strings.TrimSpace(ch.ID) == "" {
		return errors.New("id is required")
	}
	if strings.TrimSpace(ch.Type) == "" {
		return errors.New("type is required")
	}
	if strings.TrimSpace(ch.BaseURL) == "" {
		return errors.New("base_url is required")
	}
	if strings.TrimSpace(ch.APIKey) == "" {
		return errors.New("api_key is required")
	}

	u, err := url.Parse(ch.BaseURL)
	if err != nil {
		return fmt.Errorf("invalid base_url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return errors.New("base_url must include http or https scheme")
	}
	if u.Host == "" {
		return errors.New("base_url host is empty")
	}

	return nil
}

func maskKey(key string) string {
	trimmed := strings.TrimSpace(key)
	if trimmed == "" {
		return ""
	}
	if len(trimmed) <= 8 {
		return "****"
	}
	return trimmed[:4] + "..." + trimmed[len(trimmed)-4:]
}
