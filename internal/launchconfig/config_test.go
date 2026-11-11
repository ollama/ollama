package launchconfig

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadIntegrationReadsLegacyConfigWithoutMigration(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	path, _ := Path()
	legacy := filepath.Join(filepath.Dir(path), "config", "config.json")
	if err := os.MkdirAll(filepath.Dir(legacy), 0o700); err != nil {
		t.Fatal(err)
	}
	data := []byte(`{"unknown":"retain","integrations":{"claude-desktop":{"models":["local"],"aliases":{"claude-sonnet-5":"local"},"automode":false}}}`)
	if err := os.WriteFile(legacy, data, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadIntegration("claude-desktop")
	if err != nil || len(cfg.Models) != 1 || cfg.Aliases["claude-sonnet-5"] != "local" {
		t.Fatalf("config = %+v, %v", cfg, err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("read migrated config: %v", err)
	}
	after, err := os.ReadFile(legacy)
	if err != nil || string(after) != string(data) {
		t.Fatalf("legacy config changed: %q %v", after, err)
	}
}

func TestLoadIntegrationRejectsMalformedSavedState(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	path, _ := Path()
	os.MkdirAll(filepath.Dir(path), 0o700)
	for _, data := range []string{`null`, `{`, `{"integrations":{"claude-desktop":null}}`, `{"integrations":{"claude-desktop":{"models":"wrong"}}}`} {
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadIntegration("claude-desktop"); err == nil || os.IsNotExist(err) {
			t.Fatalf("invalid config %s returned %v", data, err)
		}
	}
}
