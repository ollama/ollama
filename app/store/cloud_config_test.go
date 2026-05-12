//go:build windows || darwin

package store

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCloudDisabled(t *testing.T) {
	tests := []struct {
		name          string
		envValue      string
		configContent string
		wantDisabled  bool
		wantSource    string
	}{
		{
			name:         "default enabled",
			wantDisabled: false,
			wantSource:   "none",
		},
		{
			name:         "env disables cloud",
			envValue:     "1",
			wantDisabled: true,
			wantSource:   "env",
		},
		{
			name:          "config disables cloud",
			configContent: `{"disable_ollama_cloud": true}`,
			wantDisabled:  true,
			wantSource:    "config",
		},
		{
			name:          "env and config",
			envValue:      "1",
			configContent: `{"disable_ollama_cloud": false}`,
			wantDisabled:  true,
			wantSource:    "env",
		},
		{
			name:          "invalid config is ignored",
			configContent: `{bad`,
			wantDisabled:  false,
			wantSource:    "none",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpHome := t.TempDir()
			setTestHome(t, tmpHome)
			t.Setenv("OLLAMA_NO_CLOUD", tt.envValue)

			if tt.configContent != "" {
				configDir := filepath.Join(tmpHome, ".ollama")
				if err := os.MkdirAll(configDir, 0o755); err != nil {
					t.Fatalf("mkdir config dir: %v", err)
				}
				configPath := filepath.Join(configDir, serverConfigFilename)
				if err := os.WriteFile(configPath, []byte(tt.configContent), 0o644); err != nil {
					t.Fatalf("write config: %v", err)
				}
			}

			s := &Store{DBPath: filepath.Join(tmpHome, "db.sqlite")}
			defer s.Close()

			disabled, err := s.CloudDisabled()
			if err != nil {
				t.Fatalf("CloudDisabled() error = %v", err)
			}
			if disabled != tt.wantDisabled {
				t.Fatalf("CloudDisabled() = %v, want %v", disabled, tt.wantDisabled)
			}

			statusDisabled, source, err := s.CloudStatus()
			if err != nil {
				t.Fatalf("CloudStatus() error = %v", err)
			}
			if statusDisabled != tt.wantDisabled {
				t.Fatalf("CloudStatus() disabled = %v, want %v", statusDisabled, tt.wantDisabled)
			}
			if source != tt.wantSource {
				t.Fatalf("CloudStatus() source = %v, want %v", source, tt.wantSource)
			}
		})
	}
}

func TestSetCloudEnabled(t *testing.T) {
	tmpHome := t.TempDir()
	setTestHome(t, tmpHome)

	configDir := filepath.Join(tmpHome, ".ollama")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	configPath := filepath.Join(configDir, serverConfigFilename)
	if err := os.WriteFile(configPath, []byte(`{"another_key":"value","disable_ollama_cloud":true}`), 0o644); err != nil {
		t.Fatalf("seed config: %v", err)
	}

	s := &Store{DBPath: filepath.Join(tmpHome, "db.sqlite")}
	defer s.Close()

	if err := s.SetCloudEnabled(true); err != nil {
		t.Fatalf("SetCloudEnabled(true) error = %v", err)
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal config: %v", err)
	}

	if got["disable_ollama_cloud"] != false {
		t.Fatalf("disable_ollama_cloud = %v, want false", got["disable_ollama_cloud"])
	}
	if got["another_key"] != "value" {
		t.Fatalf("another_key = %v, want value", got["another_key"])
	}
}
