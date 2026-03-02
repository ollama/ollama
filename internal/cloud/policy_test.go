package cloud

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStatus(t *testing.T) {
	tests := []struct {
		name          string
		envValue      string
		configContent string
		disabled      bool
		source        string
	}{
		{
			name:     "none",
			disabled: false,
			source:   "none",
		},
		{
			name:     "env only",
			envValue: "1",
			disabled: true,
			source:   "env",
		},
		{
			name:          "config only",
			configContent: `{"disable_ollama_cloud": true}`,
			disabled:      true,
			source:        "config",
		},
		{
			name:          "both",
			envValue:      "1",
			configContent: `{"disable_ollama_cloud": true}`,
			disabled:      true,
			source:        "both",
		},
		{
			name:          "invalid config ignored",
			configContent: `{invalid json`,
			disabled:      false,
			source:        "none",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			home := t.TempDir()
			if tt.configContent != "" {
				configPath := filepath.Join(home, ".ollama", "server.json")
				if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(configPath, []byte(tt.configContent), 0o644); err != nil {
					t.Fatal(err)
				}
			}

			setTestHome(t, home)
			t.Setenv("OLLAMA_NO_CLOUD", tt.envValue)

			disabled, source := Status()
			if disabled != tt.disabled {
				t.Fatalf("disabled: expected %v, got %v", tt.disabled, disabled)
			}
			if source != tt.source {
				t.Fatalf("source: expected %q, got %q", tt.source, source)
			}
		})
	}
}

func TestDisabledError(t *testing.T) {
	if got := DisabledError(""); got != DisabledMessagePrefix {
		t.Fatalf("expected %q, got %q", DisabledMessagePrefix, got)
	}

	want := DisabledMessagePrefix + ": remote inference is unavailable"
	if got := DisabledError("remote inference is unavailable"); got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}
