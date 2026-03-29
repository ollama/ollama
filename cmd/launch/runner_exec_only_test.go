package launch

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEditorRunsDoNotRewriteConfig(t *testing.T) {
	tests := []struct {
		name      string
		binary    string
		runner    Runner
		checkPath func(home string) string
	}{
		{
			name:   "droid",
			binary: "droid",
			runner: &Droid{},
			checkPath: func(home string) string {
				return filepath.Join(home, ".factory", "settings.json")
			},
		},
		{
			name:   "opencode",
			binary: "opencode",
			runner: &OpenCode{},
			checkPath: func(home string) string {
				return filepath.Join(home, ".config", "opencode", "opencode.json")
			},
		},
		{
			name:   "cline",
			binary: "cline",
			runner: &Cline{},
			checkPath: func(home string) string {
				return filepath.Join(home, ".cline", "data", "globalState.json")
			},
		},
		{
			name:   "pi",
			binary: "pi",
			runner: &Pi{},
			checkPath: func(home string) string {
				return filepath.Join(home, ".pi", "agent", "models.json")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			home := t.TempDir()
			setTestHome(t, home)

			binDir := t.TempDir()
			writeFakeBinary(t, binDir, tt.binary)
			if tt.name == "pi" {
				writeFakeBinary(t, binDir, "npm")
			}
			t.Setenv("PATH", binDir)

			configPath := tt.checkPath(home)
			if err := tt.runner.Run("llama3.2", nil); err != nil {
				t.Fatalf("Run returned error: %v", err)
			}
			if _, err := os.Stat(configPath); !os.IsNotExist(err) {
				t.Fatalf("expected Run to leave %s untouched, got err=%v", configPath, err)
			}
		})
	}
}
