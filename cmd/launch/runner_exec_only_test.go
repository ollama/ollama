package launch

import (
	"fmt"
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
				return filepath.Join(home, ".local", "state", "opencode", "model.json")
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
		{
			name:   "pool",
			binary: "pool",
			runner: &Poolside{},
			checkPath: func(home string) string {
				return filepath.Join(home, ".poolside", "config")
			},
		},
		{
			name:   "kimi",
			binary: "kimi",
			runner: &Kimi{},
			checkPath: func(home string) string {
				return filepath.Join(home, ".kimi", "config.toml")
			},
		},
		{
			name:   "omp",
			binary: "omp",
			runner: &OMP{},
			checkPath: func(home string) string {
				return filepath.Join(home, ".omp", "agent", "models.yml")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "pool" && poolsideGOOS == "windows" {
				t.Skip("Poolside is intentionally unsupported on Windows")
			}

			home := t.TempDir()
			setTestHome(t, home)
			if tt.name == "opencode" {
				previous := openCodeLoadedContextLength
				openCodeLoadedContextLength = func(string) (int, error) { return 8192, nil }
				t.Cleanup(func() { openCodeLoadedContextLength = previous })
			}

			binDir := t.TempDir()
			writeFakeBinary(t, binDir, tt.binary)
			if tt.name == "pi" {
				writeFakeBinary(t, binDir, "npm")
				plugin := filepath.Join(home, "ollama-plugin")
				if err := os.MkdirAll(plugin, 0o755); err != nil {
					t.Fatal(err)
				}
				for name, data := range map[string]string{
					"package.json": `{"name":"@ollama/pi"}`,
					"index.ts":     "",
				} {
					if err := os.WriteFile(filepath.Join(plugin, name), []byte(data), 0o644); err != nil {
						t.Fatal(err)
					}
				}
				script := fmt.Sprintf("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 1.5.2; exit 0; fi\nif [ \"$1\" = \"list\" ]; then echo %q; fi\n", plugin)
				if err := os.WriteFile(filepath.Join(binDir, "pi"), []byte(script), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			if tt.name == "kimi" {
				writeFakeBinary(t, binDir, "curl")
				writeFakeBinary(t, binDir, "bash")
			}
			t.Setenv("PATH", binDir)

			configPath := tt.checkPath(home)
			if err := tt.runner.Run("llama3.2", nil, nil); err != nil {
				t.Fatalf("Run returned error: %v", err)
			}
			if _, err := os.Stat(configPath); !os.IsNotExist(err) {
				t.Fatalf("expected Run to leave %s untouched, got err=%v", configPath, err)
			}
		})
	}
}
