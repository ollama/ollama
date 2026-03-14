// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package launch

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOctomindString(t *testing.T) {
	o := &Octomind{}
	if got := o.String(); got != "Octomind" {
		t.Errorf("Octomind.String() = %q, want %q", got, "Octomind")
	}
}

func TestOctomindPaths(t *testing.T) {
	o := &Octomind{}
	paths := o.Paths()
	// Returns nil if config doesn't exist
	if paths != nil && len(paths) > 0 {
		// If config exists, verify path
		home, _ := os.UserHomeDir()
		expected := filepath.Join(home, ".config", "octomind", "config.toml")
		if paths[0] != expected {
			t.Errorf("Octomind.Paths()[0] = %q, want %q", paths[0], expected)
		}
	}
}

func TestOctomindModels(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected []string
	}{
		{
			name: "ollama model",
			content: `model = "ollama:qwen3-coder"

[providers.ollama]
base_url = "http://localhost:11434/v1"
`,
			expected: []string{"qwen3-coder"},
		},
		{
			name: "non-ollama model",
			content: `model = "openai:gpt-4"

[providers.openai]
api_key = "sk-..."
`,
			expected: nil,
		},
		{
			name:     "empty config",
			content:  "",
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temp config file
			tmpDir := t.TempDir()
			configDir := filepath.Join(tmpDir, ".config", "octomind")
			if err := os.MkdirAll(configDir, 0o755); err != nil {
				t.Fatal(err)
			}
			configPath := filepath.Join(configDir, "config.toml")
			if err := os.WriteFile(configPath, []byte(tt.content), 0o644); err != nil {
				t.Fatal(err)
			}

			// Override home for test
			originalHome := os.Getenv("HOME")
			os.Setenv("HOME", tmpDir)
			defer os.Setenv("HOME", originalHome)

			o := &Octomind{}
			models := o.Models()

			if tt.expected == nil {
				if models != nil {
					t.Errorf("Octomind.Models() = %v, want nil", models)
				}
				return
			}

			if len(models) != len(tt.expected) {
				t.Errorf("Octomind.Models() length = %d, want %d", len(models), len(tt.expected))
				return
			}

			for i, m := range models {
				if m != tt.expected[i] {
					t.Errorf("Octomind.Models()[%d] = %q, want %q", i, m, tt.expected[i])
				}
			}
		})
	}
}

func TestOctomindEdit(t *testing.T) {
	tests := []struct {
		name         string
		existing     string
		models       []string
		wantContains string
	}{
		{
			name:         "new config",
			existing:     "",
			models:       []string{"qwen3-coder"},
			wantContains: `model = "ollama:qwen3-coder"`,
		},
		{
			name:         "update existing",
			existing:     `model = "ollama:old-model"`,
			models:       []string{"glm-4.7-flash"},
			wantContains: `model = "ollama:glm-4.7-flash"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			configDir := filepath.Join(tmpDir, ".config", "octomind")
			if err := os.MkdirAll(configDir, 0o755); err != nil {
				t.Fatal(err)
			}

			if tt.existing != "" {
				configPath := filepath.Join(configDir, "config.toml")
				if err := os.WriteFile(configPath, []byte(tt.existing), 0o644); err != nil {
					t.Fatal(err)
				}
			}

			// Override home for test
			originalHome := os.Getenv("HOME")
			os.Setenv("HOME", tmpDir)
			defer os.Setenv("HOME", originalHome)

			o := &Octomind{}
			if err := o.Edit(tt.models); err != nil {
				t.Errorf("Octomind.Edit() error = %v", err)
				return
			}

			// Verify config was created
			configPath := filepath.Join(configDir, "config.toml")
			data, err := os.ReadFile(configPath)
			if err != nil {
				t.Errorf("Failed to read config: %v", err)
				return
			}

			if !contains(string(data), tt.wantContains) {
				t.Errorf("Config does not contain expected content.\nGot: %s\nWant: %s", string(data), tt.wantContains)
			}
		})
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}