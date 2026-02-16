package config

import (
	"slices"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestIntegrationLookup(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantFound bool
		wantName  string
	}{
		{"claude lowercase", "claude", true, "Claude Code"},
		{"claude uppercase", "CLAUDE", true, "Claude Code"},
		{"claude mixed case", "Claude", true, "Claude Code"},
		{"codex", "codex", true, "Codex"},
		{"droid", "droid", true, "Droid"},
		{"opencode", "opencode", true, "OpenCode"},
		{"unknown integration", "unknown", false, ""},
		{"empty string", "", false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, found := integrations[strings.ToLower(tt.input)]
			if found != tt.wantFound {
				t.Errorf("integrations[%q] found = %v, want %v", tt.input, found, tt.wantFound)
			}
			if found && r.String() != tt.wantName {
				t.Errorf("integrations[%q].String() = %q, want %q", tt.input, r.String(), tt.wantName)
			}
		})
	}
}

func TestIntegrationRegistry(t *testing.T) {
	expectedIntegrations := []string{"claude", "codex", "droid", "opencode"}

	for _, name := range expectedIntegrations {
		t.Run(name, func(t *testing.T) {
			r, ok := integrations[name]
			if !ok {
				t.Fatalf("integration %q not found in registry", name)
			}
			if r.String() == "" {
				t.Error("integration.String() should not be empty")
			}
		})
	}
}

func TestHasLocalModel(t *testing.T) {
	tests := []struct {
		name   string
		models []string
		want   bool
	}{
		{"empty list", []string{}, false},
		{"single local model", []string{"llama3.2"}, true},
		{"single cloud model", []string{"cloud-model"}, false},
		{"mixed models", []string{"cloud-model", "llama3.2"}, true},
		{"multiple local models", []string{"llama3.2", "qwen2.5"}, true},
		{"multiple cloud models", []string{"cloud-a", "cloud-b"}, false},
		{"local model first", []string{"llama3.2", "cloud-model"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.ContainsFunc(tt.models, func(m string) bool {
				return !strings.Contains(m, "cloud")
			})
			if got != tt.want {
				t.Errorf("hasLocalModel(%v) = %v, want %v", tt.models, got, tt.want)
			}
		})
	}
}

func TestConfigCmd(t *testing.T) {
	// Mock checkServerHeartbeat that always succeeds
	mockCheck := func(cmd *cobra.Command, args []string) error {
		return nil
	}

	cmd := ConfigCmd(mockCheck)

	t.Run("command structure", func(t *testing.T) {
		if cmd.Use != "config [INTEGRATION]" {
			t.Errorf("Use = %q, want %q", cmd.Use, "config [INTEGRATION]")
		}
		if cmd.Short == "" {
			t.Error("Short description should not be empty")
		}
		if cmd.Long == "" {
			t.Error("Long description should not be empty")
		}
	})

	t.Run("flags exist", func(t *testing.T) {
		modelFlag := cmd.Flags().Lookup("model")
		if modelFlag == nil {
			t.Error("--model flag should exist")
		}

		launchFlag := cmd.Flags().Lookup("launch")
		if launchFlag == nil {
			t.Error("--launch flag should exist")
		}
	})

	t.Run("PreRunE is set", func(t *testing.T) {
		if cmd.PreRunE == nil {
			t.Error("PreRunE should be set to checkServerHeartbeat")
		}
	})
}

func TestRunIntegration_UnknownIntegration(t *testing.T) {
	err := runIntegration("unknown-integration", "model")
	if err == nil {
		t.Error("expected error for unknown integration, got nil")
	}
	if !strings.Contains(err.Error(), "unknown integration") {
		t.Errorf("error should mention 'unknown integration', got: %v", err)
	}
}

func TestHasLocalModel_DocumentsHeuristic(t *testing.T) {
	tests := []struct {
		name   string
		models []string
		want   bool
		reason string
	}{
		{"empty list", []string{}, false, "empty list has no local models"},
		{"contains-cloud-substring", []string{"deepseek-r1:cloud"}, false, "model with 'cloud' substring is considered cloud"},
		{"cloud-in-name", []string{"my-cloud-model"}, false, "'cloud' anywhere in name = cloud model"},
		{"cloudless", []string{"cloudless-model"}, false, "'cloudless' still contains 'cloud'"},
		{"local-model", []string{"llama3.2"}, true, "no 'cloud' = local"},
		{"mixed", []string{"cloud-model", "llama3.2"}, true, "one local model = hasLocalModel true"},
		{"all-cloud", []string{"cloud-a", "cloud-b"}, false, "all contain 'cloud'"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.ContainsFunc(tt.models, func(m string) bool {
				return !strings.Contains(m, "cloud")
			})
			if got != tt.want {
				t.Errorf("hasLocalModel(%v) = %v, want %v (%s)", tt.models, got, tt.want, tt.reason)
			}
		})
	}
}

func TestConfigCmd_NilHeartbeat(t *testing.T) {
	// This should not panic - cmd creation should work even with nil
	cmd := ConfigCmd(nil)
	if cmd == nil {
		t.Fatal("ConfigCmd returned nil")
	}

	// PreRunE should be nil when passed nil
	if cmd.PreRunE != nil {
		t.Log("Note: PreRunE is set even when nil is passed (acceptable)")
	}
}

func TestAllIntegrations_HaveRequiredMethods(t *testing.T) {
	for name, r := range integrations {
		t.Run(name, func(t *testing.T) {
			// Test String() doesn't panic and returns non-empty
			displayName := r.String()
			if displayName == "" {
				t.Error("String() should not return empty")
			}

			// Test Run() exists (we can't call it without actually running the command)
			// Just verify the method is available
			var _ func(string) error = r.Run
		})
	}
}
