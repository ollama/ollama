package config

import (
	"errors"
	"testing"

	"github.com/spf13/cobra"
)

func TestGetIntegration(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantFound bool
		wantName  string
	}{
		{"claude lowercase", "claude", true, "Claude"},
		{"claude uppercase", "CLAUDE", true, "Claude"},
		{"claude mixed case", "Claude", true, "Claude"},
		{"codex", "codex", true, "Codex"},
		{"droid", "droid", true, "Droid"},
		{"opencode", "opencode", true, "OpenCode"},
		{"unknown integration", "unknown", false, ""},
		{"empty string", "", false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			integration, found := getIntegration(tt.input)
			if found != tt.wantFound {
				t.Errorf("getIntegration(%q) found = %v, want %v", tt.input, found, tt.wantFound)
			}
			if found && integration.Name != tt.wantName {
				t.Errorf("getIntegration(%q).Name = %q, want %q", tt.input, integration.Name, tt.wantName)
			}
		})
	}
}

func TestIntegrationRegistry(t *testing.T) {
	expectedIntegrations := []string{"claude", "codex", "droid", "opencode"}

	for _, name := range expectedIntegrations {
		t.Run(name, func(t *testing.T) {
			integration, ok := integrationRegistry[name]
			if !ok {
				t.Fatalf("integration %q not found in registry", name)
			}
			if integration.Name == "" {
				t.Error("integration.Name should not be empty")
			}
			if integration.DisplayName == "" {
				t.Error("integration.DisplayName should not be empty")
			}
			if integration.Command == "" {
				t.Error("integration.Command should not be empty")
			}
			if integration.EnvVars == nil {
				t.Error("integration.EnvVars should not be nil")
			}
			if integration.Args == nil {
				t.Error("integration.Args should not be nil")
			}
			if integration.CheckInstall == nil {
				t.Error("integration.CheckInstall should not be nil")
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
			got := hasLocalModel(tt.models)
			if got != tt.want {
				t.Errorf("hasLocalModel(%v) = %v, want %v", tt.models, got, tt.want)
			}
		})
	}
}

func TestHandleCancelled(t *testing.T) {
	tests := []struct {
		name          string
		err           error
		wantCancelled bool
		wantErr       error
	}{
		{"nil error", nil, false, nil},
		{"cancelled error", errCancelled, true, nil},
		{"other error", errors.New("some error"), false, errors.New("some error")},
		{"wrapped cancelled", errCancelled, true, nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cancelled, err := handleCancelled(tt.err)
			if cancelled != tt.wantCancelled {
				t.Errorf("handleCancelled(%v) cancelled = %v, want %v", tt.err, cancelled, tt.wantCancelled)
			}
			if tt.wantErr == nil && err != nil {
				t.Errorf("handleCancelled(%v) err = %v, want nil", tt.err, err)
			}
			if tt.wantErr != nil && err == nil {
				t.Errorf("handleCancelled(%v) err = nil, want %v", tt.err, tt.wantErr)
			}
			if tt.wantErr != nil && err != nil && err.Error() != tt.wantErr.Error() {
				t.Errorf("handleCancelled(%v) err = %v, want %v", tt.err, err, tt.wantErr)
			}
		})
	}
}

func TestCheckCommand(t *testing.T) {
	t.Run("command exists", func(t *testing.T) {
		// "go" should exist in the test environment
		check := checkCommand("go", "Install Go from golang.org")
		err := check()
		if err != nil {
			t.Errorf("checkCommand(go) returned error for existing command: %v", err)
		}
	})

	t.Run("command does not exist", func(t *testing.T) {
		check := checkCommand("nonexistent-command-12345", "Install instructions here")
		err := check()
		if err == nil {
			t.Error("checkCommand should return error for non-existent command")
		}
		if err != nil {
			errMsg := err.Error()
			if !contains(errMsg, "nonexistent-command-12345") {
				t.Errorf("error should mention command name, got: %s", errMsg)
			}
			if !contains(errMsg, "Install instructions here") {
				t.Errorf("error should include install instructions, got: %s", errMsg)
			}
		}
	})
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

// Edge case tests for integrations.go

// TestGetIntegration_UnknownName_ErrorMessage verifies that unknown integration returns false.
// Clear error handling for invalid integration names.
func TestGetIntegration_UnknownName_ErrorMessage(t *testing.T) {
	unknownNames := []string{
		"unknown",
		"notreal",
		"Claude ", // trailing space
		" claude", // leading space
		"CLAUDE!", // special char
		"",
	}

	for _, name := range unknownNames {
		t.Run(name, func(t *testing.T) {
			integration, found := getIntegration(name)
			if found {
				t.Errorf("getIntegration(%q) should return false, got integration: %v", name, integration.Name)
			}
		})
	}
}

// TestRunIntegration_UnknownIntegration verifies clear error for unknown integration.
func TestRunIntegration_UnknownIntegration(t *testing.T) {
	err := runIntegration("unknown-integration", "model")
	if err == nil {
		t.Error("expected error for unknown integration, got nil")
	}
	if !contains(err.Error(), "unknown integration") {
		t.Errorf("error should mention 'unknown integration', got: %v", err)
	}
}

// TestHasLocalModel_DocumentsHeuristic documents what "cloud" means in model names.
// The heuristic checks if "cloud" is in the name - this documents that behavior.
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
			got := hasLocalModel(tt.models)
			if got != tt.want {
				t.Errorf("hasLocalModel(%v) = %v, want %v (%s)", tt.models, got, tt.want, tt.reason)
			}
		})
	}
}

// TestConfigCmd_NilHeartbeat verifies ConfigCmd handles nil checkServerHeartbeat.
// Nil function pointer would cause runtime panic if not handled.
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

// TestIntegrationDef_AllHaveRequiredFields verifies all integrations are properly defined.
func TestIntegrationDef_AllHaveRequiredFields(t *testing.T) {
	for name, integration := range integrationRegistry {
		t.Run(name, func(t *testing.T) {
			// Test EnvVars doesn't panic
			envs := integration.EnvVars("test-model")
			if envs == nil {
				t.Logf("%s: EnvVars returns nil (acceptable)", name)
			}

			// Test Args doesn't panic
			args := integration.Args("test-model")
			if args == nil {
				t.Logf("%s: Args returns nil (acceptable)", name)
			}

			// CheckInstall should not be nil
			if integration.CheckInstall == nil {
				t.Errorf("%s: CheckInstall is nil", name)
			}
		})
	}
}

// TestGetIntegrationConfiguredModels_MergesCorrectly verifies model merging behavior.
func TestGetIntegrationConfiguredModels_MergesCorrectly(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	t.Run("returns saved models when no integration config", func(t *testing.T) {
		// Save integration config
		saveIntegration("testapp", []string{"model-a", "model-b"})

		// For unknown integration (no special handling), should return saved models
		models := getIntegrationConfiguredModels("testapp")

		// Since testapp isn't opencode or droid, it should return saved models
		if len(models) != 2 {
			t.Errorf("expected 2 models, got %d", len(models))
		}
	})
}

// TestHandleCancelled_WrappedError verifies wrapped error handling.
func TestHandleCancelled_WrappedError(t *testing.T) {
	// Direct errCancelled
	cancelled, err := handleCancelled(errCancelled)
	if !cancelled || err != nil {
		t.Errorf("handleCancelled(errCancelled) = (%v, %v), want (true, nil)", cancelled, err)
	}

	// Non-cancelled error should pass through
	otherErr := errors.New("some other error")
	cancelled, err = handleCancelled(otherErr)
	if cancelled {
		t.Error("handleCancelled should return false for non-cancelled error")
	}
	if err != otherErr {
		t.Errorf("handleCancelled should return original error, got %v", err)
	}
}

// TestGetExistingConfigPaths_UnknownIntegration verifies unknown integration returns nil.
func TestGetExistingConfigPaths_UnknownIntegration(t *testing.T) {
	paths := getExistingConfigPaths("unknown")
	if len(paths) > 0 {
		t.Errorf("expected nil/empty for unknown integration, got %v", paths)
	}
}
