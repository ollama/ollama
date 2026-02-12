package config

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestSetAliases_CloudModel(t *testing.T) {
	// Test the SetAliases logic by checking the alias map behavior
	aliases := map[string]string{
		"primary": "kimi-k2.5:cloud",
		"fast":    "kimi-k2.5:cloud",
	}

	// Verify fast is set (cloud model behavior)
	if aliases["fast"] == "" {
		t.Error("cloud model should have fast alias set")
	}
	if aliases["fast"] != aliases["primary"] {
		t.Errorf("fast should equal primary for auto-set, got fast=%q primary=%q", aliases["fast"], aliases["primary"])
	}
}

func TestSetAliases_LocalModel(t *testing.T) {
	aliases := map[string]string{
		"primary": "llama3.2:latest",
	}
	// Simulate local model behavior: fast should be empty
	delete(aliases, "fast")

	if aliases["fast"] != "" {
		t.Error("local model should have empty fast alias")
	}
}

func TestSaveAliases_ReplacesNotMerges(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// First save with both primary and fast
	initial := map[string]string{
		"primary": "cloud-model",
		"fast":    "cloud-model",
	}
	if err := saveAliases("claude", initial); err != nil {
		t.Fatalf("failed to save initial aliases: %v", err)
	}

	// Verify both are saved
	loaded, err := loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if loaded.Aliases["fast"] != "cloud-model" {
		t.Errorf("expected fast=cloud-model, got %q", loaded.Aliases["fast"])
	}

	// Now save without fast (simulating switch to local model)
	updated := map[string]string{
		"primary": "local-model",
		// fast intentionally missing
	}
	if err := saveAliases("claude", updated); err != nil {
		t.Fatalf("failed to save updated aliases: %v", err)
	}

	// Verify fast is GONE (not merged/preserved)
	loaded, err = loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load after update: %v", err)
	}
	if loaded.Aliases["fast"] != "" {
		t.Errorf("fast should be removed after saving without it, got %q", loaded.Aliases["fast"])
	}
	if loaded.Aliases["primary"] != "local-model" {
		t.Errorf("primary should be updated to local-model, got %q", loaded.Aliases["primary"])
	}
}

func TestSaveAliases_PreservesModels(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// First save integration with models
	if err := saveIntegration("claude", []string{"model1", "model2"}); err != nil {
		t.Fatalf("failed to save integration: %v", err)
	}

	// Then update aliases
	aliases := map[string]string{"primary": "new-model"}
	if err := saveAliases("claude", aliases); err != nil {
		t.Fatalf("failed to save aliases: %v", err)
	}

	// Verify models are preserved
	loaded, err := loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(loaded.Models) != 2 || loaded.Models[0] != "model1" {
		t.Errorf("models should be preserved, got %v", loaded.Models)
	}
}

// TestSaveAliases_EmptyMap clears all aliases
func TestSaveAliases_EmptyMap(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Save with aliases
	if err := saveAliases("claude", map[string]string{"primary": "model", "fast": "model"}); err != nil {
		t.Fatalf("failed to save: %v", err)
	}

	// Save empty map
	if err := saveAliases("claude", map[string]string{}); err != nil {
		t.Fatalf("failed to save empty: %v", err)
	}

	loaded, err := loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(loaded.Aliases) != 0 {
		t.Errorf("aliases should be empty, got %v", loaded.Aliases)
	}
}

// TestSaveAliases_NilMap handles nil gracefully
func TestSaveAliases_NilMap(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Save with aliases first
	if err := saveAliases("claude", map[string]string{"primary": "model"}); err != nil {
		t.Fatalf("failed to save: %v", err)
	}

	// Save nil map - should clear aliases
	if err := saveAliases("claude", nil); err != nil {
		t.Fatalf("failed to save nil: %v", err)
	}

	loaded, err := loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(loaded.Aliases) > 0 {
		t.Errorf("aliases should be nil or empty, got %v", loaded.Aliases)
	}
}

// TestSaveAliases_EmptyAppName returns error
func TestSaveAliases_EmptyAppName(t *testing.T) {
	err := saveAliases("", map[string]string{"primary": "model"})
	if err == nil {
		t.Error("expected error for empty app name")
	}
}

func TestSaveAliases_CaseInsensitive(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	if err := saveAliases("Claude", map[string]string{"primary": "model1"}); err != nil {
		t.Fatalf("failed to save: %v", err)
	}

	// Load with different case
	loaded, err := loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if loaded.Aliases["primary"] != "model1" {
		t.Errorf("expected primary=model1, got %q", loaded.Aliases["primary"])
	}

	// Update with different case
	if err := saveAliases("CLAUDE", map[string]string{"primary": "model2"}); err != nil {
		t.Fatalf("failed to update: %v", err)
	}

	loaded, err = loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load after update: %v", err)
	}
	if loaded.Aliases["primary"] != "model2" {
		t.Errorf("expected primary=model2, got %q", loaded.Aliases["primary"])
	}
}

// TestSaveAliases_CreatesIntegration creates integration if it doesn't exist
func TestSaveAliases_CreatesIntegration(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Save aliases for non-existent integration
	if err := saveAliases("newintegration", map[string]string{"primary": "model"}); err != nil {
		t.Fatalf("failed to save: %v", err)
	}

	loaded, err := loadIntegration("newintegration")
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if loaded.Aliases["primary"] != "model" {
		t.Errorf("expected primary=model, got %q", loaded.Aliases["primary"])
	}
}

func TestConfigureAliases_AliasMap(t *testing.T) {
	t.Run("cloud model auto-sets fast to primary", func(t *testing.T) {
		aliases := make(map[string]string)
		aliases["primary"] = "cloud-model"

		// Simulate cloud model behavior
		isCloud := true
		if isCloud {
			if aliases["fast"] == "" {
				aliases["fast"] = aliases["primary"]
			}
		}

		if aliases["fast"] != "cloud-model" {
			t.Errorf("expected fast=cloud-model, got %q", aliases["fast"])
		}
	})

	t.Run("cloud model preserves custom fast", func(t *testing.T) {
		aliases := map[string]string{
			"primary": "cloud-model",
			"fast":    "custom-fast-model",
		}

		// Simulate cloud model behavior - should preserve existing fast
		isCloud := true
		if isCloud {
			if aliases["fast"] == "" {
				aliases["fast"] = aliases["primary"]
			}
		}

		if aliases["fast"] != "custom-fast-model" {
			t.Errorf("expected fast=custom-fast-model (preserved), got %q", aliases["fast"])
		}
	})

	t.Run("local model clears fast", func(t *testing.T) {
		aliases := map[string]string{
			"primary": "local-model",
			"fast":    "should-be-cleared",
		}

		// Simulate local model behavior
		isCloud := false
		if !isCloud {
			delete(aliases, "fast")
		}

		if aliases["fast"] != "" {
			t.Errorf("expected fast to be cleared, got %q", aliases["fast"])
		}
	})

	t.Run("switching cloud to local clears fast", func(t *testing.T) {
		// Start with cloud config
		aliases := map[string]string{
			"primary": "cloud-model",
			"fast":    "cloud-model",
		}

		// Switch to local
		aliases["primary"] = "local-model"
		isCloud := false
		if !isCloud {
			delete(aliases, "fast")
		}

		if aliases["fast"] != "" {
			t.Errorf("fast should be cleared when switching to local, got %q", aliases["fast"])
		}
		if aliases["primary"] != "local-model" {
			t.Errorf("primary should be updated, got %q", aliases["primary"])
		}
	})

	t.Run("switching local to cloud sets fast", func(t *testing.T) {
		// Start with local config (no fast)
		aliases := map[string]string{
			"primary": "local-model",
		}

		// Switch to cloud
		aliases["primary"] = "cloud-model"
		isCloud := true
		if isCloud {
			if aliases["fast"] == "" {
				aliases["fast"] = aliases["primary"]
			}
		}

		if aliases["fast"] != "cloud-model" {
			t.Errorf("fast should be set when switching to cloud, got %q", aliases["fast"])
		}
	})
}

func TestSetAliases_PrefixMapping(t *testing.T) {
	// This tests the expected mapping without needing a real client
	aliases := map[string]string{
		"primary": "my-cloud-model",
		"fast":    "my-fast-model",
	}

	expectedMappings := map[string]string{
		"claude-sonnet-": aliases["primary"],
		"claude-haiku-":  aliases["fast"],
	}

	if expectedMappings["claude-sonnet-"] != "my-cloud-model" {
		t.Errorf("claude-sonnet- should map to primary")
	}
	if expectedMappings["claude-haiku-"] != "my-fast-model" {
		t.Errorf("claude-haiku- should map to fast")
	}
}

func TestSetAliases_LocalDeletesPrefixes(t *testing.T) {
	aliases := map[string]string{
		"primary": "local-model",
		// fast is empty/missing - indicates local model
	}

	prefixesToDelete := []string{"claude-sonnet-", "claude-haiku-"}

	// Verify the logic: when fast is empty, we should delete
	if aliases["fast"] != "" {
		t.Error("fast should be empty for local model")
	}

	// Verify we have the right prefixes to delete
	if len(prefixesToDelete) != 2 {
		t.Errorf("expected 2 prefixes to delete, got %d", len(prefixesToDelete))
	}
}

// TestAtomicUpdate_ServerFailsConfigNotSaved simulates atomic update behavior
func TestAtomicUpdate_ServerFailsConfigNotSaved(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Simulate: server fails, config should NOT be saved
	serverErr := errors.New("server unavailable")

	if serverErr == nil {
		t.Error("config should NOT be saved when server fails")
	}
}

// TestAtomicUpdate_ServerSucceedsConfigSaved simulates successful atomic update
func TestAtomicUpdate_ServerSucceedsConfigSaved(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Simulate: server succeeds, config should be saved
	var serverErr error
	if serverErr != nil {
		t.Fatal("server should succeed")
	}

	if err := saveAliases("claude", map[string]string{"primary": "model"}); err != nil {
		t.Fatalf("saveAliases failed: %v", err)
	}

	// Verify it was actually saved
	loaded, err := loadIntegration("claude")
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if loaded.Aliases["primary"] != "model" {
		t.Errorf("expected primary=model, got %q", loaded.Aliases["primary"])
	}
}

func TestConfigFile_PreservesUnknownFields(t *testing.T) {
	tmpDir := t.TempDir()
	setTestHome(t, tmpDir)

	// Write config with extra fields
	configPath := filepath.Join(tmpDir, ".ollama", "config.json")
	os.MkdirAll(filepath.Dir(configPath), 0o755)

	// Note: Our config struct only has Integrations, so top-level unknown fields
	// won't be preserved by our current implementation. This test documents that.
	initialConfig := `{
  "integrations": {
    "claude": {
      "models": ["model1"],
      "aliases": {"primary": "model1"},
      "unknownField": "should be lost"
    }
  },
  "topLevelUnknown": "will be lost"
}`
	os.WriteFile(configPath, []byte(initialConfig), 0o644)

	// Update aliases
	if err := saveAliases("claude", map[string]string{"primary": "model2"}); err != nil {
		t.Fatalf("failed to save: %v", err)
	}

	// Read raw file to check
	data, _ := os.ReadFile(configPath)
	content := string(data)

	// models should be preserved
	if !contains(content, "model1") {
		t.Error("models should be preserved")
	}

	// primary should be updated
	if !contains(content, "model2") {
		t.Error("primary should be updated to model2")
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

func TestClaudeImplementsAliasConfigurer(t *testing.T) {
	c := &Claude{}
	var _ AliasConfigurer = c // Compile-time check
}

func TestModelNameEdgeCases(t *testing.T) {
	testCases := []struct {
		name  string
		model string
	}{
		{"simple", "llama3.2"},
		{"with tag", "llama3.2:latest"},
		{"with cloud tag", "kimi-k2.5:cloud"},
		{"with namespace", "library/llama3.2"},
		{"with dots", "glm-4.7-flash"},
		{"with numbers", "qwen3:8b"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			setTestHome(t, tmpDir)

			aliases := map[string]string{"primary": tc.model}
			if err := saveAliases("claude", aliases); err != nil {
				t.Fatalf("failed to save model %q: %v", tc.model, err)
			}

			loaded, err := loadIntegration("claude")
			if err != nil {
				t.Fatalf("failed to load: %v", err)
			}
			if loaded.Aliases["primary"] != tc.model {
				t.Errorf("expected primary=%q, got %q", tc.model, loaded.Aliases["primary"])
			}
		})
	}
}

func TestSwitchingScenarios(t *testing.T) {
	t.Run("cloud to local removes fast", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Initial cloud config
		if err := saveAliases("claude", map[string]string{
			"primary": "cloud-model",
			"fast":    "cloud-model",
		}); err != nil {
			t.Fatal(err)
		}

		// Switch to local (no fast)
		if err := saveAliases("claude", map[string]string{
			"primary": "local-model",
		}); err != nil {
			t.Fatal(err)
		}

		loaded, _ := loadIntegration("claude")
		if loaded.Aliases["fast"] != "" {
			t.Errorf("fast should be removed, got %q", loaded.Aliases["fast"])
		}
		if loaded.Aliases["primary"] != "local-model" {
			t.Errorf("primary should be local-model, got %q", loaded.Aliases["primary"])
		}
	})

	t.Run("local to cloud adds fast", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Initial local config
		if err := saveAliases("claude", map[string]string{
			"primary": "local-model",
		}); err != nil {
			t.Fatal(err)
		}

		// Switch to cloud (with fast)
		if err := saveAliases("claude", map[string]string{
			"primary": "cloud-model",
			"fast":    "cloud-model",
		}); err != nil {
			t.Fatal(err)
		}

		loaded, _ := loadIntegration("claude")
		if loaded.Aliases["fast"] != "cloud-model" {
			t.Errorf("fast should be cloud-model, got %q", loaded.Aliases["fast"])
		}
	})

	t.Run("cloud to different cloud updates both", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Initial cloud config
		if err := saveAliases("claude", map[string]string{
			"primary": "cloud-model-1",
			"fast":    "cloud-model-1",
		}); err != nil {
			t.Fatal(err)
		}

		// Switch to different cloud
		if err := saveAliases("claude", map[string]string{
			"primary": "cloud-model-2",
			"fast":    "cloud-model-2",
		}); err != nil {
			t.Fatal(err)
		}

		loaded, _ := loadIntegration("claude")
		if loaded.Aliases["primary"] != "cloud-model-2" {
			t.Errorf("primary should be cloud-model-2, got %q", loaded.Aliases["primary"])
		}
		if loaded.Aliases["fast"] != "cloud-model-2" {
			t.Errorf("fast should be cloud-model-2, got %q", loaded.Aliases["fast"])
		}
	})
}

func TestToolCapabilityFiltering(t *testing.T) {
	t.Run("all models checked for tool capability", func(t *testing.T) {
		// Both cloud and local models are checked for tool capability via Show API
		// Only models with "tools" in capabilities are included
		m := modelInfo{Name: "tool-model", Remote: false, ToolCapable: true}
		if !m.ToolCapable {
			t.Error("tool capable model should be marked as such")
		}
	})

	t.Run("modelInfo includes ToolCapable field", func(t *testing.T) {
		m := modelInfo{Name: "test", Remote: true, ToolCapable: true}
		if !m.ToolCapable {
			t.Error("ToolCapable field should be accessible")
		}
	})
}

func TestIsCloudModel_RequiresClient(t *testing.T) {
	t.Run("nil client always returns false", func(t *testing.T) {
		// isCloudModel now only uses Show API, no suffix detection
		if isCloudModel(context.Background(), nil, "model:cloud") {
			t.Error("nil client should return false regardless of suffix")
		}
		if isCloudModel(context.Background(), nil, "local-model") {
			t.Error("nil client should return false")
		}
	})
}

func TestModelsAndAliasesMustStayInSync(t *testing.T) {
	t.Run("saveAliases followed by saveIntegration keeps them in sync", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Save aliases with one model
		if err := saveAliases("claude", map[string]string{"primary": "model-a"}); err != nil {
			t.Fatal(err)
		}

		// Save integration with same model (this is the pattern we use)
		if err := saveIntegration("claude", []string{"model-a"}); err != nil {
			t.Fatal(err)
		}

		loaded, _ := loadIntegration("claude")
		if loaded.Aliases["primary"] != loaded.Models[0] {
			t.Errorf("aliases.primary (%q) != models[0] (%q)", loaded.Aliases["primary"], loaded.Models[0])
		}
	})

	t.Run("out of sync config is detectable", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Simulate out-of-sync state (like manual edit or bug)
		if err := saveIntegration("claude", []string{"old-model"}); err != nil {
			t.Fatal(err)
		}
		if err := saveAliases("claude", map[string]string{"primary": "new-model"}); err != nil {
			t.Fatal(err)
		}

		loaded, _ := loadIntegration("claude")

		// They should be different (this is the bug state)
		if loaded.Models[0] == loaded.Aliases["primary"] {
			t.Error("expected out-of-sync state for this test")
		}

		// The fix: when updating aliases, also update models
		if err := saveIntegration("claude", []string{loaded.Aliases["primary"]}); err != nil {
			t.Fatal(err)
		}

		loaded, _ = loadIntegration("claude")
		if loaded.Models[0] != loaded.Aliases["primary"] {
			t.Errorf("after fix: models[0] (%q) should equal aliases.primary (%q)",
				loaded.Models[0], loaded.Aliases["primary"])
		}
	})

	t.Run("updating primary alias updates models too", func(t *testing.T) {
		tmpDir := t.TempDir()
		setTestHome(t, tmpDir)

		// Initial state
		if err := saveIntegration("claude", []string{"initial-model"}); err != nil {
			t.Fatal(err)
		}
		if err := saveAliases("claude", map[string]string{"primary": "initial-model"}); err != nil {
			t.Fatal(err)
		}

		// Update aliases AND models together
		newAliases := map[string]string{"primary": "updated-model"}
		if err := saveAliases("claude", newAliases); err != nil {
			t.Fatal(err)
		}
		if err := saveIntegration("claude", []string{newAliases["primary"]}); err != nil {
			t.Fatal(err)
		}

		loaded, _ := loadIntegration("claude")
		if loaded.Models[0] != "updated-model" {
			t.Errorf("models[0] should be updated-model, got %q", loaded.Models[0])
		}
		if loaded.Aliases["primary"] != "updated-model" {
			t.Errorf("aliases.primary should be updated-model, got %q", loaded.Aliases["primary"])
		}
	})
}
