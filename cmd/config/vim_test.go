package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestVim_detectVim(t *testing.T) {
	v := &Vim{}
	
	// Test vim detection - should find vim in PATH if available
	path, err := v.detectVim()
	
	// This test may fail in CI if vim isn't installed
	// Skip if vim not available
	if err != nil {
		t.Skipf("vim not available in PATH: %v", err)
	}
	
	if path == "" {
		t.Error("detectVim returned empty path without error")
	}
	
	t.Logf("Found vim at: %s", path)
}

func TestVim_writeConfig(t *testing.T) {
	// Create temp home dir
	tempHome := t.TempDir()
	originalHome := os.Getenv("HOME")
	os.Setenv("HOME", tempHome)
	defer os.Setenv("HOME", originalHome)
	
	v := &Vim{
		model: "codellama:7b",
		host:  "http://localhost:11434",
	}
	
	err := v.writeConfig()
	if err != nil {
		t.Fatalf("writeConfig failed: %v", err)
	}
	
	// Verify config file was created
	configPath := filepath.Join(tempHome, ".vim", "config", "ollama.vim")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Errorf("Config file not created at %s", configPath)
	}
	
	// Verify plugin file was created
	pluginPath := filepath.Join(tempHome, ".vim", "plugin", "ollama-launch.vim")
	if _, err := os.Stat(pluginPath); os.IsNotExist(err) {
		t.Errorf("Plugin file not created at %s", pluginPath)
	}
	
	// Verify config content
	content, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read config: %v", err)
	}
	
	configStr := string(content)
	if !contains(configStr, "codellama:7b") {
		t.Error("Config doesn't contain model name")
	}
	if !contains(configStr, "http://localhost:11434") {
		t.Error("Config doesn't contain host")
	}
	if !contains(configStr, "g:ollama_enabled = 1") {
		t.Error("Config doesn't enable ollama")
	}
	
	t.Log("Config files created successfully")
}

func TestVim_backupIfExists(t *testing.T) {
	tempHome := t.TempDir()
	testFile := filepath.Join(tempHome, "test.txt")
	
	// Create test file
	originalContent := []byte("original content")
	if err := os.WriteFile(testFile, originalContent, 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	
	v := &Vim{}
	
	// Should create backup
	err := v.backupIfExists(testFile)
	if err != nil {
		t.Fatalf("backupIfExists failed: %v", err)
	}
	
	// Check backup was created
	entries, err := os.ReadDir(tempHome)
	if err != nil {
		t.Fatalf("Failed to read dir: %v", err)
	}
	
	foundBackup := false
	for _, entry := range entries {
		if contains(entry.Name(), "test.txt.bak.") {
			foundBackup = true
			break
		}
	}
	
	if !foundBackup {
		t.Error("Backup file was not created")
	}
	
	t.Log("Backup creation works")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsAt(s, substr, 0))
}

func containsAt(s, substr string, start int) bool {
	for i := start; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
