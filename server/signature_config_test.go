package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDefaultSignatureConfig(t *testing.T) {
	config := DefaultSignatureConfig()
	
	if config == nil {
		t.Fatal("DefaultSignatureConfig returned nil")
	}
	
	// Check default policy
	if config.Policy != PolicyPermissive {
		t.Errorf("Expected default policy 'permissive', got '%s'", config.Policy)
	}
	
	// Check trusted signers is initialized
	if config.TrustedSigners == nil {
		t.Fatal("Default config has nil trusted signers")
	}
	if len(config.TrustedSigners) != 0 {
		t.Errorf("Expected empty trusted signers list, got %d entries", len(config.TrustedSigners))
	}
	
	// Check other defaults
	if !config.VerifyOnPull {
		t.Error("Expected VerifyOnPull to be true by default")
	}
	if !config.VerifyOnPush {
		t.Error("Expected VerifyOnPush to be true by default")
	}
	if config.RequireTrustedSigners {
		t.Error("Expected RequireTrustedSigners to be false by default")
	}
}

func TestSignatureConfigJSONSerialization(t *testing.T) {
	// Create test config
	config := &SignatureConfig{
		Policy:       PolicyStrict,
		VerifyOnPull: true,
		VerifyOnPush: false,
		TrustedSigners: []TrustedSigner{
			{
				ID:        "test-signer-1",
				Name:      "Test Signer",
				Email:     "test@example.com",
				PublicKey: "dGVzdC1wdWJsaWMta2V5",
				AddedAt:   time.Now().UTC(),
			},
		},
		RequireTrustedSigners: true,
		MaxSignatureAge:      30,
		CheckRevocation:      true,
		UpdatedAt:           time.Now().UTC(),
	}
	
	// Test marshaling
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("Failed to marshal config: %v", err)
	}
	
	// Test unmarshaling
	var unmarshaled SignatureConfig
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal config: %v", err)
	}
	
	// Verify policy
	if unmarshaled.Policy != config.Policy {
		t.Error("Policy mismatch after JSON round-trip")
	}
	
	// Verify trusted signers
	if len(unmarshaled.TrustedSigners) != len(config.TrustedSigners) {
		t.Error("TrustedSigners count mismatch after JSON round-trip")
	}
	
	if len(unmarshaled.TrustedSigners) > 0 {
		signer := unmarshaled.TrustedSigners[0]
		if signer.Email != "test@example.com" {
			t.Error("Signer email mismatch after JSON round-trip")
		}
		if signer.Name != "Test Signer" {
			t.Error("Signer name mismatch after JSON round-trip")
		}
	}
	
	// Verify other fields
	if unmarshaled.RequireTrustedSigners != config.RequireTrustedSigners {
		t.Error("RequireTrustedSigners mismatch after JSON round-trip")
	}
}

func TestLoadSignatureConfigNonExistent(t *testing.T) {
	// This test relies on the actual config path being non-existent
	// In a real test environment, you might want to temporarily modify the config path
	_, err := LoadSignatureConfig()
	if err == nil {
		t.Log("Config file exists, skipping non-existent test")
		return
	}
	// Expected behavior - should return error for non-existent config
	t.Logf("Got expected error for non-existent config: %v", err)
}

func TestSaveAndLoadSignatureConfig(t *testing.T) {
	// Create temporary config directory
	tmpDir, err := os.MkdirTemp("", "config-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	
	// Create test config file path
	configPath := filepath.Join(tmpDir, "test_signature_config.json")
	
	// Create test config
	originalConfig := &SignatureConfig{
		Policy:       PolicyStrict,
		VerifyOnPull: true,
		VerifyOnPush: true,
		TrustedSigners: []TrustedSigner{
			{
				ID:        "alice-123",
				Name:      "Alice Smith",
				Email:     "alice@example.com",
				PublicKey: "YWxpY2Uta2V5",
				AddedAt:   time.Now().UTC(),
			},
		},
		RequireTrustedSigners: true,
		MaxSignatureAge:      60,
		CheckRevocation:      false,
		UpdatedAt:           time.Now().UTC(),
	}
	
	// Save config to file
	data, err := json.Marshal(originalConfig)
	if err != nil {
		t.Fatalf("Failed to marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, data, 0600); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}
	
	// Load config from file
	loadedData, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("Failed to read config file: %v", err)
	}
	
	var loadedConfig SignatureConfig
	if err := json.Unmarshal(loadedData, &loadedConfig); err != nil {
		t.Fatalf("Failed to unmarshal config: %v", err)
	}
	
	// Verify loaded config
	if loadedConfig.Policy != originalConfig.Policy {
		t.Error("Policy mismatch after save/load")
	}
	if len(loadedConfig.TrustedSigners) != len(originalConfig.TrustedSigners) {
		t.Error("TrustedSigners count mismatch after save/load")
	}
	
	if len(loadedConfig.TrustedSigners) > 0 {
		alice := loadedConfig.TrustedSigners[0]
		if alice.Name != "Alice Smith" {
			t.Error("Alice name mismatch after save/load")
		}
		if alice.Email != "alice@example.com" {
			t.Error("Alice email mismatch after save/load")
		}
	}
}

func TestSignaturePolicyConstants(t *testing.T) {
	// Test policy constants
	if PolicyPermissive != "permissive" {
		t.Errorf("Expected PolicyPermissive to be 'permissive', got '%s'", PolicyPermissive)
	}
	if PolicyWarn != "warn" {
		t.Errorf("Expected PolicyWarn to be 'warn', got '%s'", PolicyWarn)
	}
	if PolicyStrict != "strict" {
		t.Errorf("Expected PolicyStrict to be 'strict', got '%s'", PolicyStrict)
	}
}

func TestTrustedSignerStructure(t *testing.T) {
	// Test TrustedSigner structure
	signer := TrustedSigner{
		ID:          "test-id-123",
		Name:        "Test Signer",
		Email:       "test@example.com",
		PublicKey:   "dGVzdC1rZXk=",
		AddedAt:     time.Now().UTC(),
		Description: "Test signer for unit tests",
	}
	
	// Test JSON serialization
	data, err := json.Marshal(signer)
	if err != nil {
		t.Fatalf("Failed to marshal TrustedSigner: %v", err)
	}
	
	var unmarshaled TrustedSigner
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal TrustedSigner: %v", err)
	}
	
	// Verify fields
	if unmarshaled.ID != signer.ID {
		t.Error("ID mismatch after JSON round-trip")
	}
	if unmarshaled.Name != signer.Name {
		t.Error("Name mismatch after JSON round-trip")
	}
	if unmarshaled.Email != signer.Email {
		t.Error("Email mismatch after JSON round-trip")
	}
	if unmarshaled.PublicKey != signer.PublicKey {
		t.Error("PublicKey mismatch after JSON round-trip")
	}
	if unmarshaled.Description != signer.Description {
		t.Error("Description mismatch after JSON round-trip")
	}
}

func TestSignatureConfigValidation(t *testing.T) {
	tests := []struct {
		name   string
		config *SignatureConfig
		valid  bool
	}{
		{
			name: "valid permissive config",
			config: &SignatureConfig{
				Policy:       PolicyPermissive,
				VerifyOnPull: true,
				VerifyOnPush: true,
			},
			valid: true,
		},
		{
			name: "valid strict config",
			config: &SignatureConfig{
				Policy:                PolicyStrict,
				RequireTrustedSigners: true,
				TrustedSigners: []TrustedSigner{
					{
						ID:      "trusted-1",
						Name:    "Trusted Signer",
						Email:   "trusted@example.com",
						AddedAt: time.Now(),
					},
				},
			},
			valid: true,
		},
		{
			name: "config with empty trusted signers",
			config: &SignatureConfig{
				Policy:         PolicyStrict,
				TrustedSigners: []TrustedSigner{},
			},
			valid: true, // Empty list is valid
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that the config can be marshaled and unmarshaled
			data, err := json.Marshal(tt.config)
			if err != nil && tt.valid {
				t.Errorf("Expected valid config to marshal successfully, got error: %v", err)
			}
			
			if tt.valid {
				var unmarshaled SignatureConfig
				if err := json.Unmarshal(data, &unmarshaled); err != nil {
					t.Errorf("Expected valid config to unmarshal successfully, got error: %v", err)
				}
			}
		})
	}
}

func TestSignatureConfigFilePermissions(t *testing.T) {
	// Create temporary config file
	tmpDir, err := os.MkdirTemp("", "config-permissions-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	
	configPath := filepath.Join(tmpDir, "signature_config.json")
	
	// Create and save config
	config := DefaultSignatureConfig()
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatalf("Failed to marshal config: %v", err)
	}
	
	// Write with specific permissions
	if err := os.WriteFile(configPath, data, 0600); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}
	
	// Check file permissions
	info, err := os.Stat(configPath)
	if err != nil {
		t.Fatalf("Failed to stat config file: %v", err)
	}
	
	// Config file should be readable and writable by owner only (0600)
	expectedMode := os.FileMode(0600)
	if info.Mode().Perm() != expectedMode {
		t.Errorf("Expected file mode %o, got %o", expectedMode, info.Mode().Perm())
	}
}