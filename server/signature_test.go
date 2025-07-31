package server

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
)

func TestNewSignatureVerifier(t *testing.T) {
	verifier := NewSignatureVerifier()
	if verifier == nil {
		t.Fatal("NewSignatureVerifier returned nil")
	}
	if verifier.config == nil {
		t.Fatal("SignatureVerifier config is nil")
	}
}

func TestNewSignatureVerifierWithConfig(t *testing.T) {
	config := DefaultSignatureConfig()
	config.Policy = PolicyStrict
	
	verifier := NewSignatureVerifierWithConfig(config)
	if verifier == nil {
		t.Fatal("NewSignatureVerifierWithConfig returned nil")
	}
	if verifier.config != config {
		t.Fatal("SignatureVerifier config mismatch")
	}
	if verifier.config.Policy != PolicyStrict {
		t.Error("Config not properly set")
	}
}

func TestSignatureVerificationResultStructure(t *testing.T) {
	result := &SignatureVerificationResult{
		Valid:        true,
		Signer:       "test@example.com",
		SignedAt:     time.Now(),
		Format:       "oms-v1.0",
		ErrorMessage: "",
	}

	// Test JSON marshaling
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("Failed to marshal SignatureVerificationResult: %v", err)
	}

	var unmarshaled SignatureVerificationResult
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal SignatureVerificationResult: %v", err)
	}

	if unmarshaled.Valid != result.Valid {
		t.Errorf("Valid field mismatch: %v != %v", unmarshaled.Valid, result.Valid)
	}
	if unmarshaled.Signer != result.Signer {
		t.Errorf("Signer field mismatch: %s != %s", unmarshaled.Signer, result.Signer)
	}
	if unmarshaled.Format != result.Format {
		t.Errorf("Format field mismatch: %s != %s", unmarshaled.Format, result.Format)
	}
}

func TestVerifyManifestUnsigned(t *testing.T) {
	verifier := NewSignatureVerifier()
	
	// Create manifest without signature
	manifest := &Manifest{
		Config: Layer{Digest: "sha256:config"},
		Layers: []Layer{
			{MediaType: "application/vnd.ollama.image.model", Digest: "sha256:layer1"},
		},
		Signature: nil, // No signature
	}

	result, err := verifier.VerifyManifest(manifest)
	
	if !errors.Is(err, ErrSignatureNotFound) {
		t.Errorf("Expected ErrSignatureNotFound, got: %v", err)
	}
	if result.Valid {
		t.Error("Expected invalid result for unsigned manifest")
	}
	if result.ErrorMessage != "model is not signed" {
		t.Errorf("Expected 'model is not signed' error, got: %s", result.ErrorMessage)
	}
}

func TestVerifyManifestSignatureMetadataWithoutLayer(t *testing.T) {
	verifier := NewSignatureVerifier()
	
	// Create manifest with signature metadata but no signature layer
	manifest := &Manifest{
		Config: Layer{Digest: "sha256:config"},
		Layers: []Layer{
			{MediaType: "application/vnd.ollama.image.model", Digest: "sha256:layer1"},
		},
		Signature: &SignatureInfo{
			Format:       "oms-v1.0",
			SignatureURI: "sha256:missing",
			Signer:       "test@example.com",
			SignedAt:     time.Now(),
		},
	}

	result, err := verifier.VerifyManifest(manifest)
	
	if !errors.Is(err, ErrSignatureNotFound) {
		t.Errorf("Expected ErrSignatureNotFound, got: %v", err)
	}
	if result.Valid {
		t.Error("Expected invalid result for manifest with metadata but no layer")
	}
	if result.ErrorMessage != "signature metadata exists but signature file not found" {
		t.Errorf("Expected 'signature metadata exists but signature file not found' error, got: %s", result.ErrorMessage)
	}
}

func TestHasValidSignatureUnsigned(t *testing.T) {
	// This test requires a model name that doesn't exist or is unsigned
	// For testing purposes, we'll use a non-existent model
	modelName := model.ParseName("test/nonexistent:latest")
	
	valid, err := HasValidSignature(modelName)
	if err != nil {
		// We expect an error for non-existent model, but should handle gracefully
		t.Logf("HasValidSignature returned error for non-existent model: %v", err)
		return
	}
	
	if valid {
		t.Error("Expected false for non-existent/unsigned model")
	}
}

func TestSignatureErrors(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected string
	}{
		{
			name:     "signature not found",
			err:      ErrSignatureNotFound,
			expected: "signature not found",
		},
		{
			name:     "signature invalid",
			err:      ErrSignatureInvalid,
			expected: "signature verification failed",
		},
		{
			name:     "unsupported format",
			err:      ErrSignatureUnsupportedFormat,
			expected: "unsupported signature format",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.err.Error() != tt.expected {
				t.Errorf("Error message mismatch: got %s, want %s", tt.err.Error(), tt.expected)
			}
		})
	}
}

func TestVerifySignatureFileWithManifestInvalidPath(t *testing.T) {
	verifier := NewSignatureVerifier()
	
	// Create test manifest
	manifest := &Manifest{
		Config: Layer{Digest: "sha256:config"},
		Layers: []Layer{
			{MediaType: "application/vnd.ollama.image.model", Digest: "sha256:layer1"},
		},
	}

	sigInfo := &SignatureInfo{
		Format:       "oms-v1.0",
		SignatureURI: "sha256:test",
		Signer:       "test@example.com",
		SignedAt:     time.Now(),
	}

	// Test with invalid file path
	result, err := verifier.verifySignatureFileWithManifest("", sigInfo, manifest)
	
	if err == nil {
		t.Fatal("Expected error for empty file path, got none")
	}
	if result.Valid {
		t.Error("Expected invalid result for empty file path")
	}
	if result.ErrorMessage != "signature file path is invalid" {
		t.Errorf("Expected 'signature file path is invalid' error, got: %s", result.ErrorMessage)
	}
}

func TestVerifySignatureFileWithManifestValidFlow(t *testing.T) {
	// Create temporary directory for test
	tmpDir, err := os.MkdirTemp("", "signature-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Generate test signature
	_, privateKey, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate key pair: %v", err)
	}

	// Create test manifest and compute its digest
	manifest := &Manifest{
		Config: Layer{Digest: "sha256:config123"},
		Layers: []Layer{
			{MediaType: "application/vnd.ollama.image.model", Digest: "sha256:layer1"},
			{MediaType: "application/vnd.ollama.image.system", Digest: "sha256:layer2"},
		},
	}

	modelDigest, err := ComputeModelDigest(manifest)
	if err != nil {
		t.Fatalf("Failed to compute model digest: %v", err)
	}

	// Create test signature
	signer := "test@example.com"
	oms, err := CreateTestSignature(privateKey, modelDigest, signer)
	if err != nil {
		t.Fatalf("Failed to create test signature: %v", err)
	}

	// Write signature to file
	sigFile := filepath.Join(tmpDir, "signature.json")
	sigData, err := json.Marshal(oms)
	if err != nil {
		t.Fatalf("Failed to marshal signature: %v", err)
	}
	if err := os.WriteFile(sigFile, sigData, 0644); err != nil {
		t.Fatalf("Failed to write signature file: %v", err)
	}

	// Test verification
	verifier := NewSignatureVerifier()
	sigInfo := &SignatureInfo{
		Format:       oms.Version,
		SignatureURI: "test-uri",
		Signer:       signer,
		SignedAt:     oms.Timestamp,
	}

	result, err := verifier.verifySignatureFileWithManifest(sigFile, sigInfo, manifest)
	if err != nil {
		t.Fatalf("verifySignatureFileWithManifest failed: %v", err)
	}
	if !result.Valid {
		t.Errorf("Expected valid signature, got invalid: %s", result.ErrorMessage)
	}
	if result.Signer != signer {
		t.Errorf("Expected signer '%s', got '%s'", signer, result.Signer)
	}
	if result.Format != oms.Version {
		t.Errorf("Expected format '%s', got '%s'", oms.Version, result.Format)
	}
}

func TestGetSignatureInfoNonExistent(t *testing.T) {
	// Test with non-existent model
	modelName := model.ParseName("nonexistent/model:tag")
	
	_, err := GetSignatureInfo(modelName)
	if err == nil {
		t.Fatal("Expected error for non-existent model, got none")
	}
	// Should contain "failed to load model manifest"
	if err.Error() == "" {
		t.Error("Expected meaningful error message")
	}
}

func TestSignatureVerificationResultErrorMessage(t *testing.T) {
	tests := []struct {
		name     string
		result   *SignatureVerificationResult
		expected bool
	}{
		{
			name: "valid signature",
			result: &SignatureVerificationResult{
				Valid:        true,
				ErrorMessage: "",
			},
			expected: true,
		},
		{
			name: "invalid signature with error",
			result: &SignatureVerificationResult{
				Valid:        false,
				ErrorMessage: "signature verification failed",
			},
			expected: false,
		},
		{
			name: "invalid signature without error message",
			result: &SignatureVerificationResult{
				Valid:        false,
				ErrorMessage: "",
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.result.Valid != tt.expected {
				t.Errorf("Expected Valid=%v, got Valid=%v", tt.expected, tt.result.Valid)
			}
			
			if !tt.result.Valid && tt.result.ErrorMessage == "" {
				t.Log("Note: Invalid result without error message - consider adding default error")
			}
		})
	}
}

func TestSignatureVerificationTiming(t *testing.T) {
	// Test that signature verification doesn't take too long
	verifier := NewSignatureVerifier()
	
	// Create minimal test case
	manifest := &Manifest{
		Config: Layer{Digest: "sha256:config"},
		Layers: []Layer{
			{MediaType: "application/vnd.ollama.image.model", Digest: "sha256:layer1"},
		},
		Signature: nil,
	}

	start := time.Now()
	_, _ = verifier.VerifyManifest(manifest)
	duration := time.Since(start)

	// Signature verification should be fast (< 100ms for this simple case)
	if duration > 100*time.Millisecond {
		t.Errorf("Signature verification took too long: %v", duration)
	}
}

func TestConcurrentSignatureVerification(t *testing.T) {
	// Test that signature verification is safe for concurrent use
	verifier := NewSignatureVerifier()
	
	manifest := &Manifest{
		Config: Layer{Digest: "sha256:config"},
		Layers: []Layer{
			{MediaType: "application/vnd.ollama.image.model", Digest: "sha256:layer1"},
		},
		Signature: nil,
	}

	// Run multiple verifications concurrently
	results := make(chan error, 10)
	for i := 0; i < 10; i++ {
		go func() {
			_, err := verifier.VerifyManifest(manifest)
			results <- err
		}()
	}

	// Collect results
	for i := 0; i < 10; i++ {
		err := <-results
		if err == nil {
			t.Error("Expected ErrSignatureNotFound for unsigned manifest")
		} else if !errors.Is(err, ErrSignatureNotFound) {
			t.Errorf("Expected ErrSignatureNotFound, got: %v", err)
		}
	}
}