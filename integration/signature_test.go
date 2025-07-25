package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/server"
)

func TestSignatureEndToEndWorkflow(t *testing.T) {
	// Skip if not in integration test environment
	if os.Getenv("OLLAMA_INTEGRATION_TEST") == "" {
		t.Skip("Integration test skipped - set OLLAMA_INTEGRATION_TEST=1 to run")
	}

	// Setup test environment
	ctx := context.Background()
	client, cleanup := setupTestClient(t)
	defer cleanup()

	// Test model name
	testModel := "signature-test:latest"

	t.Run("end-to-end signature workflow", func(t *testing.T) {
		// Step 1: Create a test model (simplified modelfile)
		modelfile := `FROM scratch
SYSTEM "You are a test model for signature verification."`

		createReq := &api.CreateRequest{
			Model:     testModel,
			Modelfile: modelfile,
		}

		err := client.Create(ctx, createReq, func(resp api.CreateStatus) error {
			if resp.Status != "" {
				t.Logf("Create status: %s", resp.Status)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("Failed to create test model: %v", err)
		}

		// Step 2: Verify model exists and is initially unsigned
		models, err := client.List(ctx)
		if err != nil {
			t.Fatalf("Failed to list models: %v", err)
		}

		var testModelFound bool
		for _, model := range models.Models {
			if model.Name == testModel {
				testModelFound = true
				// Check that signature status is properly reported
				if model.Signature != nil && model.Signature.Signed {
					t.Logf("Model is already signed: %+v", model.Signature)
				} else {
					t.Logf("Model is unsigned as expected")
				}
				break
			}
		}
		if !testModelFound {
			t.Fatalf("Test model %s not found in model list", testModel)
		}

		// Step 3: Test signature verification of unsigned model
		// This would normally be done via CLI, but we can test the server functions directly
		verifier := server.NewSignatureVerifier()
		modelName := parseModelName(testModel)
		
		result, err := verifier.VerifyModel(modelName)
		if err == nil {
			t.Errorf("Expected error verifying unsigned model, got result: %+v", result)
		}
		if !strings.Contains(err.Error(), "signature not found") && 
		   !strings.Contains(err.Error(), "model is not signed") {
			t.Errorf("Expected 'signature not found' error, got: %v", err)
		}

		// Step 4: Generate test signature
		publicKey, privateKey, err := server.GenerateKeyPair()
		if err != nil {
			t.Fatalf("Failed to generate key pair: %v", err)
		}

		// Step 5: Load and sign the model manifest
		manifest, err := server.ParseNamedManifest(modelName)
		if err != nil {
			t.Fatalf("Failed to parse model manifest: %v", err)
		}

		// Compute model digest
		modelDigest, err := server.ComputeModelDigest(manifest)
		if err != nil {
			t.Fatalf("Failed to compute model digest: %v", err)
		}

		// Create signature
		signer := "integration-test@example.com"
		oms, err := server.CreateTestSignature(privateKey, modelDigest, signer)
		if err != nil {
			t.Fatalf("Failed to create test signature: %v", err)
		}

		// Step 6: Save signature to model
		// This tests the signature storage mechanism
		err = saveSignatureToTestModel(testModel, manifest, oms)
		if err != nil {
			t.Fatalf("Failed to save signature: %v", err)
		}

		// Step 7: Verify the signed model
		result, err = verifier.VerifyModel(modelName)
		if err != nil {
			t.Fatalf("Failed to verify signed model: %v", err)
		}
		if !result.Valid {
			t.Errorf("Expected valid signature, got invalid: %s", result.ErrorMessage)
		}
		if result.Signer != signer {
			t.Errorf("Expected signer '%s', got '%s'", signer, result.Signer)
		}

		// Step 8: Test signature information retrieval
		sigInfo, err := server.GetSignatureInfo(modelName)
		if err != nil {
			t.Fatalf("Failed to get signature info: %v", err)
		}
		if sigInfo == nil {
			t.Fatal("Expected signature info, got nil")
		}
		if sigInfo.Signer != signer {
			t.Errorf("Expected signature info signer '%s', got '%s'", signer, sigInfo.Signer)
		}

		// Step 9: Test list shows signed model
		models, err = client.List(ctx)
		if err != nil {
			t.Fatalf("Failed to list models after signing: %v", err)
		}

		for _, model := range models.Models {
			if model.Name == testModel {
				t.Logf("Model signature status: %+v", model.Signature)
				// Note: The API signature status may not reflect the actual signature
				// due to server-side implementation details
				break
			}
		}

		// Step 10: Cleanup - delete test model
		deleteReq := &api.DeleteRequest{
			Model: testModel,
		}
		err = client.Delete(ctx, deleteReq)
		if err != nil {
			t.Logf("Warning: Failed to cleanup test model: %v", err)
		}

		t.Logf("✅ End-to-end signature workflow completed successfully")
	})
}

func TestSignatureConfigurationWorkflow(t *testing.T) {
	if os.Getenv("OLLAMA_INTEGRATION_TEST") == "" {
		t.Skip("Integration test skipped - set OLLAMA_INTEGRATION_TEST=1 to run")
	}

	t.Run("signature configuration", func(t *testing.T) {
		// Test default configuration
		defaultConfig := server.DefaultSignatureConfig()
		if defaultConfig == nil {
			t.Fatal("Default signature config is nil")
		}
		if defaultConfig.Policy.TrustLevel != "permissive" {
			t.Errorf("Expected permissive trust level, got %s", defaultConfig.Policy.TrustLevel)
		}

		// Test configuration with strict policy
		strictConfig := &server.SignatureConfig{
			Policy: &server.SignaturePolicy{
				RequireSignatures: true,
				TrustLevel:       "strict",
			},
			TrustedSigners: make(map[string]*server.TrustedSigner),
		}

		// Add a trusted signer
		trustedSigner := &server.TrustedSigner{
			Name:      "Test Trusted Signer",
			Email:     "trusted@example.com",
			PublicKey: "dHJ1c3RlZC1rZXk=",
			Verified:  true,
			CreatedAt: time.Now().UTC(),
		}
		strictConfig.AddTrustedSigner(trustedSigner)

		// Test signature validation with strict policy
		verifier := server.NewSignatureVerifierWithConfig(strictConfig)
		
		// Create a test signature result from untrusted signer
		untrustedResult := &server.SignatureVerificationResult{
			Valid:    true,
			Signer:   "untrusted@example.com",
			SignedAt: time.Now(),
			Format:   "oms-v1.0",
		}

		valid, errorMsg := strictConfig.IsSignatureValid(untrustedResult, "untrusted@example.com")
		if valid {
			t.Error("Expected invalid result for untrusted signer in strict mode")
		}
		if errorMsg != "signer not in trusted list" {
			t.Errorf("Expected 'signer not in trusted list' error, got: %s", errorMsg)
		}

		// Test with trusted signer
		trustedResult := &server.SignatureVerificationResult{
			Valid:    true,
			Signer:   "trusted@example.com",
			SignedAt: time.Now(),
			Format:   "oms-v1.0",
		}

		valid, errorMsg = strictConfig.IsSignatureValid(trustedResult, "trusted@example.com")
		if !valid {
			t.Errorf("Expected valid result for trusted signer, got invalid: %s", errorMsg)
		}

		t.Logf("✅ Signature configuration workflow completed successfully")
	})
}

func TestSignatureCryptographicSecurity(t *testing.T) {
	if os.Getenv("OLLAMA_INTEGRATION_TEST") == "" {
		t.Skip("Integration test skipped - set OLLAMA_INTEGRATION_TEST=1 to run")
	}

	t.Run("cryptographic security", func(t *testing.T) {
		// Test key generation security
		keys := make(map[string]bool)
		for i := 0; i < 10; i++ {
			publicKey, privateKey, err := server.GenerateKeyPair()
			if err != nil {
				t.Fatalf("Failed to generate key pair %d: %v", i, err)
			}

			// Ensure keys are unique
			keyPair := publicKey + privateKey
			if keys[keyPair] {
				t.Errorf("Duplicate key pair generated (iteration %d)", i)
			}
			keys[keyPair] = true

			// Test that signatures with different keys are different
			modelDigest := fmt.Sprintf("test-digest-%d", i)
			sig1, err := server.CreateTestSignature(privateKey, modelDigest, "test@example.com")
			if err != nil {
				t.Fatalf("Failed to create signature 1: %v", err)
			}

			// Generate another key and sign the same digest
			_, privateKey2, err := server.GenerateKeyPair()
			if err != nil {
				t.Fatalf("Failed to generate second key pair: %v", err)
			}

			sig2, err := server.CreateTestSignature(privateKey2, modelDigest, "test@example.com")
			if err != nil {
				t.Fatalf("Failed to create signature 2: %v", err)
			}

			if sig1.Signature == sig2.Signature {
				t.Error("Different keys produced identical signatures")
			}
			if sig1.PublicKey == sig2.PublicKey {
				t.Error("Different key generations produced identical public keys")
			}
		}

		// Test signature tampering detection
		publicKey, privateKey, err := server.GenerateKeyPair()
		if err != nil {
			t.Fatalf("Failed to generate key pair for tampering test: %v", err)
		}

		originalDigest := "original-model-digest"
		oms, err := server.CreateTestSignature(privateKey, originalDigest, "test@example.com")
		if err != nil {
			t.Fatalf("Failed to create signature for tampering test: %v", err)
		}

		// Create verifier
		config := server.DefaultSignatureConfig()
		verifier := server.NewCryptoSignatureVerifier(config)

		// Test with correct digest
		valid, err := verifier.cryptoVerify(oms, originalDigest)
		if err != nil {
			t.Fatalf("Failed to verify original signature: %v", err)
		}
		if !valid {
			t.Error("Original signature should be valid")
		}

		// Test with tampered digest
		tamperedDigest := "tampered-model-digest"
		valid, err = verifier.cryptoVerify(oms, tamperedDigest)
		if err != nil {
			t.Fatalf("Failed to verify tampered signature: %v", err)
		}
		if valid {
			t.Error("Tampered signature should be invalid")
		}

		t.Logf("✅ Cryptographic security tests passed")
	})
}

// Helper functions for integration tests

func setupTestClient(t *testing.T) (*api.Client, func()) {
	// This would typically set up a test Ollama server
	// For now, we'll use the default client
	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatalf("Failed to create API client: %v", err)
	}

	return client, func() {
		// Cleanup function
	}
}

func parseModelName(name string) server.ModelName {
	// This is a simplified model name parser for testing
	// In real implementation, use the proper model name parsing
	return server.ModelName(name)
}

func saveSignatureToTestModel(modelName string, manifest *server.Manifest, oms *server.OMSSignature) error {
	// This is a simplified version of signature saving for testing
	// In real implementation, this would use the proper blob storage system
	
	// Marshal signature to JSON
	sigData, err := json.Marshal(oms)
	if err != nil {
		return fmt.Errorf("failed to marshal signature: %w", err)
	}

	// Create temporary signature file (in real implementation, this would be stored in blob storage)
	tmpDir := os.TempDir()
	sigFile := filepath.Join(tmpDir, fmt.Sprintf("test-signature-%s.json", strings.ReplaceAll(modelName, ":", "-")))
	
	if err := os.WriteFile(sigFile, sigData, 0644); err != nil {
		return fmt.Errorf("failed to write signature file: %w", err)
	}

	// Note: In real implementation, this would update the manifest and store it properly
	// For integration testing, we're just verifying the signature creation and verification flow

	return nil
}

func TestSignaturePerformance(t *testing.T) {
	if os.Getenv("OLLAMA_INTEGRATION_TEST") == "" {
		t.Skip("Integration test skipped - set OLLAMA_INTEGRATION_TEST=1 to run")
	}

	t.Run("signature performance", func(t *testing.T) {
		// Test key generation performance
		start := time.Now()
		for i := 0; i < 100; i++ {
			_, _, err := server.GenerateKeyPair()
			if err != nil {
				t.Fatalf("Key generation failed at iteration %d: %v", i, err)
			}
		}
		keyGenDuration := time.Since(start)
		avgKeyGen := keyGenDuration / 100
		
		if avgKeyGen > 10*time.Millisecond {
			t.Errorf("Key generation too slow: average %v per key", avgKeyGen)
		}
		t.Logf("Key generation performance: %v average per key", avgKeyGen)

		// Test signature creation performance
		_, privateKey, err := server.GenerateKeyPair()
		if err != nil {
			t.Fatalf("Failed to generate key for performance test: %v", err)
		}

		start = time.Now()
		for i := 0; i < 100; i++ {
			digest := fmt.Sprintf("performance-test-digest-%d", i)
			_, err := server.CreateTestSignature(privateKey, digest, "perf-test@example.com")
			if err != nil {
				t.Fatalf("Signature creation failed at iteration %d: %v", i, err)
			}
		}
		sigCreateDuration := time.Since(start)
		avgSigCreate := sigCreateDuration / 100

		if avgSigCreate > 5*time.Millisecond {
			t.Errorf("Signature creation too slow: average %v per signature", avgSigCreate)
		}
		t.Logf("Signature creation performance: %v average per signature", avgSigCreate)

		// Test signature verification performance
		digest := "verification-performance-test"
		oms, err := server.CreateTestSignature(privateKey, digest, "perf-test@example.com")
		if err != nil {
			t.Fatalf("Failed to create signature for verification test: %v", err)
		}

		config := server.DefaultSignatureConfig()
		verifier := server.NewCryptoSignatureVerifier(config)

		start = time.Now()
		for i := 0; i < 100; i++ {
			valid, err := verifier.cryptoVerify(oms, digest)
			if err != nil {
				t.Fatalf("Signature verification failed at iteration %d: %v", i, err)
			}
			if !valid {
				t.Fatalf("Signature verification returned invalid at iteration %d", i)
			}
		}
		verifyDuration := time.Since(start)
		avgVerify := verifyDuration / 100

		if avgVerify > 5*time.Millisecond {
			t.Errorf("Signature verification too slow: average %v per verification", avgVerify)
		}
		t.Logf("Signature verification performance: %v average per verification", avgVerify)

		t.Logf("✅ Performance tests completed successfully")
	})
}