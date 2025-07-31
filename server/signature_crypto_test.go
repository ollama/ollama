package server

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestGenerateKeyPair(t *testing.T) {
	publicKey, privateKey, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("GenerateKeyPair failed: %v", err)
	}

	// Validate public key format
	pubKeyBytes, err := base64.StdEncoding.DecodeString(publicKey)
	if err != nil {
		t.Fatalf("Public key is not valid base64: %v", err)
	}
	if len(pubKeyBytes) != ed25519.PublicKeySize {
		t.Fatalf("Public key size is incorrect: got %d, want %d", len(pubKeyBytes), ed25519.PublicKeySize)
	}

	// Validate private key format
	privKeyBytes, err := base64.StdEncoding.DecodeString(privateKey)
	if err != nil {
		t.Fatalf("Private key is not valid base64: %v", err)
	}
	if len(privKeyBytes) != ed25519.PrivateKeySize {
		t.Fatalf("Private key size is incorrect: got %d, want %d", len(privKeyBytes), ed25519.PrivateKeySize)
	}

	// Verify key pair compatibility
	privKey := ed25519.PrivateKey(privKeyBytes)
	derivedPubKey := privKey.Public().(ed25519.PublicKey)
	if !strings.EqualFold(base64.StdEncoding.EncodeToString(derivedPubKey), publicKey) {
		t.Fatalf("Generated public key doesn't match private key")
	}
}

func TestCreateTestSignature(t *testing.T) {
	// Generate test key pair
	publicKey, privateKey, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate key pair: %v", err)
	}

	// Test data
	modelDigest := "test-model-digest-12345"
	signer := "test-signer@example.com"

	// Create signature
	oms, err := CreateTestSignature(privateKey, modelDigest, signer)
	if err != nil {
		t.Fatalf("CreateTestSignature failed: %v", err)
	}

	// Validate OMS structure
	if oms.Version != "oms-v1.0" {
		t.Errorf("Expected version 'oms-v1.0', got '%s'", oms.Version)
	}
	if oms.Algorithm != "ed25519" {
		t.Errorf("Expected algorithm 'ed25519', got '%s'", oms.Algorithm)
	}
	if oms.Signer != signer {
		t.Errorf("Expected signer '%s', got '%s'", signer, oms.Signer)
	}
	if oms.ModelDigest != modelDigest {
		t.Errorf("Expected model digest '%s', got '%s'", modelDigest, oms.ModelDigest)
	}
	if oms.PublicKey != publicKey {
		t.Errorf("Public key mismatch")
	}

	// Validate signature format
	_, err = base64.StdEncoding.DecodeString(oms.Signature)
	if err != nil {
		t.Errorf("Signature is not valid base64: %v", err)
	}

	// Validate timestamp is recent
	if time.Since(oms.Timestamp) > time.Minute {
		t.Errorf("Timestamp is too old: %v", oms.Timestamp)
	}
}

func TestCreateTestSignatureInvalidKey(t *testing.T) {
	_, err := CreateTestSignature("invalid-base64!", "digest", "signer")
	if err == nil {
		t.Fatal("Expected error for invalid private key, got none")
	}
	if !strings.Contains(err.Error(), "failed to decode private key") {
		t.Errorf("Expected 'failed to decode private key' error, got: %v", err)
	}
}

func TestComputeModelDigest(t *testing.T) {
	// Create test manifest
	manifest := &Manifest{
		Config: Layer{
			Digest: "sha256:config123",
		},
		Layers: []Layer{
			{
				MediaType: "application/vnd.ollama.image.model",
				Digest:    "sha256:layer1",
			},
			{
				MediaType: "application/vnd.ollama.image.system",
				Digest:    "sha256:layer2",
			},
			{
				MediaType: "application/vnd.oms.signature.v1+json",
				Digest:    "sha256:signature",
			},
		},
	}

	digest, err := ComputeModelDigest(manifest)
	if err != nil {
		t.Fatalf("ComputeModelDigest failed: %v", err)
	}

	// Digest should be a hex string
	if len(digest) != 64 { // SHA256 produces 64 hex characters
		t.Errorf("Expected digest length 64, got %d", len(digest))
	}

	// Should be deterministic
	digest2, err := ComputeModelDigest(manifest)
	if err != nil {
		t.Fatalf("Second ComputeModelDigest failed: %v", err)
	}
	if digest != digest2 {
		t.Errorf("ComputeModelDigest is not deterministic: %s != %s", digest, digest2)
	}

	// Should exclude signature layers
	// Add another signature layer and verify digest is the same
	manifest.Layers = append(manifest.Layers, Layer{
		MediaType: "application/vnd.oms.signature.v1+json",
		Digest:    "sha256:anothersig",
	})
	digest3, err := ComputeModelDigest(manifest)
	if err != nil {
		t.Fatalf("Third ComputeModelDigest failed: %v", err)
	}
	if digest != digest3 {
		t.Errorf("ComputeModelDigest should exclude signature layers: %s != %s", digest, digest3)
	}
}

func TestCryptoSignatureVerifier(t *testing.T) {
	// Create test config
	config := DefaultSignatureConfig()
	verifier := NewCryptoSignatureVerifier(config)

	// Generate test signature
	_, privateKey, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate key pair: %v", err)
	}

	modelDigest := "test-model-digest"
	signer := "test@example.com"
	oms, err := CreateTestSignature(privateKey, modelDigest, signer)
	if err != nil {
		t.Fatalf("Failed to create test signature: %v", err)
	}

	// Create temporary signature file
	tmpDir, err := os.MkdirTemp("", "signature-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	sigFile := filepath.Join(tmpDir, "signature.json")
	sigData, err := json.Marshal(oms)
	if err != nil {
		t.Fatalf("Failed to marshal signature: %v", err)
	}
	if err := os.WriteFile(sigFile, sigData, 0644); err != nil {
		t.Fatalf("Failed to write signature file: %v", err)
	}

	// Test verification
	result, err := verifier.VerifySignatureFile(sigFile, &SignatureInfo{
		Signer:       signer,
		SignedAt:     oms.Timestamp,
		Format:       oms.Version,
		SignatureURI: "test-uri",
	}, modelDigest)

	if err != nil {
		t.Fatalf("VerifySignatureFile failed: %v", err)
	}
	if !result.Valid {
		t.Errorf("Expected valid signature, got invalid: %s", result.ErrorMessage)
	}
	if result.Signer != signer {
		t.Errorf("Expected signer '%s', got '%s'", signer, result.Signer)
	}
}

func TestCryptoSignatureVerifierInvalidSignature(t *testing.T) {
	config := DefaultSignatureConfig()
	verifier := NewCryptoSignatureVerifier(config)

	// Generate test signature with wrong model digest
	_, privateKey, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate key pair: %v", err)
	}

	oms, err := CreateTestSignature(privateKey, "wrong-digest", "test@example.com")
	if err != nil {
		t.Fatalf("Failed to create test signature: %v", err)
	}

	// Create temporary signature file
	tmpDir, err := os.MkdirTemp("", "signature-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	sigFile := filepath.Join(tmpDir, "signature.json")
	sigData, err := json.Marshal(oms)
	if err != nil {
		t.Fatalf("Failed to marshal signature: %v", err)
	}
	if err := os.WriteFile(sigFile, sigData, 0644); err != nil {
		t.Fatalf("Failed to write signature file: %v", err)
	}

	// Test verification with different model digest
	result, err := verifier.VerifySignatureFile(sigFile, &SignatureInfo{}, "correct-digest")

	if err == nil {
		t.Fatal("Expected error for invalid signature, got none")
	}
	if result.Valid {
		t.Error("Expected invalid signature, got valid")
	}
}

func TestCryptoSignatureVerifierMalformedFile(t *testing.T) {
	config := DefaultSignatureConfig()
	verifier := NewCryptoSignatureVerifier(config)

	// Create temporary malformed signature file
	tmpDir, err := os.MkdirTemp("", "signature-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	sigFile := filepath.Join(tmpDir, "signature.json")
	if err := os.WriteFile(sigFile, []byte("invalid json"), 0644); err != nil {
		t.Fatalf("Failed to write signature file: %v", err)
	}

	// Test verification
	result, err := verifier.VerifySignatureFile(sigFile, &SignatureInfo{}, "digest")

	if err == nil {
		t.Fatal("Expected error for malformed signature file, got none")
	}
	if result.Valid {
		t.Error("Expected invalid signature for malformed file, got valid")
	}
	if !strings.Contains(result.ErrorMessage, "failed to parse signature") {
		t.Errorf("Expected parse error message, got: %s", result.ErrorMessage)
	}
}

func TestVerifyFileIntegrity(t *testing.T) {
	// Create temporary test file
	tmpDir, err := os.MkdirTemp("", "integrity-test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	testFile := filepath.Join(tmpDir, "test.txt")
	testContent := "Hello, World!"
	if err := os.WriteFile(testFile, []byte(testContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	// Expected SHA256 of "Hello, World!"
	expectedDigest := "sha256:dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"

	// Test valid integrity
	valid, err := VerifyFileIntegrity(testFile, expectedDigest)
	if err != nil {
		t.Fatalf("VerifyFileIntegrity failed: %v", err)
	}
	if !valid {
		t.Error("Expected valid file integrity, got invalid")
	}

	// Test invalid integrity
	valid, err = VerifyFileIntegrity(testFile, "sha256:wrongdigest")
	if err != nil {
		t.Fatalf("VerifyFileIntegrity failed: %v", err)
	}
	if valid {
		t.Error("Expected invalid file integrity for wrong digest, got valid")
	}

	// Test non-existent file
	_, err = VerifyFileIntegrity("/nonexistent/file", expectedDigest)
	if err == nil {
		t.Fatal("Expected error for non-existent file, got none")
	}
}

func TestOMSSignatureParsing(t *testing.T) {
	// Test valid OMS signature
	validOMS := &OMSSignature{
		Version:     "oms-v1.0",
		Signature:   "dGVzdC1zaWduYXR1cmU=", // base64 for "test-signature"
		Algorithm:   "ed25519",
		PublicKey:   "dGVzdC1wdWJsaWMta2V5", // base64 for "test-public-key"
		Signer:      "test@example.com",
		Timestamp:   time.Now(),
		ModelDigest: "test-digest",
	}

	// Marshal and unmarshal
	data, err := json.Marshal(validOMS)
	if err != nil {
		t.Fatalf("Failed to marshal OMS: %v", err)
	}

	var parsedOMS OMSSignature
	if err := json.Unmarshal(data, &parsedOMS); err != nil {
		t.Fatalf("Failed to unmarshal OMS: %v", err)
	}

	// Verify fields
	if parsedOMS.Version != validOMS.Version {
		t.Errorf("Version mismatch: %s != %s", parsedOMS.Version, validOMS.Version)
	}
	if parsedOMS.Algorithm != validOMS.Algorithm {
		t.Errorf("Algorithm mismatch: %s != %s", parsedOMS.Algorithm, validOMS.Algorithm)
	}
	if parsedOMS.Signer != validOMS.Signer {
		t.Errorf("Signer mismatch: %s != %s", parsedOMS.Signer, validOMS.Signer)
	}
}

func TestEd25519VerificationFlow(t *testing.T) {
	// Create verifier
	config := DefaultSignatureConfig()
	verifier := NewCryptoSignatureVerifier(config)

	// Generate keys and signature
	_, privateKey, err := GenerateKeyPair()
	if err != nil {
		t.Fatalf("Failed to generate key pair: %v", err)
	}

	modelDigest := "test-digest-for-verification"
	oms, err := CreateTestSignature(privateKey, modelDigest, "test-signer")
	if err != nil {
		t.Fatalf("Failed to create signature: %v", err)
	}

	// Test ed25519 verification directly
	valid, err := verifier.verifyEd25519(oms, modelDigest)
	if err != nil {
		t.Fatalf("verifyEd25519 failed: %v", err)
	}
	if !valid {
		t.Error("Expected valid ed25519 signature, got invalid")
	}

	// Test with wrong digest
	valid, err = verifier.verifyEd25519(oms, "wrong-digest")
	if err != nil {
		t.Fatalf("verifyEd25519 failed with wrong digest: %v", err)
	}
	if valid {
		t.Error("Expected invalid signature with wrong digest, got valid")
	}

	// Test with malformed public key
	oms.PublicKey = "invalid-base64!"
	_, err = verifier.verifyEd25519(oms, modelDigest)
	if err == nil {
		t.Fatal("Expected error for invalid public key, got none")
	}
}