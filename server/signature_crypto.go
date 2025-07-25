package server

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"
)

// OMSSignature represents the OpenSSF Model Signing (OMS) signature format
type OMSSignature struct {
	Version     string                 `json:"version"`              // OMS format version
	Signature   string                 `json:"signature"`            // Base64-encoded signature
	Algorithm   string                 `json:"algorithm"`            // Signature algorithm (e.g., "ed25519")
	PublicKey   string                 `json:"public_key"`           // Base64-encoded public key
	Signer      string                 `json:"signer"`               // Signer identity
	Timestamp   time.Time              `json:"timestamp"`            // Signing timestamp
	ModelDigest string                 `json:"model_digest"`         // SHA256 digest of model being signed
	Metadata    map[string]interface{} `json:"metadata,omitempty"`   // Additional metadata
}

// CryptoSignatureVerifier handles real cryptographic signature verification
type CryptoSignatureVerifier struct {
	config *SignatureConfig
}

// NewCryptoSignatureVerifier creates a new cryptographic signature verifier
func NewCryptoSignatureVerifier(config *SignatureConfig) *CryptoSignatureVerifier {
	return &CryptoSignatureVerifier{config: config}
}

// VerifySignatureFile performs actual cryptographic signature verification
func (csv *CryptoSignatureVerifier) VerifySignatureFile(sigFilePath string, sigInfo *SignatureInfo, modelDigest string) (*SignatureVerificationResult, error) {
	slog.Debug("starting cryptographic signature verification", "file", sigFilePath)

	// Read and parse signature file
	oms, err := csv.parseOMSSignature(sigFilePath)
	if err != nil {
		return &SignatureVerificationResult{
			Valid:        false,
			ErrorMessage: fmt.Sprintf("failed to parse signature: %v", err),
		}, err
	}

	slog.Debug("parsed OMS signature", "algorithm", oms.Algorithm, "signer", oms.Signer)

	// Verify the signature cryptographically
	valid, err := csv.cryptoVerify(oms, modelDigest)
	if err != nil {
		return &SignatureVerificationResult{
			Valid:        false,
			Signer:       oms.Signer,
			SignedAt:     oms.Timestamp,
			Format:       oms.Version,
			ErrorMessage: fmt.Sprintf("cryptographic verification failed: %v", err),
		}, err
	}

	result := &SignatureVerificationResult{
		Valid:    valid,
		Signer:   oms.Signer,
		SignedAt: oms.Timestamp,
		Format:   oms.Version,
	}

	if !valid {
		result.ErrorMessage = "signature verification failed"
		return result, ErrSignatureInvalid
	}

	// Apply policy-based validation
	if policyValid, policyError := csv.config.IsSignatureValid(result, oms.Signer); !policyValid {
		result.Valid = false
		result.ErrorMessage = policyError
		return result, ErrSignatureInvalid
	}

	slog.Debug("signature verification successful", "signer", oms.Signer)
	return result, nil
}

// parseOMSSignature reads and parses an OMS signature file
func (csv *CryptoSignatureVerifier) parseOMSSignature(sigFilePath string) (*OMSSignature, error) {
	data, err := os.ReadFile(sigFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read signature file: %w", err)
	}

	var oms OMSSignature
	if err := json.Unmarshal(data, &oms); err != nil {
		return nil, fmt.Errorf("failed to parse OMS signature: %w", err)
	}

	// Validate required fields
	if oms.Version == "" {
		oms.Version = "oms-v1.0" // Default version
	}
	if oms.Signature == "" {
		return nil, fmt.Errorf("signature field is required")
	}
	if oms.Algorithm == "" {
		return nil, fmt.Errorf("algorithm field is required")
	}
	if oms.PublicKey == "" {
		return nil, fmt.Errorf("public_key field is required")
	}

	return &oms, nil
}

// cryptoVerify performs the actual cryptographic verification
func (csv *CryptoSignatureVerifier) cryptoVerify(oms *OMSSignature, modelDigest string) (bool, error) {
	switch strings.ToLower(oms.Algorithm) {
	case "ed25519":
		return csv.verifyEd25519(oms, modelDigest)
	default:
		return false, fmt.Errorf("unsupported signature algorithm: %s", oms.Algorithm)
	}
}

// verifyEd25519 verifies an ed25519 signature
func (csv *CryptoSignatureVerifier) verifyEd25519(oms *OMSSignature, modelDigest string) (bool, error) {
	// Decode public key
	pubKeyBytes, err := base64.StdEncoding.DecodeString(oms.PublicKey)
	if err != nil {
		return false, fmt.Errorf("failed to decode public key: %w", err)
	}

	if len(pubKeyBytes) != ed25519.PublicKeySize {
		return false, fmt.Errorf("invalid ed25519 public key size: %d", len(pubKeyBytes))
	}

	pubKey := ed25519.PublicKey(pubKeyBytes)

	// Decode signature
	signature, err := base64.StdEncoding.DecodeString(oms.Signature)
	if err != nil {
		return false, fmt.Errorf("failed to decode signature: %w", err)
	}

	// Create message to verify (combination of model digest and metadata)
	message := csv.createSigningMessage(modelDigest, oms)

	// Verify signature
	valid := ed25519.Verify(pubKey, message, signature)
	
	slog.Debug("ed25519 verification result", "valid", valid, "message_length", len(message))
	return valid, nil
}

// createSigningMessage creates the message that was signed
func (csv *CryptoSignatureVerifier) createSigningMessage(modelDigest string, oms *OMSSignature) []byte {
	// Create a canonical message format
	// Format: "model:<digest>|signer:<signer>|timestamp:<timestamp>|algorithm:<algorithm>"
	message := fmt.Sprintf("model:%s|signer:%s|timestamp:%s|algorithm:%s",
		modelDigest,
		oms.Signer,
		oms.Timestamp.Format(time.RFC3339),
		oms.Algorithm,
	)
	
	// Hash the message for consistent signing
	hash := sha256.Sum256([]byte(message))
	return hash[:]
}

// ComputeModelDigest computes the SHA256 digest of a model's layers
func ComputeModelDigest(manifest *Manifest) (string, error) {
	hasher := sha256.New()
	
	// Include all non-signature layers in the digest
	for _, layer := range manifest.Layers {
		if !layer.IsSignature() {
			// Write layer digest to hasher
			hasher.Write([]byte(layer.Digest))
		}
	}
	
	// Include config digest
	hasher.Write([]byte(manifest.Config.Digest))
	
	digest := hex.EncodeToString(hasher.Sum(nil))
	slog.Debug("computed model digest", "digest", digest[:12]+"...")
	return digest, nil
}

// GenerateKeyPair generates a new ed25519 key pair for testing/development
func GenerateKeyPair() (publicKey, privateKey string, err error) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		return "", "", fmt.Errorf("failed to generate key pair: %w", err)
	}

	publicKeyB64 := base64.StdEncoding.EncodeToString(pub)
	privateKeyB64 := base64.StdEncoding.EncodeToString(priv)

	return publicKeyB64, privateKeyB64, nil
}

// CreateTestSignature creates a test signature for development/testing
func CreateTestSignature(privateKeyB64, modelDigest, signer string) (*OMSSignature, error) {
	// Decode private key
	privKeyBytes, err := base64.StdEncoding.DecodeString(privateKeyB64)
	if err != nil {
		return nil, fmt.Errorf("failed to decode private key: %w", err)
	}

	privateKey := ed25519.PrivateKey(privKeyBytes)
	publicKey := privateKey.Public().(ed25519.PublicKey)

	// Create OMS signature structure
	oms := &OMSSignature{
		Version:     "oms-v1.0",
		Algorithm:   "ed25519",
		PublicKey:   base64.StdEncoding.EncodeToString(publicKey),
		Signer:      signer,
		Timestamp:   time.Now(),
		ModelDigest: modelDigest,
	}

	// Create signing message
	csv := &CryptoSignatureVerifier{}
	message := csv.createSigningMessage(modelDigest, oms)

	// Sign the message
	signature := ed25519.Sign(privateKey, message)
	oms.Signature = base64.StdEncoding.EncodeToString(signature)

	return oms, nil
}

// VerifyFileIntegrity verifies the integrity of a file against its expected digest
func VerifyFileIntegrity(filePath, expectedDigest string) (bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return false, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return false, fmt.Errorf("failed to read file: %w", err)
	}

	actualDigest := "sha256:" + hex.EncodeToString(hasher.Sum(nil))
	return actualDigest == expectedDigest, nil
}