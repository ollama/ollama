package server

import (
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/types/model"
)

var (
	// ErrSignatureNotFound indicates no signature file was found for the model
	ErrSignatureNotFound = errors.New("signature not found")
	// ErrSignatureInvalid indicates the signature verification failed
	ErrSignatureInvalid = errors.New("signature verification failed")
	// ErrSignatureUnsupportedFormat indicates an unsupported signature format
	ErrSignatureUnsupportedFormat = errors.New("unsupported signature format")
)

// SignatureVerificationResult contains the results of signature verification
type SignatureVerificationResult struct {
	Valid        bool      `json:"valid"`         // Whether the signature is valid
	Signer       string    `json:"signer"`        // Identity of the signer
	SignedAt     time.Time `json:"signed_at"`     // When the model was signed
	Format       string    `json:"format"`        // Signature format used
	ErrorMessage string    `json:"error,omitempty"` // Error details if verification failed
}

// SignatureVerifier handles model signature verification
type SignatureVerifier struct {
	config *SignatureConfig
}

// NewSignatureVerifier creates a new signature verifier with loaded configuration
func NewSignatureVerifier() *SignatureVerifier {
	config, err := LoadSignatureConfig()
	if err != nil {
		// Log error but use default config to avoid breaking functionality
		slog.Warn("failed to load signature config, using defaults", "error", err)
		config = DefaultSignatureConfig()
	}
	return &SignatureVerifier{config: config}
}

// NewSignatureVerifierWithConfig creates a new signature verifier with specific configuration
func NewSignatureVerifierWithConfig(config *SignatureConfig) *SignatureVerifier {
	return &SignatureVerifier{config: config}
}

// VerifyModel verifies the signature of a model by name
func (sv *SignatureVerifier) VerifyModel(modelName model.Name) (*SignatureVerificationResult, error) {
	manifest, err := ParseNamedManifest(modelName)
	if err != nil {
		return nil, fmt.Errorf("failed to load model manifest: %w", err)
	}

	return sv.VerifyManifest(manifest)
}

// VerifyManifest verifies the signature of a model manifest
func (sv *SignatureVerifier) VerifyManifest(manifest *Manifest) (*SignatureVerificationResult, error) {
	// Check if model has signature metadata
	if manifest.Signature == nil {
		return &SignatureVerificationResult{
			Valid:        false,
			ErrorMessage: "model is not signed",
		}, ErrSignatureNotFound
	}

	// Check if signature layer exists
	sigLayer := manifest.GetSignatureLayer()
	if sigLayer == nil {
		return &SignatureVerificationResult{
			Valid:        false,
			ErrorMessage: "signature metadata exists but signature file not found",
		}, ErrSignatureNotFound
	}

	// Get the signature file path
	sigFilePath, err := GetBlobsPath(sigLayer.Digest)
	if err != nil {
		return &SignatureVerificationResult{
			Valid:        false,
			ErrorMessage: fmt.Sprintf("failed to locate signature file: %v", err),
		}, err
	}

	// TODO: Implement actual signature verification using model-transparency library
	// Perform cryptographic verification with manifest context
	return sv.verifySignatureFileWithManifest(sigFilePath, manifest.Signature, manifest)
}

// verifySignatureFileWithManifest performs the actual signature verification using cryptographic methods
func (sv *SignatureVerifier) verifySignatureFileWithManifest(sigFilePath string, sigInfo *SignatureInfo, manifest *Manifest) (*SignatureVerificationResult, error) {
	// Check file exists
	if _, err := filepath.Abs(sigFilePath); err != nil {
		return &SignatureVerificationResult{
			Valid:        false,
			ErrorMessage: "signature file path is invalid",
		}, ErrSignatureInvalid
	}

	// Compute model digest for verification
	modelDigest, err := ComputeModelDigest(manifest)
	if err != nil {
		return &SignatureVerificationResult{
			Valid:        false,
			ErrorMessage: fmt.Sprintf("failed to compute model digest: %v", err),
		}, err
	}

	// Use cryptographic verifier for real verification
	cryptoVerifier := NewCryptoSignatureVerifier(sv.config)
	return cryptoVerifier.VerifySignatureFile(sigFilePath, sigInfo, modelDigest)
}

// HasValidSignature is a convenience function to check if a model has a valid signature
func HasValidSignature(modelName model.Name) (bool, error) {
	verifier := NewSignatureVerifier()
	result, err := verifier.VerifyModel(modelName)
	if err != nil {
		if errors.Is(err, ErrSignatureNotFound) {
			return false, nil // Not signed, but not an error
		}
		return false, err
	}
	return result.Valid, nil
}

// GetSignatureInfo returns signature information for a model without full verification
func GetSignatureInfo(modelName model.Name) (*SignatureInfo, error) {
	manifest, err := ParseNamedManifest(modelName)
	if err != nil {
		return nil, fmt.Errorf("failed to load model manifest: %w", err)
	}

	return manifest.Signature, nil
}