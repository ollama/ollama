package server

import (
	"errors"
	"fmt"
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
	// Future: Will contain configuration for different verification methods
	// - Public keys for key-based verification
	// - Trust roots for certificate-based verification  
	// - Sigstore configuration for keyless verification
}

// NewSignatureVerifier creates a new signature verifier with default configuration
func NewSignatureVerifier() *SignatureVerifier {
	return &SignatureVerifier{}
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
	// For now, we'll do basic file existence checks and return a placeholder result
	return sv.verifySignatureFile(sigFilePath, manifest.Signature)
}

// verifySignatureFile performs the actual signature verification (placeholder implementation)
func (sv *SignatureVerifier) verifySignatureFile(sigFilePath string, sigInfo *SignatureInfo) (*SignatureVerificationResult, error) {
	// Placeholder implementation - just check that signature file exists
	// In a future commit, this will be replaced with actual verification using model-transparency

	// Check file exists
	if _, err := filepath.Abs(sigFilePath); err != nil {
		return &SignatureVerificationResult{
			Valid:        false,
			ErrorMessage: "signature file path is invalid",
		}, ErrSignatureInvalid
	}

	// TODO: Replace this placeholder with actual verification:
	// 1. Load signature file and parse OMS format
	// 2. Extract public key or certificate from signature
	// 3. Verify signature against model files using model-transparency library
	// 4. Validate signature chain and trust roots
	// 5. Return detailed verification results

	// For now, return a successful verification if signature metadata exists
	return &SignatureVerificationResult{
		Valid:        true, // PLACEHOLDER: Always succeed for now
		Signer:       sigInfo.Signer,
		SignedAt:     sigInfo.SignedAt,
		Format:       sigInfo.Format,
		ErrorMessage: "placeholder verification - not yet fully implemented",
	}, nil
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