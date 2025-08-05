package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// SignaturePolicy defines the signature verification policy
type SignaturePolicy string

const (
	// PolicyPermissive allows unsigned models (default)
	PolicyPermissive SignaturePolicy = "permissive"
	// PolicyWarn requires signatures but allows unsigned with warnings
	PolicyWarn SignaturePolicy = "warn"
	// PolicyStrict requires valid signatures for all models
	PolicyStrict SignaturePolicy = "strict"
)

// TrustedSigner represents a trusted model signer
type TrustedSigner struct {
	ID          string    `json:"id"`           // Unique identifier
	Name        string    `json:"name"`         // Human-readable name
	Email       string    `json:"email"`        // Signer email
	PublicKey   string    `json:"public_key"`   // Public key or certificate
	AddedAt     time.Time `json:"added_at"`     // When added to trust store
	Description string    `json:"description"`  // Optional description
}

// SignatureConfig contains signature verification configuration
type SignatureConfig struct {
	// Global policy for signature verification
	Policy SignaturePolicy `json:"policy"`

	// Whether to verify signatures during pull operations
	VerifyOnPull bool `json:"verify_on_pull"`

	// Whether to verify signatures during push operations
	VerifyOnPush bool `json:"verify_on_push"`

	// List of trusted signers
	TrustedSigners []TrustedSigner `json:"trusted_signers"`

	// Whether to require signatures from trusted signers only
	RequireTrustedSigners bool `json:"require_trusted_signers"`

	// Maximum signature age in days (0 = no limit)
	MaxSignatureAge int `json:"max_signature_age"`

	// Whether to check signature revocation
	CheckRevocation bool `json:"check_revocation"`

	// Last updated timestamp
	UpdatedAt time.Time `json:"updated_at"`
}

// DefaultSignatureConfig returns the default signature configuration
func DefaultSignatureConfig() *SignatureConfig {
	return &SignatureConfig{
		Policy:                PolicyPermissive,
		VerifyOnPull:          true,
		VerifyOnPush:          true,
		TrustedSigners:        []TrustedSigner{},
		RequireTrustedSigners: false,
		MaxSignatureAge:       0, // No limit
		CheckRevocation:       false,
		UpdatedAt:             time.Now(),
	}
}

// GetSignatureConfigPath returns the path to the signature configuration file
func GetSignatureConfigPath() (string, error) {
	configDir := filepath.Join(envconfig.Models(), "config")
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		return "", fmt.Errorf("failed to create config directory: %w", err)
	}
	return filepath.Join(configDir, "signature.json"), nil
}

// LoadSignatureConfig loads the signature configuration from disk
func LoadSignatureConfig() (*SignatureConfig, error) {
	configPath, err := GetSignatureConfigPath()
	if err != nil {
		return nil, err
	}

	// If config file doesn't exist, return default config
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return DefaultSignatureConfig(), nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read signature config: %w", err)
	}

	var config SignatureConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse signature config: %w", err)
	}

	return &config, nil
}

// SaveSignatureConfig saves the signature configuration to disk
func SaveSignatureConfig(config *SignatureConfig) error {
	configPath, err := GetSignatureConfigPath()
	if err != nil {
		return err
	}

	config.UpdatedAt = time.Now()

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal signature config: %w", err)
	}

	if err := os.WriteFile(configPath, data, 0o644); err != nil {
		return fmt.Errorf("failed to write signature config: %w", err)
	}

	return nil
}

// AddTrustedSigner adds a trusted signer to the configuration
func (c *SignatureConfig) AddTrustedSigner(signer TrustedSigner) error {
	// Check if signer already exists
	for _, existing := range c.TrustedSigners {
		if existing.ID == signer.ID || existing.Email == signer.Email {
			return fmt.Errorf("signer already exists: %s", signer.Email)
		}
	}

	signer.AddedAt = time.Now()
	c.TrustedSigners = append(c.TrustedSigners, signer)
	return nil
}

// RemoveTrustedSigner removes a trusted signer from the configuration
func (c *SignatureConfig) RemoveTrustedSigner(signerID string) error {
	for i, signer := range c.TrustedSigners {
		if signer.ID == signerID {
			c.TrustedSigners = append(c.TrustedSigners[:i], c.TrustedSigners[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("signer not found: %s", signerID)
}

// IsTrustedSigner checks if a signer is in the trusted signers list
func (c *SignatureConfig) IsTrustedSigner(signerEmail string) bool {
	for _, signer := range c.TrustedSigners {
		if signer.Email == signerEmail {
			return true
		}
	}
	return false
}

// IsSignatureValid checks if a signature meets the policy requirements
func (c *SignatureConfig) IsSignatureValid(result *SignatureVerificationResult, signerEmail string) (bool, string) {
	if !result.Valid {
		return false, "signature verification failed"
	}

	// Check if signature is from a trusted signer (if required)
	if c.RequireTrustedSigners && !c.IsTrustedSigner(signerEmail) {
		return false, "signature not from trusted signer"
	}

	// Check signature age
	if c.MaxSignatureAge > 0 {
		maxAge := time.Duration(c.MaxSignatureAge) * 24 * time.Hour
		if time.Since(result.SignedAt) > maxAge {
			return false, fmt.Sprintf("signature older than %d days", c.MaxSignatureAge)
		}
	}

	return true, ""
}