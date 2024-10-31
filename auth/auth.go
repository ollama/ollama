// Package auth provides cryptographic authentication utilities using SSH keys
package auth

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/ssh"
)

var (
	ErrMalformedKey     = errors.New("malformed public key")
	ErrKeyNotFound      = errors.New("private key not found")
	ErrInvalidKeyFormat = errors.New("invalid key format")
)

const (
	defaultPrivateKey = "id_ed25519"
	defaultKeyPerms   = 0600
)

// keyPath returns the path to the private key file
func keyPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}

	return filepath.Join(home, ".ollama", defaultPrivateKey), nil
}

// GetPublicKey retrieves the public key corresponding to the private key
func GetPublicKey() (string, error) {
	keyPath, err := keyPath()
	if err != nil {
		return "", fmt.Errorf("failed to get key path: %w", err)
	}

	privateKeyFile, err := os.ReadFile(keyPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", ErrKeyNotFound
		}
		slog.Error("Failed to load private key", "error", err)
		return "", fmt.Errorf("failed to read private key: %w", err)
	}

	privateKey, err := ssh.ParsePrivateKey(privateKeyFile)
	if err != nil {
		return "", fmt.Errorf("failed to parse private key: %w", err)
	}

	publicKey := ssh.MarshalAuthorizedKey(privateKey.PublicKey())
	return strings.TrimSpace(string(publicKey)), nil
}

// NewNonce generates a cryptographically secure random nonce of the specified length
func NewNonce(r io.Reader, length int) (string, error) {
	if length <= 0 {
		return "", fmt.Errorf("nonce length must be positive, got %d", length)
	}

	if r == nil {
		r = rand.Reader
	}

	nonce := make([]byte, length)
	if _, err := io.ReadFull(r, nonce); err != nil {
		return "", fmt.Errorf("failed to generate nonce: %w", err)
	}

	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

// Sign signs the provided data using the private key
func Sign(ctx context.Context, data []byte) (string, error) {
	if len(data) == 0 {
		return "", errors.New("cannot sign empty data")
	}

	select {
	case <-ctx.Done():
		return "", ctx.Err()
	default:
	}

	keyPath, err := keyPath()
	if err != nil {
		return "", fmt.Errorf("failed to get key path: %w", err)
	}

	privateKeyFile, err := os.ReadFile(keyPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", ErrKeyNotFound
		}
		slog.Error("Failed to load private key", "error", err)
		return "", fmt.Errorf("failed to read private key: %w", err)
	}

	privateKey, err := ssh.ParsePrivateKey(privateKeyFile)
	if err != nil {
		return "", fmt.Errorf("failed to parse private key: %w", err)
	}

	publicKey := ssh.MarshalAuthorizedKey(privateKey.PublicKey())
	parts := bytes.Split(publicKey, []byte(" "))
	if len(parts) < 2 {
		return "", ErrMalformedKey
	}

	signedData, err := privateKey.Sign(rand.Reader, data)
	if err != nil {
		return "", fmt.Errorf("failed to sign data: %w", err)
	}

	signature := fmt.Sprintf("%s:%s",
		bytes.TrimSpace(parts[1]),
		base64.StdEncoding.EncodeToString(signedData.Blob))
	
	return signature, nil
}
