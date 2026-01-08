package auth

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/crypto/ssh"
)

const (
	defaultPrivateKey = "id_ed25519"
	signInStateFile   = "signin.json"
)

// SignInState represents the locally cached sign-in state
type SignInState struct {
	Name     string    `json:"name"`
	Email    string    `json:"email"`
	CachedAt time.Time `json:"cached_at"`
}

func GetPublicKey() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	keyPath := filepath.Join(home, ".ollama", defaultPrivateKey)
	privateKeyFile, err := os.ReadFile(keyPath)
	if err != nil {
		slog.Info(fmt.Sprintf("Failed to load private key: %v", err))
		return "", err
	}

	privateKey, err := ssh.ParsePrivateKey(privateKeyFile)
	if err != nil {
		return "", err
	}

	publicKey := ssh.MarshalAuthorizedKey(privateKey.PublicKey())

	return strings.TrimSpace(string(publicKey)), nil
}

func NewNonce(r io.Reader, length int) (string, error) {
	nonce := make([]byte, length)
	if _, err := io.ReadFull(r, nonce); err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

func Sign(ctx context.Context, bts []byte) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	keyPath := filepath.Join(home, ".ollama", defaultPrivateKey)
	privateKeyFile, err := os.ReadFile(keyPath)
	if err != nil {
		slog.Info(fmt.Sprintf("Failed to load private key: %v", err))
		return "", err
	}

	privateKey, err := ssh.ParsePrivateKey(privateKeyFile)
	if err != nil {
		return "", err
	}

	// get the pubkey, but remove the type
	publicKey := ssh.MarshalAuthorizedKey(privateKey.PublicKey())
	parts := bytes.Split(publicKey, []byte(" "))
	if len(parts) < 2 {
		return "", errors.New("malformed public key")
	}

	signedData, err := privateKey.Sign(rand.Reader, bts)
	if err != nil {
		return "", err
	}

	// signature is <pubkey>:<signature>
	return fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob)), nil
}

// GetSignInState reads the locally cached sign-in state from ~/.ollama/signin.json
func GetSignInState() (*SignInState, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	statePath := filepath.Join(home, ".ollama", signInStateFile)
	data, err := os.ReadFile(statePath)
	if err != nil {
		return nil, err
	}

	var state SignInState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}

	return &state, nil
}

// SetSignInState atomically writes the sign-in state to ~/.ollama/signin.json
func SetSignInState(state *SignInState) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	ollamaDir := filepath.Join(home, ".ollama")
	statePath := filepath.Join(ollamaDir, signInStateFile)
	tmpPath := statePath + ".tmp"

	state.CachedAt = time.Now()

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}

	// Write to temp file first
	if err := os.WriteFile(tmpPath, data, 0o600); err != nil {
		return err
	}

	// Atomic rename
	return os.Rename(tmpPath, statePath)
}

// ClearSignInState removes the locally cached sign-in state
func ClearSignInState() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	statePath := filepath.Join(home, ".ollama", signInStateFile)
	err = os.Remove(statePath)
	if errors.Is(err, os.ErrNotExist) {
		return nil // Already cleared
	}
	return err
}

// IsSignedIn returns true if there is a valid locally cached sign-in state
func IsSignedIn() bool {
	state, err := GetSignInState()
	if err != nil {
		return false
	}
	return state.Name != ""
}
