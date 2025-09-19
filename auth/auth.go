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

const (
	defaultPrivateKey = "id_ed25519"
	keyDir            = ".ollama"
)

func keyPath() (string, error) {
	fileIsReadable := func(fp string) bool {
		info, err := os.Stat(fp)
		if err != nil {
			return false
		}

		// Check that it's a regular file, not a directory or other file type
		if !info.Mode().IsRegular() {
			return false
		}

		// Try to open it to check readability
		file, err := os.Open(fp)
		if err != nil {
			return false
		}
		file.Close()
		return true
	}

	systemPath := filepath.Join("/usr/share/ollama/.ollama", defaultPrivateKey)
	if fileIsReadable(systemPath) {
		return systemPath, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}
	return filepath.Join(home, keyDir, defaultPrivateKey), nil
}

func loadPrivateKey() (ssh.Signer, error) {
	path, err := keyPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		slog.Info("Failed to read private key", "error", err, "path", path)
		return nil, fmt.Errorf("failed to read private key: %w", err)
	}

	signer, err := ssh.ParsePrivateKey(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse private key: %w", err)
	}

	return signer, nil
}

func GetPublicKey() (string, error) {
	signer, err := loadPrivateKey()
	if err != nil {
		return "", err
	}

	pubKey := ssh.MarshalAuthorizedKey(signer.PublicKey())
	return strings.TrimSpace(string(pubKey)), nil
}

func NewNonce(r io.Reader, length int) (string, error) {
	nonce := make([]byte, length)
	if _, err := io.ReadFull(r, nonce); err != nil {
		return "", fmt.Errorf("failed to generate nonce: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

func Sign(ctx context.Context, data []byte) (string, error) {
	signer, err := loadPrivateKey()
	if err != nil {
		return "", err
	}

	pubKey := ssh.MarshalAuthorizedKey(signer.PublicKey())
	parts := bytes.Split(pubKey, []byte(" "))
	if len(parts) < 2 {
		return "", errors.New("malformed public key")
	}

	signature, err := signer.Sign(rand.Reader, data)
	if err != nil {
		return "", fmt.Errorf("failed to sign data: %w", err)
	}

	return fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signature.Blob)), nil
}
