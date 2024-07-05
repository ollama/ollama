package auth

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"golang.org/x/crypto/ssh"
)

const defaultPrivateKey = "id_ed25519"

func keyPath() (ssh.Signer, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	keyPath := filepath.Join(home, ".ollama", defaultPrivateKey)
	privateKeyFile, err := os.ReadFile(keyPath)
	if err != nil {
		slog.Info(fmt.Sprintf("Failed to load private key: %v", err))
		return nil, err
	}

	return ssh.ParsePrivateKey(privateKeyFile)
}

func GetPublicKey() (ssh.PublicKey, error) {
	privateKey, err := keyPath()
	// if privateKey, try public key directly

	if err != nil {
		return nil, err
	}

	return privateKey.PublicKey(), nil
}

func NewNonce(r io.Reader, length int) (string, error) {
	nonce := make([]byte, length)
	if _, err := io.ReadFull(r, nonce); err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

func Sign(ctx context.Context, bts []byte) (string, error) {
	privateKey, err := keyPath()
	if err != nil {
		return "", err
	}

	// get the pubkey, but remove the type
	publicKey, err := GetPublicKey()
	if err != nil {
		return "", err
	}

	publicKeyBytes := ssh.MarshalAuthorizedKey(publicKey)

	parts := bytes.Split(publicKeyBytes, []byte(" "))
	if len(parts) < 2 {
		return "", fmt.Errorf("malformed public key")
	}

	signedData, err := privateKey.Sign(rand.Reader, bts)
	if err != nil {
		return "", err
	}

	// signature is <pubkey>:<signature>
	return fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob)), nil
}
