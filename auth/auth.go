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
	"strings"

	"golang.org/x/crypto/ssh"
)

const defaultPrivateKey = "id_ed25519"

func keyPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(home, ".ollama", defaultPrivateKey), nil
}

func GetPublicKey() (string, error) {
	keyPath, err := keyPath()
	if err != nil {
		return "", err
	}

	pubKey, err := os.ReadFile(keyPath + ".pub") // note: the .pub suffix is extremely important to not leak the private key
	if err != nil {
		slog.Info(fmt.Sprintf("Failed to load public key: %v", err))
		return "", err
	}

	return strings.TrimSpace(string(pubKey)), nil
}

func NewNonce(r io.Reader, length int) (string, error) {
	nonce := make([]byte, length)
	if _, err := io.ReadFull(r, nonce); err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

func Sign(ctx context.Context, bts []byte) (string, error) {
	keyPath, err := keyPath()
	if err != nil {
		return "", err
	}

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
		return "", fmt.Errorf("malformed public key")
	}

	signedData, err := privateKey.Sign(rand.Reader, bts)
	if err != nil {
		return "", err
	}

	// signature is <pubkey>:<signature>
	return fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob)), nil
}
