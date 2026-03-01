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
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/ssh"
)

const defaultPrivateKey = "id_ed25519"

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

// SignRequest adds a nonce query parameter and an Authorization header with
// an Ed25519 signature to req.
func SignRequest(ctx context.Context, req *http.Request) error {
	nonce, err := NewNonce(rand.Reader, 16)
	if err != nil {
		return err
	}

	q := req.URL.Query()
	q.Set("nonce", nonce)
	req.URL.RawQuery = q.Encode()

	data := []byte(fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI()))
	signature, err := Sign(ctx, data)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", signature)
	return nil
}
