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

const defaultPrivateKey = "id_ed25519"

var ErrInvalidToken = errors.New("invalid token")

func keyPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(home, ".ollama", defaultPrivateKey), nil
}

func parseToken(token string) (key, sig []byte, _ error) {
	keyData, sigData, ok := strings.Cut(token, ":")
	if !ok {
		return nil, nil, fmt.Errorf("identity: parseToken: %w", ErrInvalidToken)
	}
	sig, err := base64.StdEncoding.DecodeString(sigData)
	if err != nil {
		return nil, nil, fmt.Errorf("identity: parseToken: base64 decoding signature: %w", err)
	}
	return []byte(keyData), sig, nil
}

func Authenticate(token, checkData string) (ssh.PublicKey, error) {
	keyShort, sigBytes, err := parseToken(token)
	if err != nil {
		return nil, err
	}
	keyLong := append([]byte("ssh-ed25519 "), keyShort...)
	pub, _, _, _, err := ssh.ParseAuthorizedKey(keyLong)
	if err != nil {
		return nil, err
	}

	if err := pub.Verify([]byte(checkData), &ssh.Signature{
		Format: pub.Type(),
		Blob:   sigBytes,
	}); err != nil {
		return nil, err
	}

	return pub, nil
}

func GetPublicKey() (string, error) {
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
		return "", errors.New("malformed public key")
	}

	signedData, err := privateKey.Sign(rand.Reader, bts)
	if err != nil {
		return "", err
	}

	// signature is <pubkey>:<signature>
	return fmt.Sprintf("%s:%s", bytes.TrimSpace(parts[1]), base64.StdEncoding.EncodeToString(signedData.Blob)), nil
}
