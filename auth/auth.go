package auth

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/pem"
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

// EnsureKeypair creates the default signing keypair when it does not exist.
func EnsureKeypair(out io.Writer) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", defaultPrivateKey)
	pubKeyPath := privKeyPath + ".pub"
	if _, err := os.Stat(privKeyPath); !os.IsNotExist(err) {
		return nil
	}

	fmt.Fprintf(out, "Couldn't find '%s'. Generating new private key.\n", privKeyPath)
	cryptoPublicKey, cryptoPrivateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}

	privateKeyBytes, err := ssh.MarshalPrivateKey(cryptoPrivateKey, "")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(privKeyPath), 0o755); err != nil {
		return fmt.Errorf("could not create directory %w", err)
	}
	if err := os.WriteFile(privKeyPath, pem.EncodeToMemory(privateKeyBytes), 0o600); err != nil {
		return err
	}

	sshPublicKey, err := ssh.NewPublicKey(cryptoPublicKey)
	if err != nil {
		return err
	}
	publicKeyBytes := ssh.MarshalAuthorizedKey(sshPublicKey)
	if err := os.WriteFile(pubKeyPath, publicKeyBytes, 0o644); err != nil {
		return err
	}
	fmt.Fprintf(out, "Your new public key is: \n\n%s\n", publicKeyBytes)
	return nil
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
