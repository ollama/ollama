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

	"github.com/ollama/ollama/envconfig"
)

const (
	defaultPrivateKey = "id_ed25519"
	defaultPublicKey  = "id_ed25519.pub"
)

func pubkeyFromFile(keyPath string) ([]byte, error) {
	if _, err := os.Stat(keyPath); err != nil {
		return nil, err
	}

	data, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, err
	}

	pubKey, _, _, _, err := ssh.ParseAuthorizedKey(data)
	if err != nil {
		return nil, err
	}

	return ssh.MarshalAuthorizedKey(pubKey), nil
}

func pubkeyFromPrivateKeyFile(keyPath string) ([]byte, error) {
	if _, err := os.Stat(keyPath); err != nil {
		return nil, err
	}

	data, err := os.ReadFile(keyPath)
	if err != nil {
		return nil, err
	}

	privateKey, err := ssh.ParsePrivateKey(data)
	if err != nil {
		return nil, err
	}

	return ssh.MarshalAuthorizedKey(privateKey.PublicKey()), nil
}

func GetPublicKey() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	for _, dir := range []string{envconfig.BaseDir(), home} {
		pk, _ := pubkeyFromFile(filepath.Join(dir, defaultPublicKey))
		if len(pk) > 0 {
			return strings.TrimSpace(string(pk)), nil
		}

		pk, _ = pubkeyFromPrivateKeyFile(filepath.Join(dir, defaultPrivateKey))
		if len(pk) > 0 {
			return strings.TrimSpace(string(pk)), nil
		}
	}

	return "", fmt.Errorf("no public key found")
}

func NewNonce(r io.Reader, length int) (string, error) {
	nonce := make([]byte, length)
	if _, err := io.ReadFull(r, nonce); err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

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
		return "", err
	}

	return filepath.Join(home, ".ollama", defaultPrivateKey), nil
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
