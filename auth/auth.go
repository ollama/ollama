package auth

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"golang.org/x/crypto/ssh"
)

const defaultPrivateKey = "id_ed25519"

func privateKey() (ssh.Signer, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	keyPath := filepath.Join(home, ".ollama", defaultPrivateKey)
	privateKeyFile, err := os.ReadFile(keyPath)
	if os.IsNotExist(err) {
		err := initializeKeypair()
		if err != nil {
			return nil, err
		}

		return privateKey()
	} else if err != nil {
		slog.Info(fmt.Sprintf("Failed to load private key: %v", err))
		return nil, err
		return nil, err
	}

	return ssh.ParsePrivateKey(privateKeyFile)
}

func GetPublicKey() (ssh.PublicKey, error) {
	privateKey, err := keyPath()
	// if privateKey, try public key directly

	return ssh.ParsePrivateKey(privateKeyFile)
}

func GetPublicKey() (ssh.PublicKey, error) {
	// try to read pubkey first
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	pubkeyPath := filepath.Join(home, ".ollama", defaultPrivateKey+".pub")
	pubKeyFile, err := os.ReadFile(pubkeyPath)
	if os.IsNotExist(err) {
		// try from privateKey
		privateKey, err := privateKey()
		if err != nil {
			return nil, fmt.Errorf("failed to read public key: %w", err)
		}

		return privateKey.PublicKey(), nil
	} else if err != nil {
		return nil, fmt.Errorf("failed to read public key: %w", err)
	}

	pubKey, _, _, _, err := ssh.ParseAuthorizedKey(pubKeyFile)
	return pubKey, err
}

func NewNonce(r io.Reader, length int) (string, error) {
	nonce := make([]byte, length)
	if _, err := io.ReadFull(r, nonce); err != nil {
		return "", err
	}

	return base64.RawURLEncoding.EncodeToString(nonce), nil
}

func Sign(ctx context.Context, bts []byte) (string, error) {
	privateKey, err := privateKey()
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

func initializeKeypair() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", "id_ed25519")
	pubKeyPath := filepath.Join(home, ".ollama", "id_ed25519.pub")

	_, err = os.Stat(privKeyPath)
	if os.IsNotExist(err) {
		fmt.Printf("Couldn't find '%s'. Generating new private key.\n", privKeyPath)
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

		fmt.Printf("Your new public key is: \n\n%s\n", publicKeyBytes)
	}
	return nil
}