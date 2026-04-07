package auth

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/crypto/ssh"
)

func TestNewNonce(t *testing.T) {
	t.Run("generates nonce of specified length", func(t *testing.T) {
		tests := []struct {
			length   int
			expected int
		}{
			{16, 22},
			{32, 43},
			{64, 86},
		}

		for _, tt := range tests {
			nonce, err := NewNonce(rand.Reader, tt.length)
			if err != nil {
				t.Fatalf("NewNonce(%d) returned error: %v", tt.length, err)
			}
			if len(nonce) != tt.expected {
				t.Errorf("NewNonce(%d) = %q, want length %d", tt.length, nonce, tt.expected)
			}

			decoded, err := base64.RawURLEncoding.DecodeString(nonce)
			if err != nil {
				t.Fatalf("Failed to decode nonce: %v", err)
			}
			if len(decoded) != tt.length {
				t.Errorf("Decoded nonce length = %d, want %d", len(decoded), tt.length)
			}
		}
	})

	t.Run("generates unique nonces", func(t *testing.T) {
		nonces := make(map[string]bool)
		for i := 0; i < 100; i++ {
			nonce, err := NewNonce(rand.Reader, 32)
			if err != nil {
				t.Fatalf("NewNonce returned error: %v", err)
			}
			if nonces[nonce] {
				t.Error("Generated duplicate nonce")
			}
			nonces[nonce] = true
		}
	})

	t.Run("propagates reader errors", func(t *testing.T) {
		errReader := &errorReader{err: errors.New("read error")}
		_, err := NewNonce(errReader, 16)
		if err == nil {
			t.Error("NewNonce with error reader should return error")
		}
		if !strings.Contains(err.Error(), "read error") {
			t.Errorf("NewNonce error = %v, want to contain 'read error'", err)
		}
	})
}

type errorReader struct {
	err error
}

func (e *errorReader) Read(p []byte) (n int, err error) {
	return 0, e.err
}

func setupHomeDir(t *testing.T) string {
	t.Helper()

	tmpDir := t.TempDir()

	homeKey := "HOME"
	if _, homeExists := os.LookupEnv("HOME"); !homeExists {
		if _, userProfileExists := os.LookupEnv("USERPROFILE"); userProfileExists {
			homeKey = "USERPROFILE"
		}
	}

	origHome := os.Getenv(homeKey)
	t.Cleanup(func() {
		if origHome == "" {
			os.Unsetenv(homeKey)
		} else {
			os.Setenv(homeKey, origHome)
		}
	})

	os.Setenv(homeKey, tmpDir)
	return tmpDir
}

func writeSSHEd25519PrivateKey(t *testing.T, path string) ed25519.PrivateKey {
	t.Helper()

	_, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("Failed to generate key: %v", err)
	}

	pemBlock, err := ssh.MarshalPrivateKey(privKey, "")
	if err != nil {
		t.Fatalf("Failed to marshal private key: %v", err)
	}

	privKeyPEM := pem.EncodeToMemory(pemBlock)
	if err := os.WriteFile(path, privKeyPEM, 0o600); err != nil {
		t.Fatalf("Failed to write private key: %v", err)
	}

	return privKey
}

func TestGetPublicKey(t *testing.T) {
	t.Run("returns error when key file does not exist", func(t *testing.T) {
		setupHomeDir(t)

		_, err := GetPublicKey()
		if err == nil {
			t.Error("GetPublicKey with missing key should return error")
		}
	})

	t.Run("returns valid public key from private key", func(t *testing.T) {
		tmpDir := setupHomeDir(t)

		ollamaDir := filepath.Join(tmpDir, ".ollama")
		if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
			t.Fatalf("Failed to create .ollama dir: %v", err)
		}

		privKeyFile := filepath.Join(ollamaDir, "id_ed25519")
		_ = writeSSHEd25519PrivateKey(t, privKeyFile)

		pubKey, err := GetPublicKey()
		if err != nil {
			t.Fatalf("GetPublicKey returned error: %v", err)
		}

		if pubKey == "" {
			t.Error("GetPublicKey returned empty string")
		}

		if !strings.HasPrefix(pubKey, "ssh-ed25519") {
			t.Errorf("GetPublicKey = %q, want to start with 'ssh-ed25519'", pubKey)
		}
	})
}

func TestSign(t *testing.T) {
	t.Run("returns error when key file does not exist", func(t *testing.T) {
		setupHomeDir(t)

		_, err := Sign(nil, []byte("test"))
		if err == nil {
			t.Error("Sign with missing key should return error")
		}
	})

	t.Run("signs data and returns valid signature format", func(t *testing.T) {
		tmpDir := setupHomeDir(t)

		ollamaDir := filepath.Join(tmpDir, ".ollama")
		if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
			t.Fatalf("Failed to create .ollama dir: %v", err)
		}

		privKeyFile := filepath.Join(ollamaDir, "id_ed25519")
		privKey := writeSSHEd25519PrivateKey(t, privKeyFile)

		signer, err := ssh.NewSignerFromSigner(privKey)
		if err != nil {
			t.Fatalf("Failed to create signer: %v", err)
		}
		sshPubKey := string(ssh.MarshalAuthorizedKey(signer.PublicKey()))
		sshPubKey = strings.TrimSpace(sshPubKey)
		sshPubKeyParts := strings.Split(sshPubKey, " ")
		if len(sshPubKeyParts) < 2 {
			t.Fatalf("Unexpected SSH public key format: %q", sshPubKey)
		}
		expectedPubKey := sshPubKeyParts[1]

		data := []byte("test message to sign")
		signature, err := Sign(nil, data)
		if err != nil {
			t.Fatalf("Sign returned error: %v", err)
		}

		parts := strings.SplitN(signature, ":", 2)
		if len(parts) != 2 {
			t.Fatalf("Sign returned %q, want 'pubkey:signature' format", signature)
		}

		if parts[0] != expectedPubKey {
			t.Errorf("Public key in signature = %q, want %q", parts[0], expectedPubKey)
		}

		sigBytes, err := base64.StdEncoding.DecodeString(parts[1])
		if err != nil {
			t.Fatalf("Failed to decode signature: %v", err)
		}

		pubKey, ok := signer.PublicKey().(ssh.CryptoPublicKey)
		if !ok {
			t.Fatalf("Failed to get crypto public key")
		}
		cryptoPubKey := pubKey.CryptoPublicKey()
		ed25519PubKey, ok := cryptoPubKey.(ed25519.PublicKey)
		if !ok {
			t.Fatalf("Failed to get ed25519 public key")
		}

		if !ed25519.Verify(ed25519PubKey, data, sigBytes) {
			t.Error("Signature verification failed")
		}
	})

	t.Run("generates unique signatures for different data", func(t *testing.T) {
		tmpDir := setupHomeDir(t)

		ollamaDir := filepath.Join(tmpDir, ".ollama")
		if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
			t.Fatalf("Failed to create .ollama dir: %v", err)
		}

		privKeyFile := filepath.Join(ollamaDir, "id_ed25519")
		writeSSHEd25519PrivateKey(t, privKeyFile)

		sig1, err := Sign(nil, []byte("message1"))
		if err != nil {
			t.Fatalf("Sign message1 error: %v", err)
		}

		sig2, err := Sign(nil, []byte("message2"))
		if err != nil {
			t.Fatalf("Sign message2 error: %v", err)
		}

		if sig1 == sig2 {
			t.Error("Signatures for different messages should differ")
		}
	})
}

func TestSignPublicKeyFormat(t *testing.T) {
	tmpDir := setupHomeDir(t)

	ollamaDir := filepath.Join(tmpDir, ".ollama")
	if err := os.MkdirAll(ollamaDir, 0o755); err != nil {
		t.Fatalf("Failed to create .ollama dir: %v", err)
	}

	privKeyFile := filepath.Join(ollamaDir, "id_ed25519")
	writeSSHEd25519PrivateKey(t, privKeyFile)

	data := []byte("test")
	signature, err := Sign(nil, data)
	if err != nil {
		t.Fatalf("Sign returned error: %v", err)
	}

	parts := strings.SplitN(signature, ":", 2)
	if len(parts) != 2 {
		t.Fatalf("Signature format = %q, want pubkey:base64sig", signature)
	}

	decoded, err := base64.StdEncoding.DecodeString(parts[1])
	if err != nil {
		t.Errorf("Signature is not valid base64: %v", err)
	}

	expectedSigLen := ed25519.SignatureSize
	if len(decoded) != expectedSigLen {
		t.Errorf("Signature length = %d, want %d", len(decoded), expectedSigLen)
	}
}