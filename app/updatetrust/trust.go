// Package updatetrust signs and verifies fork update artifacts.
//
// The signature is standard base64 of an ed25519 signature over the raw
// SHA-256 digest of the file (32 bytes), the same construction Flowa uses.
// The signature is stored next to the file as "<name>.ed25519". The private
// key is PKCS#8 PEM and never belongs in the repository. The public key is
// SPKI PEM embedded in PublicKeyPEM.
package updatetrust

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

const SignatureSuffix = ".ed25519"

// MaxSignatureBytes is how much of a signature sidecar the updater will read.
const MaxSignatureBytes = 4096

var (
	ErrNoPublicKey  = errors.New("no update public key is embedded")
	ErrBadSignature = errors.New("update signature rejected")
)

// GenerateKeyPair returns a new ed25519 key pair as SPKI and PKCS#8 PEM.
func GenerateKeyPair() (publicPEM, privatePEM string, err error) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return "", "", err
	}
	pubDER, err := x509.MarshalPKIXPublicKey(pub)
	if err != nil {
		return "", "", err
	}
	privDER, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return "", "", err
	}
	publicPEM = string(pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: pubDER}))
	privatePEM = string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privDER}))
	return publicPEM, privatePEM, nil
}

// PublicFromPrivate derives the SPKI public PEM from a PKCS#8 private PEM.
func PublicFromPrivate(privatePEM string) (string, error) {
	priv, err := parsePrivate(privatePEM)
	if err != nil {
		return "", err
	}
	pubDER, err := x509.MarshalPKIXPublicKey(priv.Public())
	if err != nil {
		return "", err
	}
	return string(pem.EncodeToMemory(&pem.Block{Type: "PUBLIC KEY", Bytes: pubDER})), nil
}

// SignFile returns the base64 signature of SHA-256(file).
func SignFile(privatePEM, path string) (string, error) {
	priv, err := parsePrivate(privatePEM)
	if err != nil {
		return "", err
	}
	digest, err := hashFile(path)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(ed25519.Sign(priv, digest[:])), nil
}

// WriteSignature writes path+".ed25519".
func WriteSignature(privatePEM, path string) error {
	sig, err := SignFile(privatePEM, path)
	if err != nil {
		return err
	}
	return os.WriteFile(path+SignatureSuffix, []byte(sig+"\n"), 0o644)
}

// VerifyFile checks signatureB64 against SHA-256(path) using publicPEM.
// An empty public key or signature fails closed.
func VerifyFile(publicPEM, path, signatureB64 string) error {
	pub, err := parsePublic(publicPEM)
	if err != nil {
		return err
	}
	sig, err := base64.StdEncoding.DecodeString(strings.TrimSpace(signatureB64))
	if err != nil || len(sig) == 0 {
		return ErrBadSignature
	}
	digest, err := hashFile(path)
	if err != nil {
		return err
	}
	if !ed25519.Verify(pub, digest[:], sig) {
		return ErrBadSignature
	}
	return nil
}

// VerifyFileWithSidecar reads the "<path>.ed25519" sidecar and verifies path.
func VerifyFileWithSidecar(publicPEM, path string) error {
	b, err := os.ReadFile(path + SignatureSuffix)
	if err != nil {
		return fmt.Errorf("read update signature: %w", err)
	}
	if len(b) == 0 || len(b) > MaxSignatureBytes {
		return ErrBadSignature
	}
	return VerifyFile(publicPEM, path, string(b))
}

// CheckEmbedded reports whether privatePEM matches the key compiled into
// PublicKeyPEM. An empty embedded key is an error.
func CheckEmbedded(privatePEM string) error {
	derived, err := PublicFromPrivate(privatePEM)
	if err != nil {
		return err
	}
	embedded := strings.TrimSpace(PublicKeyPEM)
	if embedded == "" {
		return errors.New("no public key is embedded in app/updatetrust/public_key.go; run scripts/generate-update-key.sh and commit that file")
	}
	if normalizePEM(derived) != normalizePEM(embedded) {
		return errors.New("the embedded public key does not match this private key")
	}
	return nil
}

// RenderPublicKeyFile returns the full source of public_key.go for publicPEM.
func RenderPublicKeyFile(publicPEM string) string {
	body := strings.TrimSpace(publicPEM)
	return `package updatetrust

// PublicKeyPEM is the SPKI ed25519 public key the app trusts for update
// artifacts. scripts/generate-update-key.sh rewrites this file on the release
// Mac. The matching private key stays on that Mac and is never committed.
// An empty value fails closed: the updater will not install an update.
const PublicKeyPEM = ` + "`" + body + "\n`" + "\n"
}

func hashFile(path string) ([32]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return [32]byte{}, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return [32]byte{}, err
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out, nil
}

func parsePrivate(privatePEM string) (ed25519.PrivateKey, error) {
	block, _ := pem.Decode([]byte(privatePEM))
	if block == nil {
		return nil, errors.New("update private key is not PEM")
	}
	key, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	priv, ok := key.(ed25519.PrivateKey)
	if !ok {
		return nil, errors.New("update private key is not ed25519")
	}
	return priv, nil
}

func parsePublic(publicPEM string) (ed25519.PublicKey, error) {
	if strings.TrimSpace(publicPEM) == "" {
		return nil, ErrNoPublicKey
	}
	block, _ := pem.Decode([]byte(publicPEM))
	if block == nil {
		return nil, ErrNoPublicKey
	}
	key, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	pub, ok := key.(ed25519.PublicKey)
	if !ok {
		return nil, errors.New("update public key is not ed25519")
	}
	return pub, nil
}

func normalizePEM(s string) string {
	return strings.TrimSpace(strings.ReplaceAll(s, "\r\n", "\n"))
}
