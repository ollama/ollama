package updatetrust

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSignVerifyAndRejectTamper(t *testing.T) {
	pub, priv, err := GenerateKeyPair()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "Ollama.zip")
	if err := os.WriteFile(path, []byte("zip-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteSignature(priv, path); err != nil {
		t.Fatal(err)
	}
	if err := VerifyFileWithSidecar(pub, path); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(path, []byte("zip-bytez"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := VerifyFileWithSidecar(pub, path); err == nil {
		t.Fatal("tampered artifact was accepted")
	}

	if err := os.WriteFile(path, []byte("zip-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+SignatureSuffix, []byte("not-a-signature\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := VerifyFileWithSidecar(pub, path); err == nil {
		t.Fatal("tampered signature was accepted")
	}

	otherPub, _, err := GenerateKeyPair()
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteSignature(priv, path); err != nil {
		t.Fatal(err)
	}
	if err := VerifyFileWithSidecar(otherPub, path); err == nil {
		t.Fatal("signature from a different key was accepted")
	}
}

func TestVerifyFailsClosedWithoutKey(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "Ollama.zip")
	if err := os.WriteFile(path, []byte("zip-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+SignatureSuffix, []byte("AAAA\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := VerifyFileWithSidecar("", path)
	if err == nil || !errorsIs(err, ErrNoPublicKey) {
		t.Fatalf("empty public key err = %v, want ErrNoPublicKey", err)
	}
	if strings.TrimSpace(PublicKeyPEM) != "" {
		t.Fatal("PublicKeyPEM must stay empty until the release Mac generates a key")
	}
	_, priv, err := GenerateKeyPair()
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckEmbedded(priv); err == nil || !strings.Contains(err.Error(), "no public key is embedded") {
		t.Fatalf("CheckEmbedded err = %v", err)
	}
}

func TestRenderPublicKeyRoundTrip(t *testing.T) {
	pub, priv, err := GenerateKeyPair()
	if err != nil {
		t.Fatal(err)
	}
	src := RenderPublicKeyFile(pub)
	const marker = "const PublicKeyPEM = `"
	i := strings.Index(src, marker)
	if i < 0 {
		t.Fatalf("rendered file has no public key: %s", src)
	}
	rest := src[i+len(marker):]
	j := strings.LastIndex(rest, "`")
	if j < 0 {
		t.Fatal("rendered file has no closing raw string")
	}
	got := rest[:j]
	dir := t.TempDir()
	path := filepath.Join(dir, "a.bin")
	if err := os.WriteFile(path, []byte("payload"), 0o644); err != nil {
		t.Fatal(err)
	}
	sig, err := SignFile(priv, path)
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyFile(got, path, sig); err != nil {
		t.Fatal(err)
	}
}

func TestMissingSidecar(t *testing.T) {
	pub, _, err := GenerateKeyPair()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "Ollama.zip")
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := VerifyFileWithSidecar(pub, path); err == nil {
		t.Fatal("missing sidecar was accepted")
	}
}

func errorsIs(err, target error) bool {
	return err == target || (err != nil && target != nil && strings.Contains(err.Error(), target.Error()))
}
