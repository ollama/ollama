package updater

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/app/updatetrust"
)

func TestVerifyStagedArtifactRejectsTamper(t *testing.T) {
	pub, priv, err := updatetrust.GenerateKeyPair()
	if err != nil {
		t.Fatal(err)
	}
	old := updatePublicKey
	updatePublicKey = pub
	t.Cleanup(func() { updatePublicKey = old })

	path := filepath.Join(t.TempDir(), "Ollama.zip")
	if err := os.WriteFile(path, []byte("good-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := updatetrust.WriteSignature(priv, path); err != nil {
		t.Fatal(err)
	}
	if err := verifyStagedArtifact(path); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(path, []byte("bad-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := verifyStagedArtifact(path); err == nil {
		t.Fatal("tampered staged artifact was accepted")
	}

	if err := os.WriteFile(path, []byte("good-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path + updatetrust.SignatureSuffix); err != nil {
		t.Fatal(err)
	}
	if err := verifyStagedArtifact(path); err == nil {
		t.Fatal("missing signature was accepted")
	}
}

func TestVerifyStagedArtifactFailsClosedWithoutKey(t *testing.T) {
	old := updatePublicKey
	updatePublicKey = ""
	t.Cleanup(func() { updatePublicKey = old })

	path := filepath.Join(t.TempDir(), "Ollama.zip")
	if err := os.WriteFile(path, []byte("good-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+updatetrust.SignatureSuffix, []byte("AAAA\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := verifyStagedArtifact(path)
	if err == nil || !strings.Contains(err.Error(), "no update public key") {
		t.Fatalf("err = %v", err)
	}
}
