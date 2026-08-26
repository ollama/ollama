package auth

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestEnsureKeypair(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	var output bytes.Buffer
	if err := EnsureKeypair(&output); err != nil {
		t.Fatal(err)
	}
	if output.Len() == 0 {
		t.Fatal("EnsureKeypair() did not report the generated public key")
	}

	for _, name := range []string{"id_ed25519", "id_ed25519.pub"} {
		if _, err := os.Stat(filepath.Join(home, ".ollama", name)); err != nil {
			t.Fatalf("generated key %s: %v", name, err)
		}
	}
	if _, err := Sign(context.Background(), []byte("request")); err != nil {
		t.Fatalf("Sign() after EnsureKeypair(): %v", err)
	}

	output.Reset()
	if err := EnsureKeypair(&output); err != nil {
		t.Fatal(err)
	}
	if output.Len() != 0 {
		t.Fatal("EnsureKeypair() replaced an existing key")
	}
}
