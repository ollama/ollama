package imagegen

import (
	"path/filepath"
	"testing"
)

func TestManifestAndBlobDirsRespectOLLAMAModels(t *testing.T) {
	modelsDir := filepath.Join(t.TempDir(), "models")

	// Simulate packaged/systemd environment
	t.Setenv("OLLAMA_MODELS", modelsDir)
	t.Setenv("HOME", "/usr/share/ollama")

	// Manifest dir must respect OLLAMA_MODELS
	wantManifest := filepath.Join(modelsDir, "manifests")
	if got := DefaultManifestDir(); got != wantManifest {
		t.Fatalf("DefaultManifestDir() = %q, want %q", got, wantManifest)
	}

	// Blob dir must respect OLLAMA_MODELS
	wantBlobs := filepath.Join(modelsDir, "blobs")
	if got := DefaultBlobDir(); got != wantBlobs {
		t.Fatalf("DefaultBlobDir() = %q, want %q", got, wantBlobs)
	}
}
