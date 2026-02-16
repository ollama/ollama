package manifest

import (
	"path/filepath"
	"testing"
)

func TestTotalTensorSize(t *testing.T) {
	m := &ModelManifest{
		Manifest: &Manifest{
			Layers: []ManifestLayer{
				{MediaType: "application/vnd.ollama.image.tensor", Size: 1000},
				{MediaType: "application/vnd.ollama.image.tensor", Size: 2000},
				{MediaType: "application/vnd.ollama.image.json", Size: 500}, // not a tensor
				{MediaType: "application/vnd.ollama.image.tensor", Size: 3000},
			},
		},
	}

	got := m.TotalTensorSize()
	want := int64(6000)
	if got != want {
		t.Errorf("TotalTensorSize() = %d, want %d", got, want)
	}
}

func TestTotalTensorSizeEmpty(t *testing.T) {
	m := &ModelManifest{
		Manifest: &Manifest{
			Layers: []ManifestLayer{},
		},
	}

	if got := m.TotalTensorSize(); got != 0 {
		t.Errorf("TotalTensorSize() = %d, want 0", got)
	}
}

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
