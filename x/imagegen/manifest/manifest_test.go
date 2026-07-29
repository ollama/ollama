package manifest

import (
	"os"
	"path/filepath"
	"testing"

	rootmanifest "github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
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

func TestLoadManifestPrefersV2(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	name := model.ParseName("example")

	legacyPath, err := rootmanifest.PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(legacyPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(legacyPath, []byte(`{"schemaVersion":2,"mediaType":"legacy"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	v2Path, err := rootmanifest.V2PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(v2Path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(v2Path, []byte(`{"schemaVersion":2,"mediaType":"v2"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	m, err := LoadManifest(name.String())
	if err != nil {
		t.Fatal(err)
	}
	if m.Manifest.MediaType != "v2" {
		t.Fatalf("media type = %q, want %q", m.Manifest.MediaType, "v2")
	}
}
