package create

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestSafetensorsModelfileLayersIncludesParameters(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	layers, err := safetensorsModelfileLayers("", "", nil, map[string]any{
		"temperature": float32(0.7),
		"stop":        []string{"USER:", "ASSISTANT:"},
	})
	if err != nil {
		t.Fatal(err)
	}

	if len(layers) != 1 {
		t.Fatalf("len(layers) = %d, want 1", len(layers))
	}
	if layers[0].MediaType != "application/vnd.ollama.image.params" {
		t.Fatalf("MediaType = %q, want %q", layers[0].MediaType, "application/vnd.ollama.image.params")
	}

	blobPath, err := manifest.BlobsPath(layers[0].Digest)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatal(err)
	}

	var got map[string]any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got["temperature"] != float64(0.7) {
		t.Fatalf("temperature = %v, want %v", got["temperature"], float64(0.7))
	}
}

func TestNewSafetensorsManifestWriterDoesNotDuplicateVisionCapability(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "model_index.json"), []byte(`{"_class_name":"Flux2KleinPipeline"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	writeManifest := NewSafetensorsManifestWriter(SafetensorsManifestOptions{
		ModelDir:     modelDir,
		Capabilities: []string{"completion", "vision"},
	})
	if err := writeManifest("test", ManifestInfo{}); err != nil {
		t.Fatal(err)
	}

	m, err := manifest.ParseNamedManifest(model.ParseName("test"))
	if err != nil {
		t.Fatal(err)
	}
	configPath, err := manifest.BlobsPath(m.Config.Digest)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	var config struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	want := []string{"completion", "vision"}
	if !slices.Equal(config.Capabilities, want) {
		t.Fatalf("capabilities = %v, want %v", config.Capabilities, want)
	}
}
