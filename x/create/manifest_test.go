package create

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/ollama/ollama/manifest"
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
