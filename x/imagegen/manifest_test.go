package imagegen

import (
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestLoadManifestRespectsOLLAMA_MODELS(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", "/custom/models/path")

	// Verify envconfig.Models() returns our custom path
	if got := envconfig.Models(); got != "/custom/models/path" {
		t.Fatalf("envconfig.Models() = %q, want %q", got, "/custom/models/path")
	}

	// LoadManifest will fail (no manifest exists), but we can verify
	// the error message contains the custom path
	_, err := LoadManifest("test-model")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	expectedPath := filepath.Join("/custom/models/path", "manifests", "registry.ollama.ai", "library", "test-model", "latest")
	if got := err.Error(); got != "read manifest: open "+expectedPath+": no such file or directory" {
		t.Errorf("error = %q, want path to contain %q", got, expectedPath)
	}
}
