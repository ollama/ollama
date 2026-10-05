//go:build integration && release

package integration

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func runQuantization(t *testing.T) {
	if testModel != "" {
		t.Skip("exercises quantization with a fixed source model, not applicable with model override")
	}

	isolateCreateModelStore(t)
	modelDir := filepath.Join(testdataModelsDir, tinyLlamaModelDir)
	downloadHFModel(t, tinyLlamaRepo, tinyLlamaRevision, modelDir)
	ensureMLXLibraryPath(t)
	t.Setenv("OLLAMA_CREATE_REMOTE", "false")

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	t.Cleanup(cleanup)

	modelName := createIntegrationModelName("test-tinyllama-nvfp4")
	cleanupCreatedModel(t, client, modelName)
	runOllamaCreate(ctx, t, modelName, "-q", "nvfp4", "-f", tinyLlamaModelfile(t, modelDir))

	resp, err := client.Show(ctx, &api.ShowRequest{Name: modelName})
	if err != nil {
		t.Fatalf("Model show failed after create: %v", err)
	}
	if !strings.EqualFold(resp.Details.QuantizationLevel, "nvfp4") {
		t.Fatalf("QuantizationLevel = %q, want nvfp4", resp.Details.QuantizationLevel)
	}

	verifyTinyLlamaChat(ctx, t, client, modelName)
}
