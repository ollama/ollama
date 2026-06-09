//go:build integration && imagegen

package integration

import (
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestCreateImageGen(t *testing.T) {
	skipIfRemote(t)
	skipUnderMinVRAM(t, 13)

	// Allow overriding the model directory via env var for local testing,
	// since the model is ~33GB and may already be downloaded elsewhere.
	modelDir := os.Getenv("OLLAMA_TEST_IMAGEGEN_MODEL_DIR")
	if modelDir == "" {
		modelDir = filepath.Join(testdataModelsDir, "Z-Image-Turbo")
		downloadHFModel(t, "Tongyi-MAI/Z-Image-Turbo", modelDir)
	} else {
		t.Logf("Using existing imagegen model at %s", modelDir)
	}

	// Verify it looks like a valid imagegen model directory
	if _, err := os.Stat(filepath.Join(modelDir, "model_index.json")); err != nil {
		t.Fatalf("model_index.json not found in %s — not a valid imagegen model directory", modelDir)
	}

	ensureMLXLibraryPath(t)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	modelName := "test-z-image-turbo-create"

	absModelDir, err := filepath.Abs(modelDir)
	if err != nil {
		t.Fatalf("Failed to get absolute path: %v", err)
	}

	// Create a Modelfile pointing to the diffusers model directory
	tmpModelfile := filepath.Join(t.TempDir(), "Modelfile")
	if err := os.WriteFile(tmpModelfile, []byte("FROM "+absModelDir+"\n"), 0o644); err != nil {
		t.Fatalf("Failed to write Modelfile: %v", err)
	}

	t.Logf("Creating imagegen model from %s", absModelDir)
	runOllamaCreate(ctx, t, modelName, "--experimental", "-f", tmpModelfile)

	// Verify model exists via show
	showReq := &api.ShowRequest{Name: modelName}
	showResp, err := client.Show(ctx, showReq)
	if err != nil {
		t.Fatalf("Model show failed after create: %v", err)
	}
	t.Logf("Created model details: %+v", showResp.Details)

	// Generate an image to verify the model isn't corrupted
	t.Log("Generating test image...")
	imageBase64, err := generateImage(ctx, client, modelName, "A red circle on a white background")
	if err != nil {
		if strings.Contains(err.Error(), "image generation not available") {
			t.Skip("Target system does not support image generation")
		} else if strings.Contains(err.Error(), "insufficient memory for image generation") {
			t.Skip("insufficient memory for image generation")
		} else if strings.Contains(err.Error(), "ollama-mlx: no such file or directory") {
			t.Skip("unsupported architecture")
		}
		t.Fatalf("Image generation failed: %v", err)
	}

	// Verify we got valid image data
	imageData, err := base64.StdEncoding.DecodeString(imageBase64)
	if err != nil {
		t.Fatalf("Failed to decode base64 image: %v", err)
	}

	t.Logf("Generated image: %d bytes", len(imageData))

	if len(imageData) < 1000 {
		t.Fatalf("Generated image suspiciously small (%d bytes), likely corrupted", len(imageData))
	}

	// Check for PNG or JPEG magic bytes
	isPNG := len(imageData) >= 4 && imageData[0] == 0x89 && imageData[1] == 'P' && imageData[2] == 'N' && imageData[3] == 'G'
	isJPEG := len(imageData) >= 2 && imageData[0] == 0xFF && imageData[1] == 0xD8
	if !isPNG && !isJPEG {
		t.Fatalf("Generated image is neither PNG nor JPEG (first bytes: %x)", imageData[:min(8, len(imageData))])
	}
	t.Logf("Image format validated (PNG=%v, JPEG=%v)", isPNG, isJPEG)

	// Cleanup: delete the model
	deleteReq := &api.DeleteRequest{Model: modelName}
	if err := client.Delete(ctx, deleteReq); err != nil {
		t.Logf("Warning: failed to delete test model: %v", err)
	}
}
