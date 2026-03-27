package server

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestPullManifestPermissionFailureCleansBlobs(t *testing.T) {
	tmp := t.TempDir()

	// simulate models directory
	modelDir := filepath.Join(tmp, "models")
	blobDir := filepath.Join(modelDir, "blobs")
	manifestDir := filepath.Join(modelDir, "manifests")

	os.MkdirAll(blobDir, 0755)
	os.MkdirAll(manifestDir, 0755)

	// remove write permissions to simulate permission failure
	os.Chmod(manifestDir, 0000)

	defer os.Chmod(manifestDir, 0755)

	name := model.ParseName("test/model:latest")

	opts := &registryOptions{}
	err := PullModel(context.Background(), name.String(), opts, func(p api.ProgressResponse) {})

	if err == nil {
		t.Fatalf("expected error due to manifest permission failure")
	}

	entries, err := os.ReadDir(blobDir)
	if err != nil {
		t.Fatal(err)
	}

	if len(entries) != 0 {
		t.Fatalf("expected blobs to be cleaned up after failure")
	}
}
