//go:build integration

package integration

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

// CreateTestModel creates a minimal test model for export/import testing
func CreateTestModel(t *testing.T, client *api.Client, name string) {
	t.Helper()
	
	// Create a simple test model using the existing pattern
	ctx := context.Background()
	
	// First ensure we have a base model to work with
	err := PullIfMissing(ctx, client, smol)
	require.NoError(t, err, "failed to pull base model")
	
	// Create the model
	req := &api.CreateRequest{
		Model:      name,
		From:       smol,
		System:     "You are a helpful test assistant.",
		Parameters: map[string]any{
			"temperature": 0.8,
			"top_p":       0.9,
		},
	}
	
	err = client.Create(ctx, req, func(resp api.ProgressResponse) error {
		t.Logf("Creating test model: %s", resp.Status)
		return nil
	})
	require.NoError(t, err, "failed to create test model")
	
	// Verify the model was created
	_, err = client.Show(ctx, &api.ShowRequest{Model: name})
	require.NoError(t, err, "test model should exist after creation")
}

// ExportTestModel exports a model and returns the export path
func ExportTestModel(t *testing.T, client *api.Client, modelName, exportPath string) string {
	t.Helper()
	
	ctx := context.Background()
	
	// Ensure export directory exists
	exportDir := filepath.Dir(exportPath)
	err := os.MkdirAll(exportDir, 0755)
	require.NoError(t, err, "failed to create export directory")
	
	// Perform export
	req := &api.ExportRequest{
		Model: modelName,
		Path:  exportPath,
	}
	
	err = client.Export(ctx, req, func(resp api.ProgressResponse) {
		t.Logf("Export progress: %s", resp.Status)
	})
	require.NoError(t, err, "export should succeed")
	
	// Verify export exists
	_, err = os.Stat(exportPath)
	require.NoError(t, err, "export file/directory should exist")
	
	return exportPath
}

// ImportTestModel imports a model from the given path
func ImportTestModel(t *testing.T, client *api.Client, importPath, targetName string) {
	t.Helper()
	
	ctx := context.Background()
	
	// Perform import
	req := &api.ImportRequest{
		Path:  importPath,
		Model: targetName,
	}
	
	err := client.Import(ctx, req, func(resp api.ProgressResponse) {
		t.Logf("Import progress: %s", resp.Status)
	})
	require.NoError(t, err, "import should succeed")
	
	// Determine the final model name
	finalName := targetName
	if finalName == "" {
		// If no target name specified, use the original model name from metadata
		// For simplicity in tests, we'll assume it's the same as the export
		finalName = "imported-model"
	}
	
	// Verify the model was imported
	_, err = client.Show(ctx, &api.ShowRequest{Model: finalName})
	require.NoError(t, err, "imported model should exist")
}

// VerifyModelIntegrity verifies that a model works correctly after import
func VerifyModelIntegrity(t *testing.T, client *api.Client, modelName string) {
	t.Helper()
	
	ctx := context.Background()
	
	// Test basic model information
	showResp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	require.NoError(t, err, "should be able to show model info")
	require.NotEmpty(t, showResp.Modelfile, "model should have modelfile")
	
	// Test basic generation
	req := &api.GenerateRequest{
		Model:  modelName,
		Prompt: "Say hello",
		Stream: &[]bool{false}[0],
	}
	
	var response string
	err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		response += resp.Response
		return nil
	})
	require.NoError(t, err, "should be able to generate with imported model")
	require.NotEmpty(t, response, "should receive a response")
}

// CleanupTestFiles removes test files and directories
func CleanupTestFiles(t *testing.T, paths []string) {
	t.Helper()
	
	for _, path := range paths {
		if err := os.RemoveAll(path); err != nil {
			t.Logf("Warning: failed to cleanup %s: %v", path, err)
		}
	}
}

// CreateTempDir creates a temporary directory for test exports
func CreateTempDir(t *testing.T) string {
	t.Helper()
	
	tempDir, err := os.MkdirTemp("", "ollama-export-test-*")
	require.NoError(t, err, "failed to create temp directory")
	
	t.Cleanup(func() {
		os.RemoveAll(tempDir)
	})
	
	return tempDir
}

// WaitForCompletion waits for an operation to complete with timeout
func WaitForCompletion(t *testing.T, timeout time.Duration, checkFn func() bool) {
	t.Helper()
	
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if checkFn() {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	
	t.Fatalf("operation did not complete within %v", timeout)
}
