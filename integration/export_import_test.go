//go:build integration

package integration

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

func TestExportImportRoundTrip(t *testing.T) {
	ctx := context.Background()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Create a test model
	testModelName := "test-model-roundtrip"
	CreateTestModel(t, client, testModelName)
	defer func() {
		client.Delete(ctx, &api.DeleteRequest{Model: testModelName})
	}()

	tempDir := CreateTempDir(t)
	exportPath := filepath.Join(tempDir, "roundtrip-export.tar")

	// Export the model
	t.Log("Exporting model...")
	ExportTestModel(t, client, testModelName, exportPath)

	// Delete the original model
	err := client.Delete(ctx, &api.DeleteRequest{Model: testModelName})
	require.NoError(t, err, "should be able to delete original model")

	// Import the model back
	t.Log("Importing model...")
	ImportTestModel(t, client, exportPath, testModelName)

	// Verify the imported model works
	t.Log("Verifying model integrity...")
	VerifyModelIntegrity(t, client, testModelName)
}

func TestExportCompressionFormats(t *testing.T) {
	ctx := context.Background()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Create a test model
	testModelName := "test-compression-formats"
	CreateTestModel(t, client, testModelName)
	defer func() {
		client.Delete(ctx, &api.DeleteRequest{Model: testModelName})
	}()

	tempDir := CreateTempDir(t)

	// Test different compression formats
	testCases := []struct {
		name        string
		exportPath  string
		compress    string
		expectError bool
	}{
		{
			name:       "Directory Export",
			exportPath: filepath.Join(tempDir, "test-dir"),
			compress:   "",
		},
		{
			name:       "Uncompressed TAR",
			exportPath: filepath.Join(tempDir, "test-uncompressed.tar"),
			compress:   "",
		},
		{
			name:       "GZIP Compressed",
			exportPath: filepath.Join(tempDir, "test-gzip.tar.gz"),
			compress:   "gzip",
		},
		{
			name:       "ZSTD Compressed",
			exportPath: filepath.Join(tempDir, "test-zstd.tar.zst"),
			compress:   "zstd",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Export with specific format
			req := &api.ExportRequest{
				Model:    testModelName,
				Path:     tc.exportPath,
				Compress: tc.compress,
			}

			err := client.Export(ctx, req, func(resp api.ProgressResponse) {
				t.Logf("Export progress: %s", resp.Status)
			})

			if tc.expectError {
				require.Error(t, err, "should fail for invalid format")
				return
			}

			require.NoError(t, err, "export should succeed")

			// Verify export exists
			_, err = os.Stat(tc.exportPath)
			require.NoError(t, err, "export should exist")

			// Import and verify
			importModelName := fmt.Sprintf("%s-imported-%s", testModelName, strings.ReplaceAll(tc.name, " ", "-"))
			ImportTestModel(t, client, tc.exportPath, importModelName)

			// Verify imported model works
			VerifyModelIntegrity(t, client, importModelName)

			// Cleanup imported model
			client.Delete(ctx, &api.DeleteRequest{Model: importModelName})
		})
	}
}

func TestExportImportErrorHandling(t *testing.T) {
	ctx := context.Background()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	tempDir := CreateTempDir(t)

	t.Run("Export Non-existent Model", func(t *testing.T) {
		req := &api.ExportRequest{
			Model: "non-existent-model",
			Path:  filepath.Join(tempDir, "non-existent.tar"),
		}

		err := client.Export(ctx, req, func(resp api.ProgressResponse) {})
		require.Error(t, err, "should fail to export non-existent model")
		require.Contains(t, err.Error(), "not found", "error should mention model not found")
	})

	t.Run("Export to Invalid Path", func(t *testing.T) {
		testModelName := "test-invalid-path"
		CreateTestModel(t, client, testModelName)
		defer client.Delete(ctx, &api.DeleteRequest{Model: testModelName})

		// Try to export to a path that doesn't exist and can't be created
		req := &api.ExportRequest{
			Model: testModelName,
			Path:  "/root/invalid/path/export.tar", // Assume no permission to create
		}

		err := client.Export(ctx, req, func(resp api.ProgressResponse) {})
		require.Error(t, err, "should fail to export to invalid path")
	})

	t.Run("Import Non-existent File", func(t *testing.T) {
		req := &api.ImportRequest{
			Path:  "/non/existent/file.tar",
			Model: "should-not-work",
		}

		err := client.Import(ctx, req, func(resp api.ProgressResponse) {})
		require.Error(t, err, "should fail to import non-existent file")
		require.Contains(t, err.Error(), "not found", "error should mention file not found")
	})

	t.Run("Import Corrupted Archive", func(t *testing.T) {
		// Create a corrupted tar file
		corruptedPath := filepath.Join(tempDir, "corrupted.tar")
		err := os.WriteFile(corruptedPath, []byte("not a valid tar file"), 0644)
		require.NoError(t, err)

		req := &api.ImportRequest{
			Path:  corruptedPath,
			Model: "corrupted-model",
		}

		err = client.Import(ctx, req, func(resp api.ProgressResponse) {})
		require.Error(t, err, "should fail to import corrupted archive")
	})
}

func TestForceOverwriteBehavior(t *testing.T) {
	ctx := context.Background()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Create a test model
	testModelName := "test-force-overwrite"
	CreateTestModel(t, client, testModelName)
	defer client.Delete(ctx, &api.DeleteRequest{Model: testModelName})

	tempDir := CreateTempDir(t)
	exportPath := filepath.Join(tempDir, "force-test.tar")

	// First export
	ExportTestModel(t, client, testModelName, exportPath)

	t.Run("Export Without Force Should Fail", func(t *testing.T) {
		req := &api.ExportRequest{
			Model: testModelName,
			Path:  exportPath,
			Force: false,
		}

		err := client.Export(ctx, req, func(resp api.ProgressResponse) {})
		require.Error(t, err, "should fail to overwrite existing file without force")
		require.Contains(t, err.Error(), "already exists", "error should mention file already exists")
	})

	t.Run("Export With Force Should Succeed", func(t *testing.T) {
		req := &api.ExportRequest{
			Model: testModelName,
			Path:  exportPath,
			Force: true,
		}

		err := client.Export(ctx, req, func(resp api.ProgressResponse) {})
		require.NoError(t, err, "should succeed to overwrite existing file with force")
	})

	t.Run("Import Without Force Should Fail", func(t *testing.T) {
		// Create another model with the same name
		anotherModelName := "test-force-import"
		CreateTestModel(t, client, anotherModelName)
		defer client.Delete(ctx, &api.DeleteRequest{Model: anotherModelName})

		req := &api.ImportRequest{
			Path:  exportPath,
			Model: anotherModelName,
			Force: false,
		}

		err := client.Import(ctx, req, func(resp api.ProgressResponse) {})
		require.Error(t, err, "should fail to import over existing model without force")
		require.Contains(t, err.Error(), "already exists", "error should mention model already exists")
	})

	t.Run("Import With Force Should Succeed", func(t *testing.T) {
		anotherModelName := "test-force-import"
		req := &api.ImportRequest{
			Path:  exportPath,
			Model: anotherModelName,
			Force: true,
		}

		err := client.Import(ctx, req, func(resp api.ProgressResponse) {})
		require.NoError(t, err, "should succeed to import over existing model with force")

		// Verify the model works
		VerifyModelIntegrity(t, client, anotherModelName)
	})
}

func TestProgressReporting(t *testing.T) {
	ctx := context.Background()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Create a test model
	testModelName := "test-progress-reporting"
	CreateTestModel(t, client, testModelName)
	defer client.Delete(ctx, &api.DeleteRequest{Model: testModelName})

	tempDir := CreateTempDir(t)
	exportPath := filepath.Join(tempDir, "progress-test.tar")

	t.Run("Export Progress Reporting", func(t *testing.T) {
		var progressUpdates []api.ProgressResponse
		var statusUpdates []string

		req := &api.ExportRequest{
			Model: testModelName,
			Path:  exportPath,
		}

		err := client.Export(ctx, req, func(resp api.ProgressResponse) {
			progressUpdates = append(progressUpdates, resp)
			if resp.Status != "" {
				statusUpdates = append(statusUpdates, resp.Status)
			}
		})

		require.NoError(t, err, "export should succeed")
		require.NotEmpty(t, progressUpdates, "should receive progress updates")
		require.NotEmpty(t, statusUpdates, "should receive status updates")

		// Verify we get meaningful status messages
		statusStr := strings.Join(statusUpdates, " ")
		require.Contains(t, statusStr, "exporting", "should contain exporting status")
		require.Contains(t, statusStr, "complete", "should contain completion status")
	})

	t.Run("Import Progress Reporting", func(t *testing.T) {
		var progressUpdates []api.ProgressResponse
		var statusUpdates []string

		req := &api.ImportRequest{
			Path:  exportPath,
			Model: testModelName + "-imported",
		}

		err := client.Import(ctx, req, func(resp api.ProgressResponse) {
			progressUpdates = append(progressUpdates, resp)
			if resp.Status != "" {
				statusUpdates = append(statusUpdates, resp.Status)
			}
		})

		require.NoError(t, err, "import should succeed")
		require.NotEmpty(t, progressUpdates, "should receive progress updates")
		require.NotEmpty(t, statusUpdates, "should receive status updates")

		// Verify we get meaningful status messages
		statusStr := strings.Join(statusUpdates, " ")
		require.Contains(t, statusStr, "importing", "should contain importing status")
		require.Contains(t, statusStr, "complete", "should contain completion status")

		// Cleanup imported model
		client.Delete(ctx, &api.DeleteRequest{Model: testModelName + "-imported"})
	})
}

func TestExportImportWithDifferentNames(t *testing.T) {
	ctx := context.Background()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Create a test model
	originalModelName := "original-model-name"
	CreateTestModel(t, client, originalModelName)
	defer client.Delete(ctx, &api.DeleteRequest{Model: originalModelName})

	tempDir := CreateTempDir(t)
	exportPath := filepath.Join(tempDir, "renamed-export.tar")

	// Export the model
	ExportTestModel(t, client, originalModelName, exportPath)

	// Import with a different name
	newModelName := "imported-with-new-name"
	ImportTestModel(t, client, exportPath, newModelName)
	defer client.Delete(ctx, &api.DeleteRequest{Model: newModelName})

	// Verify both models exist and work
	VerifyModelIntegrity(t, client, originalModelName)
	VerifyModelIntegrity(t, client, newModelName)

	// Verify they are separate models
	originalShow, err := client.Show(ctx, &api.ShowRequest{Model: originalModelName})
	require.NoError(t, err)

	newShow, err := client.Show(ctx, &api.ShowRequest{Model: newModelName})
	require.NoError(t, err)

	// They should have the same modelfile content but different names
	require.Equal(t, originalShow.Modelfile, newShow.Modelfile, "modelfiles should be identical")
}

func TestExportImportLargeModel(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large model test in short mode")
	}

	ctx := context.Background()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	// Use a larger model for this test
	largeModelName := smol // Using the smallest available model for CI
	err := PullIfMissing(ctx, client, largeModelName)
	require.NoError(t, err, "should be able to pull large model")

	tempDir := CreateTempDir(t)
	exportPath := filepath.Join(tempDir, "large-model-export.tar.zst")

	// Export with compression to save space
	req := &api.ExportRequest{
		Model:    largeModelName,
		Path:     exportPath,
		Compress: "zstd",
	}

	// Use a longer timeout for large models
	ctxWithTimeout, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()

	err = client.Export(ctxWithTimeout, req, func(resp api.ProgressResponse) {
		t.Logf("Export progress: %s (completed: %d, total: %d)", resp.Status, resp.Completed, resp.Total)
	})
	require.NoError(t, err, "should be able to export large model")

	// Verify export exists and has reasonable size
	stat, err := os.Stat(exportPath)
	require.NoError(t, err, "export should exist")
	require.Greater(t, stat.Size(), int64(1024*1024), "export should be at least 1MB")

	// Delete the original model
	err = client.Delete(ctx, &api.DeleteRequest{Model: largeModelName})
	require.NoError(t, err, "should be able to delete original model")

	// Import it back
	importReq := &api.ImportRequest{
		Path:  exportPath,
		Model: largeModelName + "-imported",
	}

	err = client.Import(ctxWithTimeout, importReq, func(resp api.ProgressResponse) {
		t.Logf("Import progress: %s (completed: %d, total: %d)", resp.Status, resp.Completed, resp.Total)
	})
	require.NoError(t, err, "should be able to import large model")

	// Verify the imported model works
	VerifyModelIntegrity(t, client, largeModelName+"-imported")

	// Cleanup
	client.Delete(ctx, &api.DeleteRequest{Model: largeModelName + "-imported"})
}
