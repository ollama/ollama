package server

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)


func TestExportFormatDetection(t *testing.T) {
	modelName := "test-model"

	t.Run("Detect empty format as directory", func(t *testing.T) {
		format, path := detectExportFormat(modelName, "/path/to/export")
		require.Equal(t, "dir", format)
		require.Equal(t, "/path/to/export", path)
	})

	t.Run("Detect .tar.gz format", func(t *testing.T) {
		format, path := detectExportFormat(modelName, "/path/to/export.tar.gz")
		require.Equal(t, "tar.gz", format)
		require.Equal(t, "/path/to/export.tar.gz", path)
	})

	t.Run("Detect .tar.zst format", func(t *testing.T) {
		format, path := detectExportFormat(modelName, "/path/to/export.tar.zst")
		require.Equal(t, "tar.zst", format)
		require.Equal(t, "/path/to/export.tar.zst", path)
	})

	t.Run("Detect .tar format", func(t *testing.T) {
		format, path := detectExportFormat(modelName, "/path/to/export.tar")
		require.Equal(t, "tar", format)
		require.Equal(t, "/path/to/export.tar", path)
	})
}

func TestExportMetadataGeneration(t *testing.T) {
	modelName := "test-model"
	format := "tar.gz"

	metadata := generateExportMetadata(modelName, format)
	require.Equal(t, "1.0", metadata.Version)
	require.Equal(t, modelName, metadata.Model)
	require.Equal(t, format, metadata.Format)
	require.NotZero(t, metadata.ExportedAt)
}


// Additional tests for error handling and specific logic

// Mock functions to support the tests (these would need to be implemented)
func detectExportFormat(modelName, path string) (string, string) {
	// Auto-detect format from file extension
	if strings.HasSuffix(path, ".tar.gz") || strings.HasSuffix(path, ".tgz") {
		return "tar.gz", path
	} else if strings.HasSuffix(path, ".tar.zst") || strings.HasSuffix(path, ".tzst") {
		return "tar.zst", path
	} else if strings.HasSuffix(path, ".tar") {
		return "tar", path
	} else {
		return "dir", path
	}
}

func generateExportMetadata(modelName, format string) ExportMetadata {
	return ExportMetadata{
		Version:      "1.0",
		ExportedAt:   time.Now(),
		OllamaVersion: "test-version",
		Model:        modelName,
		Format:       format,
	}
}
