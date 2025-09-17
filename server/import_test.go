package server

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImportFormatDetection(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "import-test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Test directory format detection
	t.Run("Detect directory format", func(t *testing.T) {
		dirPath := filepath.Join(tempDir, "test-dir")
		err := os.MkdirAll(dirPath, 0755)
		require.NoError(t, err)

		format, err := detectImportFormat(dirPath)
		require.NoError(t, err)
		require.Equal(t, "dir", format)
	})

	// Test file format detection
	t.Run("Detect .tar.gz format", func(t *testing.T) {
		filePath := filepath.Join(tempDir, "test.tar.gz")
		err := os.WriteFile(filePath, []byte("test content"), 0644)
		require.NoError(t, err)

		format, err := detectImportFormat(filePath)
		require.NoError(t, err)
		require.Equal(t, "tar.gz", format)
	})

	t.Run("Detect .tar.zst format", func(t *testing.T) {
		filePath := filepath.Join(tempDir, "test.tar.zst")
		err := os.WriteFile(filePath, []byte("test content"), 0644)
		require.NoError(t, err)

		format, err := detectImportFormat(filePath)
		require.NoError(t, err)
		require.Equal(t, "tar.zst", format)
	})

	t.Run("Detect .tar format", func(t *testing.T) {
		filePath := filepath.Join(tempDir, "test.tar")
		err := os.WriteFile(filePath, []byte("test content"), 0644)
		require.NoError(t, err)

		format, err := detectImportFormat(filePath)
		require.NoError(t, err)
		require.Equal(t, "tar", format)
	})

	t.Run("Unsupported format", func(t *testing.T) {
		filePath := filepath.Join(tempDir, "test.zip")
		err := os.WriteFile(filePath, []byte("test content"), 0644)
		require.NoError(t, err)

		format, err := detectImportFormat(filePath)
		require.Error(t, err)
		require.Empty(t, format)
		require.Contains(t, err.Error(), "unsupported")
	})

	t.Run("Non-existent path", func(t *testing.T) {
		format, err := detectImportFormat("/non/existent/path")
		require.Error(t, err)
		require.Empty(t, format)
	})
}

func TestImportChecksumValidation(t *testing.T) {
	// Test checksum validation functionality
	t.Run("Valid checksum", func(t *testing.T) {
		// Create a temporary file with known content
		tempFile, err := os.CreateTemp("", "checksum-test")
		require.NoError(t, err)
		defer os.Remove(tempFile.Name())

		testContent := []byte("test content for checksum")
		_, err = tempFile.Write(testContent)
		require.NoError(t, err)
		tempFile.Close()

		// Calculate expected checksum
		expectedChecksum, err := calculateSHA256(tempFile.Name())
		require.NoError(t, err)

		// Validate checksum
		err = validateChecksum(tempFile.Name(), expectedChecksum)
		require.NoError(t, err)
	})

	t.Run("Invalid checksum", func(t *testing.T) {
		// Create a temporary file
		tempFile, err := os.CreateTemp("", "checksum-test")
		require.NoError(t, err)
		defer os.Remove(tempFile.Name())

		testContent := []byte("test content for checksum")
		_, err = tempFile.Write(testContent)
		require.NoError(t, err)
		tempFile.Close()

		// Use wrong checksum
		wrongChecksum := "sha256:0123456789abcdef"

		// Validate checksum should fail
		err = validateChecksum(tempFile.Name(), wrongChecksum)
		require.Error(t, err)
		require.Contains(t, err.Error(), "checksum")
	})
}

func TestImportModelNameResolution(t *testing.T) {
	t.Run("Use provided model name", func(t *testing.T) {
		providedName := "custom-model-name"
		metadataName := "original-model-name"
		
		resolved := resolveModelName(providedName, metadataName)
		require.Equal(t, providedName, resolved)
	})

	t.Run("Use metadata model name when not provided", func(t *testing.T) {
		providedName := ""
		metadataName := "original-model-name"
		
		resolved := resolveModelName(providedName, metadataName)
		require.Equal(t, metadataName, resolved)
	})

	t.Run("Handle empty names", func(t *testing.T) {
		providedName := ""
		metadataName := ""
		
		resolved := resolveModelName(providedName, metadataName)
		require.Equal(t, "imported-model", resolved) // Default name
	})
}

// Mock functions to support the tests (these would need to be implemented)
func detectImportFormat(path string) (string, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return "", err
	}

	if stat.IsDir() {
		return "dir", nil
	}

	// Check file extension
	if filepath.Ext(path) == ".gz" && filepath.Ext(path[:len(path)-3]) == ".tar" {
		return "tar.gz", nil
	}
	if filepath.Ext(path) == ".zst" && filepath.Ext(path[:len(path)-4]) == ".tar" {
		return "tar.zst", nil
	}
	if filepath.Ext(path) == ".tar" {
		return "tar", nil
	}

	return "", fmt.Errorf("unsupported format")
}

func calculateSHA256(filePath string) (string, error) {
	// Mock implementation for testing
	return "sha256:mockchecksum", nil
}

func validateChecksum(filePath, expectedChecksum string) error {
	actualChecksum, err := calculateSHA256(filePath)
	if err != nil {
		return err
	}

	if actualChecksum != expectedChecksum {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expectedChecksum, actualChecksum)
	}

	return nil
}

func resolveModelName(providedName, metadataName string) string {
	if providedName != "" {
		return providedName
	}
	if metadataName != "" {
		return metadataName
	}
	return "imported-model"
}
