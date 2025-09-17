package server

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// ImportModel imports a model from a file or directory
func ImportModel(req *api.ImportRequest, fn func(api.ProgressResponse)) error {
	// Check if source exists
	stat, err := os.Stat(req.Path)
	if err != nil {
		return fmt.Errorf("import source not found: %w", err)
	}

	// Detect import format
	var format string
	if stat.IsDir() {
		format = "dir"
	} else if strings.HasSuffix(req.Path, ".tar.gz") || strings.HasSuffix(req.Path, ".tgz") {
		format = "tar.gz"
	} else if strings.HasSuffix(req.Path, ".tar.zst") || strings.HasSuffix(req.Path, ".tzst") {
		format = "tar.zst"
	} else if strings.HasSuffix(req.Path, ".tar") {
		format = "tar"
	} else {
		return fmt.Errorf("unsupported import format")
	}

	fn(api.ProgressResponse{
		Status: "validating import package",
	})

	// Import based on format
	switch format {
	case "dir":
		return importFromDirectory(req, fn)
	case "tar":
		return importFromTar(req, fn, "")
	case "tar.gz":
		return importFromTar(req, fn, "gzip")
	case "tar.zst":
		return importFromTar(req, fn, "zstd")
	default:
		return fmt.Errorf("unsupported import format: %s", format)
	}
}

func importFromDirectory(req *api.ImportRequest, fn func(api.ProgressResponse)) error {
	// Read metadata
	metadataPath := filepath.Join(req.Path, "metadata.json")
	metadataData, err := os.ReadFile(metadataPath)
	if err != nil {
		return fmt.Errorf("failed to read metadata: %w", err)
	}

	var metadata ExportMetadata
	if err := json.Unmarshal(metadataData, &metadata); err != nil {
		return fmt.Errorf("failed to parse metadata: %w", err)
	}

	// Read manifest
	manifestPath := filepath.Join(req.Path, "manifest.json")
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return fmt.Errorf("failed to read manifest: %w", err)
	}

	var manifest Manifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return fmt.Errorf("failed to parse manifest: %w", err)
	}

	// Determine target model name
	modelName := req.Model
	if modelName == "" {
		modelName = metadata.Model
	}

	name := model.ParseName(modelName)

	// Check if model already exists
	if !req.Force {
		if _, err := ParseNamedManifest(name); err == nil {
			return fmt.Errorf("model %s already exists, use --force to overwrite", modelName)
		}
	}

	// Calculate total size for progress
	var totalSize int64
	for _, layer := range manifest.Layers {
		totalSize += layer.Size
	}
	if manifest.Config.Size > 0 {
		totalSize += manifest.Config.Size
	}

	fn(api.ProgressResponse{
		Status: fmt.Sprintf("importing %s", modelName),
		Total:  totalSize,
	})

	var completed int64
	blobsDir := filepath.Join(req.Path, "blobs")

	// Import config layer if present
	if manifest.Config.Digest != "" {
		if err := importBlob(manifest.Config, blobsDir, !req.Insecure, fn, &completed, totalSize); err != nil {
			return fmt.Errorf("failed to import config: %w", err)
		}
	}

	// Import all layers
	for _, layer := range manifest.Layers {
		if err := importBlob(layer, blobsDir, !req.Insecure, fn, &completed, totalSize); err != nil {
			return fmt.Errorf("failed to import layer %s: %w", layer.Digest, err)
		}
	}

	// Save manifest
	if err := WriteManifest(name, manifest.Config, manifest.Layers); err != nil {
		return fmt.Errorf("failed to save manifest: %w", err)
	}

	fn(api.ProgressResponse{
		Status:    "import complete",
		Completed: totalSize,
		Total:     totalSize,
	})

	return nil
}

func importFromTar(req *api.ImportRequest, fn func(api.ProgressResponse), compressionType string) error {
	file, err := os.Open(req.Path)
	if err != nil {
		return fmt.Errorf("failed to open import file: %w", err)
	}
	defer file.Close()

	var reader io.Reader = file
	switch compressionType {
	case "gzip":
		gzReader, err := gzip.NewReader(file)
		if err != nil {
			return fmt.Errorf("failed to create gzip reader: %w", err)
		}
		defer gzReader.Close()
		reader = gzReader
	case "zstd":
		zstdReader, err := zstd.NewReader(file)
		if err != nil {
			return fmt.Errorf("failed to create zstd reader: %w", err)
		}
		defer zstdReader.Close()
		reader = zstdReader
	}

	tarReader := tar.NewReader(reader)

	// First pass: read metadata and manifest
	var metadata ExportMetadata
	var manifest Manifest
	tempBlobs := make(map[string]string) // digest -> temp file path

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar: %w", err)
		}

		switch header.Name {
		case "metadata.json":
			data, err := io.ReadAll(tarReader)
			if err != nil {
				return fmt.Errorf("failed to read metadata from tar: %w", err)
			}
			if err := json.Unmarshal(data, &metadata); err != nil {
				return fmt.Errorf("failed to parse metadata: %w", err)
			}

		case "manifest.json":
			data, err := io.ReadAll(tarReader)
			if err != nil {
				return fmt.Errorf("failed to read manifest from tar: %w", err)
			}
			if err := json.Unmarshal(data, &manifest); err != nil {
				return fmt.Errorf("failed to parse manifest: %w", err)
			}

		default:
			// Handle blobs - extract to temp files for now
			if strings.HasPrefix(header.Name, "blobs/") {
				digest := strings.TrimPrefix(header.Name, "blobs/sha256-")
				tempFile, err := os.CreateTemp("", "ollama-import-*.blob")
				if err != nil {
					return fmt.Errorf("failed to create temp file: %w", err)
				}
				
				if _, err := io.Copy(tempFile, tarReader); err != nil {
					tempFile.Close()
					os.Remove(tempFile.Name())
					return fmt.Errorf("failed to extract blob: %w", err)
				}
				tempFile.Close()
				tempBlobs[digest] = tempFile.Name()
			}
		}
	}

	// Clean up temp files on exit
	defer func() {
		for _, path := range tempBlobs {
			os.Remove(path)
		}
	}()

	// Determine target model name
	modelName := req.Model
	if modelName == "" {
		modelName = metadata.Model
	}

	name := model.ParseName(modelName)

	// Check if model already exists
	if !req.Force {
		if _, err := ParseNamedManifest(name); err == nil {
			return fmt.Errorf("model %s already exists, use --force to overwrite", modelName)
		}
	}

	// Calculate total size for progress
	var totalSize int64
	for _, layer := range manifest.Layers {
		totalSize += layer.Size
	}
	if manifest.Config.Size > 0 {
		totalSize += manifest.Config.Size
	}

	fn(api.ProgressResponse{
		Status: fmt.Sprintf("importing %s", modelName),
		Total:  totalSize,
	})

	var completed int64

	// Import config layer if present
	if manifest.Config.Digest != "" {
		digest := strings.TrimPrefix(manifest.Config.Digest, "sha256:")
		if tempPath, ok := tempBlobs[digest]; ok {
			if err := importBlobFromFile(manifest.Config, tempPath, !req.Insecure, fn, &completed, totalSize); err != nil {
				return fmt.Errorf("failed to import config: %w", err)
			}
		}
	}

	// Import all layers
	for _, layer := range manifest.Layers {
		digest := strings.TrimPrefix(layer.Digest, "sha256:")
		if tempPath, ok := tempBlobs[digest]; ok {
			if err := importBlobFromFile(layer, tempPath, !req.Insecure, fn, &completed, totalSize); err != nil {
				return fmt.Errorf("failed to import layer %s: %w", layer.Digest, err)
			}
		}
	}

	// Save manifest
	if err := WriteManifest(name, manifest.Config, manifest.Layers); err != nil {
		return fmt.Errorf("failed to save manifest: %w", err)
	}

	fn(api.ProgressResponse{
		Status:    "import complete",
		Completed: totalSize,
		Total:     totalSize,
	})

	return nil
}

func importBlob(layer Layer, sourceDir string, verify bool, fn func(api.ProgressResponse), completed *int64, totalSize int64) error {
	digest := strings.TrimPrefix(layer.Digest, "sha256:")
	srcPath := filepath.Join(sourceDir, fmt.Sprintf("sha256-%s", digest))
	
	return importBlobFromFile(layer, srcPath, verify, fn, completed, totalSize)
}

func importBlobFromFile(layer Layer, srcPath string, verify bool, fn func(api.ProgressResponse), completed *int64, totalSize int64) error {
	digest := strings.TrimPrefix(layer.Digest, "sha256:")
	destPath := filepath.Join(envconfig.Models(), "blobs", fmt.Sprintf("sha256-%s", digest))

	// Check if blob already exists
	if _, err := os.Stat(destPath); err == nil {
		*completed += layer.Size
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("skipping existing %s", layer.Digest[7:19]),
			Completed: *completed,
			Total:     totalSize,
		})
		return nil
	}

	// Ensure blob directory exists
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create blob directory: %w", err)
	}

	src, err := os.Open(srcPath)
	if err != nil {
		return fmt.Errorf("failed to open source blob: %w", err)
	}
	defer src.Close()

	// Create temporary destination
	tempDest, err := os.CreateTemp(filepath.Dir(destPath), ".importing-")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tempPath := tempDest.Name()
	defer os.Remove(tempPath)

	// Copy with verification
	hasher := sha256.New()
	writer := io.Writer(tempDest)
	if verify {
		writer = io.MultiWriter(tempDest, hasher)
	}

	buf := make([]byte, 1024*1024) // 1MB buffer
	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, err := writer.Write(buf[:n]); err != nil {
				tempDest.Close()
				return fmt.Errorf("failed to write blob: %w", err)
			}
			*completed += int64(n)
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("importing %s", layer.Digest[7:19]),
				Completed: *completed,
				Total:     totalSize,
			})
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			tempDest.Close()
			return fmt.Errorf("failed to read blob: %w", err)
		}
	}

	tempDest.Close()

	// Verify checksum
	if verify {
		calculatedDigest := hex.EncodeToString(hasher.Sum(nil))
		if calculatedDigest != digest {
			return fmt.Errorf("checksum mismatch for %s: expected %s, got %s", layer.Digest, digest, calculatedDigest)
		}
	}

	// Move to final location
	if err := os.Rename(tempPath, destPath); err != nil {
		return fmt.Errorf("failed to move blob to final location: %w", err)
	}

	return nil
}