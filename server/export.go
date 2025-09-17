package server

import (
	"archive/tar"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

// ExportMetadata contains metadata about the export
type ExportMetadata struct {
	Version      string    `json:"version"`
	ExportedAt   time.Time `json:"exported_at"`
	OllamaVersion string    `json:"ollama_version"`
	Model        string    `json:"model"`
	Format       string    `json:"format"`
}

// ExportModel exports a model to a file or directory
func ExportModel(req *api.ExportRequest, fn func(api.ProgressResponse)) error {
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return fmt.Errorf("invalid model name: %s", req.Model)
	}
	
	// Validate model exists
	manifest, err := ParseNamedManifest(name)
	if err != nil {
		return fmt.Errorf("model not found: %w", err)
	}

	// Determine export format and add extension if needed
	format := req.Format
	path := req.Path
	
	if format == "" {
		if strings.HasSuffix(path, ".tar.gz") || strings.HasSuffix(path, ".tgz") {
			format = "tar.gz"
		} else if strings.HasSuffix(path, ".tar.zst") || strings.HasSuffix(path, ".tzst") {
			format = "tar.zst"
		} else if strings.HasSuffix(path, ".tar") {
			format = "tar"
		} else {
			format = "dir"
		}
	}
	
	// Add file extension if missing
	if format != "dir" && !strings.Contains(filepath.Base(path), ".") {
		switch format {
		case "tar":
			path += ".tar"
		case "tar.gz":
			path += ".tar.gz"
		case "tar.zst":
			path += ".tar.zst"
		}
	}

	// Check for existing destination (unless forced)
	if !req.Force {
		if stat, err := os.Stat(path); err == nil {
			if format == "dir" {
				if stat.IsDir() {
					// Check if directory is not empty
					if entries, err := os.ReadDir(path); err == nil && len(entries) > 0 {
						return fmt.Errorf("directory '%s' is not empty, use --force to overwrite", path)
					}
				} else {
					return fmt.Errorf("destination '%s' exists but is not a directory, use --force to overwrite", path)
				}
			} else {
				// File exists
				return fmt.Errorf("file '%s' already exists, use --force to overwrite", path)
			}
		}
	}

	// Create export metadata
	metadata := ExportMetadata{
		Version:       "1.0",
		ExportedAt:    time.Now(),
		OllamaVersion: version.Version,
		Model:         req.Model,
		Format:        format,
	}

	// Calculate total size for progress
	var totalSize int64
	for _, layer := range manifest.Layers {
		totalSize += layer.Size
	}
	if manifest.Config.Size > 0 {
		totalSize += manifest.Config.Size
	}

	// Start export progress
	fn(api.ProgressResponse{
		Status: fmt.Sprintf("exporting %s", req.Model),
		Total:  totalSize,
	})

	switch format {
	case "dir":
		return exportToDirectory(path, name, manifest, metadata, fn, totalSize)
	case "tar":
		// Use streaming export for uncompressed tar to avoid memory issues
		return exportToTarStreaming(path, name, manifest, metadata, fn, totalSize)
	case "tar.gz":
		// Use streaming compressed export
		return exportToTarStreamingCompressed(path, name, manifest, metadata, fn, totalSize, "gzip", 0)
	case "tar.zst":
		level := req.CompressionLevel
		if level == 0 {
			level = 3
		}
		// Use streaming compressed export
		return exportToTarStreamingCompressed(path, name, manifest, metadata, fn, totalSize, "zstd", level)
	default:
		return fmt.Errorf("unsupported export format: %s", format)
	}
}

func exportToDirectory(destPath string, name model.Name, manifest *Manifest, metadata ExportMetadata, fn func(api.ProgressResponse), totalSize int64) error {
	// Create destination directory
	if err := os.MkdirAll(destPath, 0755); err != nil {
		return fmt.Errorf("failed to create export directory: %w", err)
	}

	// Create blobs directory
	blobsDir := filepath.Join(destPath, "blobs")
	if err := os.MkdirAll(blobsDir, 0755); err != nil {
		return fmt.Errorf("failed to create blobs directory: %w", err)
	}

	// Write metadata
	metadataPath := filepath.Join(destPath, "metadata.json")
	metadataData, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}
	if err := os.WriteFile(metadataPath, metadataData, 0644); err != nil {
		return fmt.Errorf("failed to write metadata: %w", err)
	}

	// Write manifest
	manifestPath := filepath.Join(destPath, "manifest.json")
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}
	if err := os.WriteFile(manifestPath, manifestData, 0644); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	var completed int64

	// Copy config layer if present
	if manifest.Config.Digest != "" {
		if err := copyBlobToExport(manifest.Config, blobsDir, fn, &completed, totalSize); err != nil {
			return fmt.Errorf("failed to copy config: %w", err)
		}
	}

	// Copy all layers
	for _, layer := range manifest.Layers {
		if err := copyBlobToExport(layer, blobsDir, fn, &completed, totalSize); err != nil {
			return fmt.Errorf("failed to copy layer %s: %w", layer.Digest, err)
		}
	}

	fn(api.ProgressResponse{
		Status:    "export complete",
		Completed: totalSize,
		Total:     totalSize,
	})

	return nil
}

func copyBlobToExport(layer Layer, destDir string, fn func(api.ProgressResponse), completed *int64, totalSize int64) error {
	digest := strings.TrimPrefix(layer.Digest, "sha256:")
	srcPath, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return fmt.Errorf("failed to get blob path: %w", err)
	}
	destPath := filepath.Join(destDir, fmt.Sprintf("sha256-%s", digest))

	src, err := os.Open(srcPath)
	if err != nil {
		return fmt.Errorf("failed to open source blob: %w", err)
	}
	defer src.Close()

	dest, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination blob: %w", err)
	}
	defer dest.Close()

	// Copy with progress
	buf := make([]byte, 64*1024*1024) // 64MB buffer
	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, err := dest.Write(buf[:n]); err != nil {
				return fmt.Errorf("failed to write blob: %w", err)
			}
			*completed += int64(n)
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("exporting %s", layer.Digest[7:19]),
				Completed: *completed,
				Total:     totalSize,
			})
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read blob: %w", err)
		}
	}

	return nil
}


// Helper function for writing small files to tar
func writeTarFile(tw *tar.Writer, name string, data []byte) error {
	hdr := &tar.Header{
		Name: name,
		Mode: 0644,
		Size: int64(len(data)),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}
	if _, err := tw.Write(data); err != nil {
		return err
	}
	return nil
}