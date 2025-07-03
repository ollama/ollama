package server

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/api"
)

// exportToTarStreaming exports model to tar with optimized streaming (no memory buffering)
func exportToTarStreaming(destPath string, mp ModelPath, manifest *Manifest, metadata ExportMetadata, fn func(api.ProgressResponse), totalSize int64) error {
	// Create output file
	file, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create export file: %w", err)
	}
	defer file.Close()

	tarWriter := tar.NewWriter(file)
	defer tarWriter.Close()

	// Write metadata
	metadataData, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}
	if err := writeTarFile(tarWriter, "metadata.json", metadataData); err != nil {
		return fmt.Errorf("failed to write metadata to tar: %w", err)
	}

	// Write manifest
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}
	if err := writeTarFile(tarWriter, "manifest.json", manifestData); err != nil {
		return fmt.Errorf("failed to write manifest to tar: %w", err)
	}

	var completed int64
	lastProgressUpdate := time.Now()

	// Helper to write blob with optimized I/O
	writeBlob := func(layer Layer) error {
		digest := strings.TrimPrefix(layer.Digest, "sha256:")
		srcPath := layer.GetBlobsPath(digest)

		file, err := os.Open(srcPath)
		if err != nil {
			return fmt.Errorf("failed to open blob: %w", err)
		}
		defer file.Close()

		stat, err := file.Stat()
		if err != nil {
			return fmt.Errorf("failed to stat blob: %w", err)
		}

		hdr := &tar.Header{
			Name: filepath.Join("blobs", fmt.Sprintf("sha256-%s", digest)),
			Mode: 0644,
			Size: stat.Size(),
		}
		if err := tarWriter.WriteHeader(hdr); err != nil {
			return err
		}

		// Use io.CopyBuffer with large buffer for optimal performance
		buf := make([]byte, 128*1024*1024) // 128MB buffer
		written, err := io.CopyBuffer(tarWriter, file, buf)
		if err != nil {
			return fmt.Errorf("failed to write blob to tar: %w", err)
		}

		completed += written

		// Update progress (throttled)
		if time.Since(lastProgressUpdate) > 100*time.Millisecond {
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("exporting %s", layer.Digest[7:19]),
				Completed: completed,
				Total:     totalSize,
			})
			lastProgressUpdate = time.Now()
		}

		return nil
	}

	// Write config layer if present
	if manifest.Config.Digest != "" {
		if err := writeBlob(manifest.Config); err != nil {
			return fmt.Errorf("failed to write config to tar: %w", err)
		}
	}

	// Write all layers
	for _, layer := range manifest.Layers {
		if err := writeBlob(layer); err != nil {
			return fmt.Errorf("failed to write layer %s to tar: %w", layer.Digest, err)
		}
	}

	fn(api.ProgressResponse{
		Status:    "export complete",
		Completed: totalSize,
		Total:     totalSize,
	})

	return nil
}

// exportToTarStreamingCompressed exports model to compressed tar with streaming
func exportToTarStreamingCompressed(destPath string, mp ModelPath, manifest *Manifest, metadata ExportMetadata, fn func(api.ProgressResponse), totalSize int64, compressionType string, compressionLevel int) error {
	// Create output file
	file, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create export file: %w", err)
	}
	defer file.Close()

	// Set up compression writer
	var writer io.Writer = file
	var closeFunc func() error

	switch compressionType {
	case "gzip":
		gzWriter := gzip.NewWriter(file)
		writer = gzWriter
		closeFunc = gzWriter.Close
	case "zstd":
		level := zstd.SpeedDefault
		if compressionLevel > 0 {
			level = zstd.EncoderLevelFromZstd(compressionLevel)
		}
		zstdWriter, err := zstd.NewWriter(file, zstd.WithEncoderLevel(level))
		if err != nil {
			return fmt.Errorf("failed to create zstd writer: %w", err)
		}
		writer = zstdWriter
		closeFunc = zstdWriter.Close
	default:
		return fmt.Errorf("unsupported compression type: %s", compressionType)
	}

	if closeFunc != nil {
		defer closeFunc()
	}

	tarWriter := tar.NewWriter(writer)
	defer tarWriter.Close()

	// Write metadata
	metadataData, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}
	if err := writeTarFile(tarWriter, "metadata.json", metadataData); err != nil {
		return fmt.Errorf("failed to write metadata to tar: %w", err)
	}

	// Write manifest
	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}
	if err := writeTarFile(tarWriter, "manifest.json", manifestData); err != nil {
		return fmt.Errorf("failed to write manifest to tar: %w", err)
	}

	var completed int64
	lastProgressUpdate := time.Now()

	// Helper to write blob with streaming
	writeBlob := func(layer Layer) error {
		digest := strings.TrimPrefix(layer.Digest, "sha256:")
		srcPath := layer.GetBlobsPath(digest)

		file, err := os.Open(srcPath)
		if err != nil {
			return fmt.Errorf("failed to open blob: %w", err)
		}
		defer file.Close()

		stat, err := file.Stat()
		if err != nil {
			return fmt.Errorf("failed to stat blob: %w", err)
		}

		hdr := &tar.Header{
			Name: filepath.Join("blobs", fmt.Sprintf("sha256-%s", digest)),
			Mode: 0644,
			Size: stat.Size(),
		}
		if err := tarWriter.WriteHeader(hdr); err != nil {
			return err
		}

		// Stream directly through compression
		// Use smaller buffer for compressed streams to balance memory and performance
		buf := make([]byte, 32*1024*1024) // 32MB buffer for compressed
		written, err := io.CopyBuffer(tarWriter, file, buf)
		if err != nil {
			return fmt.Errorf("failed to write blob to tar: %w", err)
		}

		completed += written

		// Update progress (throttled)
		if time.Since(lastProgressUpdate) > 100*time.Millisecond {
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("compressing %s", layer.Digest[7:19]),
				Completed: completed,
				Total:     totalSize,
			})
			lastProgressUpdate = time.Now()
		}

		return nil
	}

	// Write config layer if present
	if manifest.Config.Digest != "" {
		if err := writeBlob(manifest.Config); err != nil {
			return fmt.Errorf("failed to write config to tar: %w", err)
		}
	}

	// Write all layers
	for _, layer := range manifest.Layers {
		if err := writeBlob(layer); err != nil {
			return fmt.Errorf("failed to write layer %s to tar: %w", layer.Digest, err)
		}
	}

	fn(api.ProgressResponse{
		Status:    "export complete",
		Completed: totalSize,
		Total:     totalSize,
	})

	return nil
}