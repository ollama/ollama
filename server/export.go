package server

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
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
	mp := ParseModelPath(req.Model)
	
	// Validate model exists
	manifest, _, err := GetManifest(mp)
	if err != nil {
		return fmt.Errorf("model not found: %w", err)
	}

	// Determine export format
	format := req.Format
	if format == "" {
		if strings.HasSuffix(req.Path, ".tar.gz") || strings.HasSuffix(req.Path, ".tgz") {
			format = "tar.gz"
		} else if strings.HasSuffix(req.Path, ".tar") {
			format = "tar"
		} else {
			format = "dir"
		}
	}

	// Create export metadata
	metadata := ExportMetadata{
		Version:       "1.0",
		ExportedAt:    time.Now(),
		OllamaVersion: "0.5.0", // TODO: Get from version package
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
		return exportToDirectory(req.Path, mp, manifest, metadata, fn, totalSize)
	case "tar":
		// Use optimized parallel export for uncompressed tar
		return exportToTarParallel(req.Path, mp, manifest, metadata, fn, totalSize)
	case "tar.gz":
		return exportToTar(req.Path, mp, manifest, metadata, fn, totalSize, true)
	default:
		return fmt.Errorf("unsupported export format: %s", format)
	}
}

func exportToDirectory(destPath string, mp ModelPath, manifest *Manifest, metadata ExportMetadata, fn func(api.ProgressResponse), totalSize int64) error {
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
			return fmt.Errorf("failed to export config: %w", err)
		}
	}

	// Copy all layers
	for _, layer := range manifest.Layers {
		if err := copyBlobToExport(layer, blobsDir, fn, &completed, totalSize); err != nil {
			return fmt.Errorf("failed to export layer %s: %w", layer.Digest, err)
		}
	}

	fn(api.ProgressResponse{
		Status:    "export complete",
		Completed: totalSize,
		Total:     totalSize,
	})

	return nil
}

func exportToTar(destPath string, mp ModelPath, manifest *Manifest, metadata ExportMetadata, fn func(api.ProgressResponse), totalSize int64, compress bool) error {
	// Create output file
	file, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create export file: %w", err)
	}
	defer file.Close()

	var writer io.Writer = file
	var gzWriter *gzip.Writer
	if compress {
		gzWriter = gzip.NewWriter(file)
		writer = gzWriter
		defer gzWriter.Close()
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

	// Write config layer if present
	if manifest.Config.Digest != "" {
		if err := writeBlobToTar(tarWriter, manifest.Config, fn, &completed, totalSize); err != nil {
			return fmt.Errorf("failed to write config to tar: %w", err)
		}
	}

	// Write all layers
	for _, layer := range manifest.Layers {
		if err := writeBlobToTar(tarWriter, layer, fn, &completed, totalSize); err != nil {
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

func copyBlobToExport(layer Layer, destDir string, fn func(api.ProgressResponse), completed *int64, totalSize int64) error {
	digest := strings.TrimPrefix(layer.Digest, "sha256:")
	srcPath := layer.GetBlobsPath(digest)
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

func writeBlobToTar(tw *tar.Writer, layer Layer, fn func(api.ProgressResponse), completed *int64, totalSize int64) error {
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
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}

	// Copy with progress
	buf := make([]byte, 64*1024*1024) // 64MB buffer
	for {
		n, err := file.Read(buf)
		if n > 0 {
			if _, err := tw.Write(buf[:n]); err != nil {
				return fmt.Errorf("failed to write blob to tar: %w", err)
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

// GetBlobsPath helper for Layer
func (l Layer) GetBlobsPath(digest string) string {
	dir := envconfig.Models()
	return filepath.Join(dir, "blobs", fmt.Sprintf("sha256-%s", digest))
}

// Blob processing result for parallel export
type blobResult struct {
	layer    Layer
	data     []byte
	size     int64
	position int
	err      error
}

// exportToTarParallel exports model to tar with parallel blob reading
func exportToTarParallel(destPath string, mp ModelPath, manifest *Manifest, metadata ExportMetadata, fn func(api.ProgressResponse), totalSize int64) error {
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

	// Prepare all layers for processing
	layers := make([]Layer, 0, len(manifest.Layers)+1)
	if manifest.Config.Digest != "" {
		layers = append(layers, manifest.Config)
	}
	layers = append(layers, manifest.Layers...)

	// Progress tracking
	var completed atomic.Int64
	progressMutex := &sync.Mutex{}
	lastProgressUpdate := time.Now()

	// Worker pool for parallel reading
	numWorkers := 4 // Default to 4 workers
	if envWorkers := os.Getenv("OLLAMA_EXPORT_WORKERS"); envWorkers != "" {
		if n, err := strconv.Atoi(envWorkers); err == nil && n > 0 && n <= 16 {
			numWorkers = n
		}
	}
	jobs := make(chan int, len(layers))
	results := make(chan blobResult, len(layers))

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				layer := layers[idx]
				digest := strings.TrimPrefix(layer.Digest, "sha256:")
				srcPath := layer.GetBlobsPath(digest)

				// Pre-read blob into memory for faster tar writing
				data, err := os.ReadFile(srcPath)
				if err != nil {
					results <- blobResult{layer: layer, position: idx, err: err}
					continue
				}

				results <- blobResult{
					layer:    layer,
					data:     data,
					size:     int64(len(data)),
					position: idx,
					err:      nil,
				}

				// Update progress (throttled)
				progressMutex.Lock()
				completed.Add(int64(len(data)))
				if time.Since(lastProgressUpdate) > 100*time.Millisecond {
					fn(api.ProgressResponse{
						Status:    fmt.Sprintf("reading %s", layer.Digest[7:19]),
						Completed: completed.Load(),
						Total:     totalSize,
					})
					lastProgressUpdate = time.Now()
				}
				progressMutex.Unlock()
			}
		}()
	}

	// Submit jobs
	for i := range layers {
		jobs <- i
	}
	close(jobs)

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect all results
	resultMap := make(map[int]blobResult)
	for result := range results {
		if result.err != nil {
			return fmt.Errorf("failed to read layer %s: %w", result.layer.Digest, result.err)
		}
		resultMap[result.position] = result
	}

	// Write blobs to tar in order
	completed.Store(0)
	for i := range layers {
		result := resultMap[i]
		digest := strings.TrimPrefix(result.layer.Digest, "sha256:")

		hdr := &tar.Header{
			Name: filepath.Join("blobs", fmt.Sprintf("sha256-%s", digest)),
			Mode: 0644,
			Size: result.size,
		}
		if err := tarWriter.WriteHeader(hdr); err != nil {
			return err
		}

		// Write in chunks with progress
		written := 0
		chunkSize := 64 * 1024 * 1024 // 64MB chunks
		for written < len(result.data) {
			end := written + chunkSize
			if end > len(result.data) {
				end = len(result.data)
			}
			
			n, err := tarWriter.Write(result.data[written:end])
			if err != nil {
				return fmt.Errorf("failed to write blob to tar: %w", err)
			}
			written += n
			completed.Add(int64(n))

			// Update progress
			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("writing %s", result.layer.Digest[7:19]),
				Completed: completed.Load(),
				Total:     totalSize,
			})
		}
	}

	fn(api.ProgressResponse{
		Status:    "export complete",
		Completed: totalSize,
		Total:     totalSize,
	})

	return nil
}