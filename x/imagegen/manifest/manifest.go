package manifest

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// ManifestLayer represents a layer in the manifest.
type ManifestLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	Name      string `json:"name,omitempty"` // Path-style name: "component/tensor" or "path/to/config.json"
}

// Manifest represents the manifest JSON structure.
type Manifest struct {
	SchemaVersion int             `json:"schemaVersion"`
	MediaType     string          `json:"mediaType"`
	Config        ManifestLayer   `json:"config"`
	Layers        []ManifestLayer `json:"layers"`
}

// ModelManifest holds a parsed manifest with helper methods.
type ModelManifest struct {
	Manifest *Manifest
	BlobDir  string
}

func DefaultBlobDir() string {
	return filepath.Join(envconfig.Models(), "blobs")
}

// DefaultManifestDir returns the manifest storage directory.
// Respects OLLAMA_MODELS.

func DefaultManifestDir() string {
	return filepath.Join(envconfig.Models(), "manifests")
}

// LoadManifest loads a manifest for the given model name.
// Model name format: "modelname" or "modelname:tag" or "host/namespace/name:tag"
func LoadManifest(modelName string) (*ModelManifest, error) {
	manifestPath := resolveManifestPath(modelName)

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("read manifest: %w", err)
	}

	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("parse manifest: %w", err)
	}

	return &ModelManifest{
		Manifest: &manifest,
		BlobDir:  DefaultBlobDir(),
	}, nil
}

// resolveManifestPath converts a model name to a manifest file path.
func resolveManifestPath(modelName string) string {
	// Parse model name into components
	// Default: registry.ollama.ai/library/<name>/<tag>
	host := "registry.ollama.ai"
	namespace := "library"
	name := modelName
	tag := "latest"

	// Handle explicit tag
	if idx := strings.LastIndex(name, ":"); idx != -1 {
		tag = name[idx+1:]
		name = name[:idx]
	}

	// Handle full path like "host/namespace/name"
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 3:
		host = parts[0]
		namespace = parts[1]
		name = parts[2]
	case 2:
		namespace = parts[0]
		name = parts[1]
	}

	return filepath.Join(DefaultManifestDir(), host, namespace, name, tag)
}

// BlobPath returns the full path to a blob given its digest.
func (m *ModelManifest) BlobPath(digest string) string {
	// Convert "sha256:abc123" to "sha256-abc123"
	blobName := strings.Replace(digest, ":", "-", 1)
	return filepath.Join(m.BlobDir, blobName)
}

// GetTensorLayers returns tensor layers, optionally filtered by component.
// If component is empty, returns all tensor layers (for LLM models).
// If component is specified (e.g., "text_encoder", "transformer", "vae"),
// returns only layers with that prefix.
func (m *ModelManifest) GetTensorLayers(component string) []ManifestLayer {
	var layers []ManifestLayer
	for _, layer := range m.Manifest.Layers {
		if layer.MediaType != "application/vnd.ollama.image.tensor" {
			continue
		}
		if component == "" || strings.HasPrefix(layer.Name, component+"/") {
			layers = append(layers, layer)
		}
	}
	return layers
}

// GetConfigLayer returns the config layer for a given path.
func (m *ModelManifest) GetConfigLayer(configPath string) *ManifestLayer {
	for _, layer := range m.Manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.json" && layer.Name == configPath {
			return &layer
		}
	}
	return nil
}

// ReadConfig reads and returns the content of a config file.
func (m *ModelManifest) ReadConfig(configPath string) ([]byte, error) {
	layer := m.GetConfigLayer(configPath)
	if layer == nil {
		return nil, fmt.Errorf("config %q not found in manifest", configPath)
	}

	blobPath := m.BlobPath(layer.Digest)
	return os.ReadFile(blobPath)
}

// ReadConfigJSON reads and unmarshals a config file.
func (m *ModelManifest) ReadConfigJSON(configPath string, v any) error {
	data, err := m.ReadConfig(configPath)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, v)
}

// OpenBlob opens a blob for reading.
func (m *ModelManifest) OpenBlob(digest string) (io.ReadCloser, error) {
	return os.Open(m.BlobPath(digest))
}

// HasTensorLayers returns true if the manifest has any tensor layers.
func (m *ModelManifest) HasTensorLayers() bool {
	for _, layer := range m.Manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.tensor" {
			return true
		}
	}
	return false
}

// TotalTensorSize returns the total size in bytes of all tensor layers.
func (m *ModelManifest) TotalTensorSize() int64 {
	var total int64
	for _, layer := range m.Manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.tensor" {
			total += layer.Size
		}
	}
	return total
}

// ModelInfo contains metadata about an image generation model.
type ModelInfo struct {
	Architecture   string
	ParameterCount int64
	Quantization   string
}

// GetModelInfo returns metadata about an image generation model.
func GetModelInfo(modelName string) (*ModelInfo, error) {
	manifest, err := LoadManifest(modelName)
	if err != nil {
		return nil, fmt.Errorf("failed to load manifest: %w", err)
	}

	info := &ModelInfo{}

	// Read model_index.json for architecture, parameter count, and quantization
	if data, err := manifest.ReadConfig("model_index.json"); err == nil {
		var index struct {
			Architecture   string `json:"architecture"`
			ParameterCount int64  `json:"parameter_count"`
			Quantization   string `json:"quantization"`
		}
		if json.Unmarshal(data, &index) == nil {
			info.Architecture = index.Architecture
			info.ParameterCount = index.ParameterCount
			info.Quantization = index.Quantization
		}
	}

	// Fallback: detect quantization from first tensor blob's __metadata__
	if info.Quantization == "" {
		info.Quantization = detectQuantizationFromBlobs(manifest)
	}
	if info.Quantization == "" {
		info.Quantization = "BF16"
	}

	// Fallback: estimate parameter count if not in config
	if info.ParameterCount == 0 {
		var totalSize int64
		for _, layer := range manifest.Manifest.Layers {
			if layer.MediaType == "application/vnd.ollama.image.tensor" {
				totalSize += layer.Size
			}
		}
		// Assume BF16 (2 bytes/param) as rough estimate
		info.ParameterCount = totalSize / 2
	}

	return info, nil
}

// detectQuantizationFromBlobs reads __metadata__ from the first tensor blob
// to detect quantization type.
func detectQuantizationFromBlobs(manifest *ModelManifest) string {
	for _, layer := range manifest.Manifest.Layers {
		if layer.MediaType != "application/vnd.ollama.image.tensor" {
			continue
		}
		data, err := readBlobHeader(manifest.BlobPath(layer.Digest))
		if err != nil {
			continue
		}
		var header map[string]json.RawMessage
		if json.Unmarshal(data, &header) != nil {
			continue
		}
		if metaRaw, ok := header["__metadata__"]; ok {
			var meta map[string]string
			if json.Unmarshal(metaRaw, &meta) == nil {
				if qt, ok := meta["quant_type"]; ok && qt != "" {
					return strings.ToUpper(qt)
				}
			}
		}
		// Only check the first tensor blob
		break
	}
	return ""
}

// ParseBlobTensorNames reads a safetensors blob and returns all "main" tensor names.
// Filters out __metadata__, .scale, and .bias entries to return only primary weight tensors.
func ParseBlobTensorNames(path string) ([]string, error) {
	data, err := readBlobHeader(path)
	if err != nil {
		return nil, err
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}

	var names []string
	for k := range header {
		if k == "__metadata__" || strings.HasSuffix(k, ".scale") || strings.HasSuffix(k, ".bias") {
			continue
		}
		names = append(names, k)
	}

	sort.Strings(names)
	return names, nil
}

// readBlobHeader reads the JSON header bytes from a safetensors blob file.
func readBlobHeader(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	if headerSize > 1024*1024 {
		return nil, fmt.Errorf("header too large: %d", headerSize)
	}
	data := make([]byte, headerSize)
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}
	return data, nil
}
