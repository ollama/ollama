package imagegen

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

// DefaultBlobDir returns the default blob storage directory.
func DefaultBlobDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(home, ".ollama", "models", "blobs")
	case "linux":
		return filepath.Join(home, ".ollama", "models", "blobs")
	case "windows":
		return filepath.Join(home, ".ollama", "models", "blobs")
	default:
		return filepath.Join(home, ".ollama", "models", "blobs")
	}
}

// DefaultManifestDir returns the default manifest storage directory.
func DefaultManifestDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".ollama", "models", "manifests")
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

// GetTensorLayers returns all tensor layers for a given component.
// Component should be "text_encoder", "transformer", or "vae".
// Tensor names are path-style: "component/tensor_name" (e.g., "text_encoder/model.embed_tokens.weight").
func (m *ModelManifest) GetTensorLayers(component string) []ManifestLayer {
	prefix := component + "/"
	var layers []ManifestLayer
	for _, layer := range m.Manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.tensor" && strings.HasPrefix(layer.Name, prefix) {
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
