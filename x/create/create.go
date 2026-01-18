package create

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// ModelConfig represents the config blob stored with a model.
type ModelConfig struct {
	ModelFormat  string   `json:"model_format"`
	Capabilities []string `json:"capabilities"`
}

// Manifest represents the manifest JSON structure.
type Manifest struct {
	SchemaVersion int             `json:"schemaVersion"`
	MediaType     string          `json:"mediaType"`
	Config        ManifestLayer   `json:"config"`
	Layers        []ManifestLayer `json:"layers"`
}

// ManifestLayer represents a layer in the manifest.
type ManifestLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	Name      string `json:"name,omitempty"`
}

// defaultManifestDir returns the manifest storage directory.
func defaultManifestDir() string {
	return filepath.Join(envconfig.Models(), "manifests")
}

// defaultBlobDir returns the blob storage directory.
func defaultBlobDir() string {
	return filepath.Join(envconfig.Models(), "blobs")
}

// resolveManifestPath converts a model name to a manifest file path.
func resolveManifestPath(modelName string) string {
	host := "registry.ollama.ai"
	namespace := "library"
	name := modelName
	tag := "latest"

	if idx := strings.LastIndex(name, ":"); idx != -1 {
		tag = name[idx+1:]
		name = name[:idx]
	}

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

	return filepath.Join(defaultManifestDir(), host, namespace, name, tag)
}

// loadManifest loads a manifest for the given model name.
func loadManifest(modelName string) (*Manifest, error) {
	manifestPath := resolveManifestPath(modelName)

	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}

	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, err
	}

	return &manifest, nil
}

// loadModelConfig loads the config blob for a model.
func loadModelConfig(modelName string) (*ModelConfig, error) {
	manifest, err := loadManifest(modelName)
	if err != nil {
		return nil, err
	}

	// Read the config blob
	blobName := strings.Replace(manifest.Config.Digest, ":", "-", 1)
	blobPath := filepath.Join(defaultBlobDir(), blobName)

	data, err := os.ReadFile(blobPath)
	if err != nil {
		return nil, err
	}

	var config ModelConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

// IsSafetensorsModel checks if a model was created with the experimental
// safetensors builder by checking the model format in the config.
func IsSafetensorsModel(modelName string) bool {
	config, err := loadModelConfig(modelName)
	if err != nil {
		return false
	}
	return config.ModelFormat == "safetensors"
}

// IsSafetensorsLLMModel checks if a model is a safetensors LLM model
// (has completion capability, not image generation).
func IsSafetensorsLLMModel(modelName string) bool {
	config, err := loadModelConfig(modelName)
	if err != nil {
		return false
	}
	return config.ModelFormat == "safetensors" && slices.Contains(config.Capabilities, "completion")
}

// IsImageGenModel checks if a model is an image generation model
// (has image capability).
func IsImageGenModel(modelName string) bool {
	config, err := loadModelConfig(modelName)
	if err != nil {
		return false
	}
	return config.ModelFormat == "safetensors" && slices.Contains(config.Capabilities, "image")
}

// GetModelArchitecture returns the architecture from the model's config.json layer.
func GetModelArchitecture(modelName string) (string, error) {
	manifest, err := loadManifest(modelName)
	if err != nil {
		return "", err
	}

	// Find the config.json layer
	for _, layer := range manifest.Layers {
		if layer.Name == "config.json" && layer.MediaType == "application/vnd.ollama.image.json" {
			blobName := strings.Replace(layer.Digest, ":", "-", 1)
			blobPath := filepath.Join(defaultBlobDir(), blobName)

			data, err := os.ReadFile(blobPath)
			if err != nil {
				return "", err
			}

			var cfg struct {
				Architectures []string `json:"architectures"`
				ModelType     string   `json:"model_type"`
			}
			if err := json.Unmarshal(data, &cfg); err != nil {
				return "", err
			}

			// Prefer model_type, fall back to first architecture
			if cfg.ModelType != "" {
				return cfg.ModelType, nil
			}
			if len(cfg.Architectures) > 0 {
				return cfg.Architectures[0], nil
			}
		}
	}

	return "", fmt.Errorf("architecture not found in model config")
}

// IsTensorModelDir checks if the directory contains a diffusers-style tensor model
// by looking for model_index.json, which is the standard diffusers pipeline config.
func IsTensorModelDir(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "model_index.json"))
	return err == nil
}

// IsSafetensorsModelDir checks if the directory contains a standard safetensors model
// by looking for config.json and at least one .safetensors file.
func IsSafetensorsModelDir(dir string) bool {
	// Must have config.json
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return false
	}

	// Must have at least one .safetensors file
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}

	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".safetensors") {
			return true
		}
	}

	return false
}

// LayerInfo holds metadata for a created layer.
type LayerInfo struct {
	Digest    string
	Size      int64
	MediaType string
	Name      string // Path-style name: "component/tensor" or "path/to/config.json"
}

// LayerCreator is called to create a blob layer.
// name is the path-style name (e.g., "tokenizer/tokenizer.json")
type LayerCreator func(r io.Reader, mediaType, name string) (LayerInfo, error)

// TensorLayerCreator creates a tensor blob layer with metadata.
// name is the path-style name including component (e.g., "text_encoder/model.embed_tokens.weight")
type TensorLayerCreator func(r io.Reader, name, dtype string, shape []int32) (LayerInfo, error)

// QuantizingTensorLayerCreator creates tensor layers with optional quantization.
// When quantize is non-empty (e.g., "fp8"), returns multiple layers (weight + scales + biases).
type QuantizingTensorLayerCreator func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]LayerInfo, error)

// ManifestWriter writes the manifest file.
type ManifestWriter func(modelName string, config LayerInfo, layers []LayerInfo) error

// ShouldQuantize returns true if a tensor should be quantized.
// For image gen models (component non-empty): quantizes linear weights, skipping VAE, embeddings, norms.
// For LLM models (component empty): quantizes linear weights, skipping embeddings, norms, and small tensors.
func ShouldQuantize(name, component string) bool {
	// Image gen specific: skip VAE entirely
	if component == "vae" {
		return false
	}

	// Skip embeddings
	if strings.Contains(name, "embed") {
		return false
	}

	// Skip layer norms and RMS norms
	if strings.Contains(name, "norm") || strings.Contains(name, "ln_") || strings.Contains(name, "layernorm") {
		return false
	}

	// Skip biases
	if strings.HasSuffix(name, ".bias") {
		return false
	}

	// Only quantize weights
	return strings.HasSuffix(name, ".weight")
}

// ShouldQuantizeTensor returns true if a tensor should be quantized based on name and shape.
// This is a more detailed check that also considers tensor dimensions.
func ShouldQuantizeTensor(name string, shape []int32) bool {
	// Use basic name-based check first
	if !ShouldQuantize(name, "") {
		return false
	}

	// Only quantize 2D tensors (linear layers) - skip 1D (biases, norms) and higher-D (convolutions if any)
	if len(shape) != 2 {
		return false
	}

	// Skip small tensors (less than 1024 elements) - not worth quantizing
	if len(shape) >= 2 && int64(shape[0])*int64(shape[1]) < 1024 {
		return false
	}

	// MLX quantization requires last dimension to be divisible by group size (32)
	if shape[len(shape)-1]%32 != 0 {
		return false
	}

	return true
}

// CreateSafetensorsModel imports a standard safetensors model from a directory.
// This handles Hugging Face style models with config.json and *.safetensors files.
// Stores each tensor as a separate blob for fine-grained deduplication.
// If quantize is non-empty (e.g., "fp8"), eligible tensors will be quantized.
func CreateSafetensorsModel(modelName, modelDir, quantize string, createLayer LayerCreator, createTensorLayer QuantizingTensorLayerCreator, writeManifest ManifestWriter, fn func(status string)) error {
	var layers []LayerInfo
	var configLayer LayerInfo

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return fmt.Errorf("failed to read directory: %w", err)
	}

	// Process all safetensors files
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") {
			continue
		}

		stPath := filepath.Join(modelDir, entry.Name())

		// Extract individual tensors from safetensors file
		extractor, err := safetensors.OpenForExtraction(stPath)
		if err != nil {
			return fmt.Errorf("failed to open %s: %w", stPath, err)
		}

		tensorNames := extractor.ListTensors()
		quantizeMsg := ""
		if quantize != "" {
			quantizeMsg = fmt.Sprintf(", quantizing to %s", quantize)
		}
		fn(fmt.Sprintf("importing %s (%d tensors%s)", entry.Name(), len(tensorNames), quantizeMsg))

		for _, tensorName := range tensorNames {
			td, err := extractor.GetTensor(tensorName)
			if err != nil {
				extractor.Close()
				return fmt.Errorf("failed to get tensor %s: %w", tensorName, err)
			}

			// Determine quantization type for this tensor (empty string if not quantizing)
			quantizeType := ""
			if quantize != "" && ShouldQuantizeTensor(tensorName, td.Shape) {
				quantizeType = quantize
			}

			// Store as minimal safetensors format (88 bytes header overhead)
			// This enables native mmap loading via mlx_load_safetensors
			// createTensorLayer returns multiple layers if quantizing (weight + scales)
			newLayers, err := createTensorLayer(td.SafetensorsReader(), tensorName, td.Dtype, td.Shape, quantizeType)
			if err != nil {
				extractor.Close()
				return fmt.Errorf("failed to create layer for %s: %w", tensorName, err)
			}
			layers = append(layers, newLayers...)
		}

		extractor.Close()
	}

	// Process all JSON config files
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		// Skip the index file as we don't need it after extraction
		if entry.Name() == "model.safetensors.index.json" {
			continue
		}

		cfgPath := entry.Name()
		fullPath := filepath.Join(modelDir, cfgPath)

		fn(fmt.Sprintf("importing config %s", cfgPath))

		f, err := os.Open(fullPath)
		if err != nil {
			return fmt.Errorf("failed to open %s: %w", cfgPath, err)
		}

		layer, err := createLayer(f, "application/vnd.ollama.image.json", cfgPath)
		f.Close()
		if err != nil {
			return fmt.Errorf("failed to create layer for %s: %w", cfgPath, err)
		}

		// Use config.json as the config layer
		if cfgPath == "config.json" {
			configLayer = layer
		}

		layers = append(layers, layer)
	}

	if configLayer.Digest == "" {
		return fmt.Errorf("config.json not found in %s", modelDir)
	}

	fn(fmt.Sprintf("writing manifest for %s", modelName))

	if err := writeManifest(modelName, configLayer, layers); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	fn(fmt.Sprintf("successfully imported %s with %d layers", modelName, len(layers)))
	return nil
}
