package imagegen

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// IsImageGenModelDir checks if the directory contains an image generation model
// by looking for model_index.json, which is the standard diffusers pipeline config.
func IsImageGenModelDir(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "model_index.json"))
	return err == nil
}

// LayerInfo holds metadata for a created layer.
type LayerInfo struct {
	Digest    string
	Size      int64
	MediaType string
	Name      string  // Path-style name: "component/tensor" or "path/to/config.json"
	Dtype     string  // Tensor dtype (for tensor layers)
	Shape     []int32 // Tensor shape (for tensor layers)
}

// LayerCreator is called to create a blob layer.
// name is the path-style name (e.g., "tokenizer/tokenizer.json")
type LayerCreator func(r io.Reader, mediaType, name string) (LayerInfo, error)

// TensorLayerCreator creates a tensor blob layer with metadata.
// name is the path-style name including component (e.g., "text_encoder/model.embed_tokens.weight")
type TensorLayerCreator func(r io.Reader, name, dtype string, shape []int32) (LayerInfo, error)

// ManifestWriter writes the manifest file.
type ManifestWriter func(modelName string, config LayerInfo, layers []LayerInfo) error

// CreateModel imports an image generation model from a directory.
// Stores each tensor as a separate blob for fine-grained deduplication.
// Layer creation and manifest writing are done via callbacks to avoid import cycles.
func CreateModel(modelName, modelDir string, createLayer LayerCreator, createTensorLayer TensorLayerCreator, writeManifest ManifestWriter, fn func(status string)) error {
	var layers []LayerInfo
	var configLayer LayerInfo

	// Components to process - extract individual tensors from each
	components := []string{"text_encoder", "transformer", "vae"}

	for _, component := range components {
		componentDir := filepath.Join(modelDir, component)
		if _, err := os.Stat(componentDir); os.IsNotExist(err) {
			continue
		}

		// Find all safetensors files in this component
		entries, err := os.ReadDir(componentDir)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", component, err)
		}

		for _, entry := range entries {
			if !strings.HasSuffix(entry.Name(), ".safetensors") {
				continue
			}

			stPath := filepath.Join(componentDir, entry.Name())

			// Extract individual tensors from safetensors file
			extractor, err := safetensors.OpenForExtraction(stPath)
			if err != nil {
				return fmt.Errorf("failed to open %s: %w", stPath, err)
			}

			tensorNames := extractor.ListTensors()
			fn(fmt.Sprintf("importing %s/%s (%d tensors)", component, entry.Name(), len(tensorNames)))

			for _, tensorName := range tensorNames {
				td, err := extractor.GetTensor(tensorName)
				if err != nil {
					extractor.Close()
					return fmt.Errorf("failed to get tensor %s: %w", tensorName, err)
				}

				// Store as minimal safetensors format (88 bytes header overhead)
				// This enables native mmap loading via mlx_load_safetensors
				// Use path-style name: "component/tensor_name"
				fullName := component + "/" + tensorName
				layer, err := createTensorLayer(td.SafetensorsReader(), fullName, td.Dtype, td.Shape)
				if err != nil {
					extractor.Close()
					return fmt.Errorf("failed to create layer for %s: %w", fullName, err)
				}
				layers = append(layers, layer)
			}

			extractor.Close()
		}
	}

	// Import config files
	configFiles := []string{
		"model_index.json",
		"text_encoder/config.json",
		"text_encoder/generation_config.json",
		"transformer/config.json",
		"vae/config.json",
		"scheduler/scheduler_config.json",
		"tokenizer/tokenizer.json",
		"tokenizer/tokenizer_config.json",
		"tokenizer/vocab.json",
	}

	for _, cfgPath := range configFiles {
		fullPath := filepath.Join(modelDir, cfgPath)
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			continue
		}

		fn(fmt.Sprintf("importing config %s", cfgPath))

		var r io.Reader

		// For model_index.json, normalize to Ollama format
		if cfgPath == "model_index.json" {
			data, err := os.ReadFile(fullPath)
			if err != nil {
				return fmt.Errorf("failed to read %s: %w", cfgPath, err)
			}

			var cfg map[string]any
			if err := json.Unmarshal(data, &cfg); err != nil {
				return fmt.Errorf("failed to parse %s: %w", cfgPath, err)
			}

			// Rename _class_name to architecture, remove diffusers-specific fields
			if className, ok := cfg["_class_name"]; ok {
				cfg["architecture"] = className
				delete(cfg, "_class_name")
			}
			delete(cfg, "_diffusers_version")

			data, err = json.MarshalIndent(cfg, "", "    ")
			if err != nil {
				return fmt.Errorf("failed to marshal %s: %w", cfgPath, err)
			}
			r = bytes.NewReader(data)
		} else {
			f, err := os.Open(fullPath)
			if err != nil {
				return fmt.Errorf("failed to open %s: %w", cfgPath, err)
			}
			defer f.Close()
			r = f
		}

		layer, err := createLayer(r, "application/vnd.ollama.image.json", cfgPath)
		if err != nil {
			return fmt.Errorf("failed to create layer for %s: %w", cfgPath, err)
		}

		// Use model_index.json as the config layer
		if cfgPath == "model_index.json" {
			configLayer = layer
		}

		layers = append(layers, layer)
	}

	if configLayer.Digest == "" {
		return fmt.Errorf("model_index.json not found in %s", modelDir)
	}

	fn(fmt.Sprintf("writing manifest for %s", modelName))

	if err := writeManifest(modelName, configLayer, layers); err != nil {
		return fmt.Errorf("failed to write manifest: %w", err)
	}

	fn(fmt.Sprintf("successfully imported %s with %d layers", modelName, len(layers)))
	return nil
}
