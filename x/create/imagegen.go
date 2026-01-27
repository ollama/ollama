package create

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

// CreateImageGenModel imports an image generation model from a directory.
// Stores each tensor as a separate blob for fine-grained deduplication.
// If quantize is specified, linear weights in transformer/text_encoder are quantized.
// Supported quantization types: fp8 (or empty for no quantization).
// Layer creation and manifest writing are done via callbacks to avoid import cycles.
func CreateImageGenModel(modelName, modelDir, quantize string, createLayer LayerCreator, createTensorLayer QuantizingTensorLayerCreator, writeManifest ManifestWriter, fn func(status string)) error {
	// Validate quantization type
	switch quantize {
	case "", "fp4", "fp8":
		// valid
	default:
		return fmt.Errorf("unsupported quantization type %q: supported types are fp4, fp8", quantize)
	}

	var layers []LayerInfo
	var configLayer LayerInfo
	var totalParams int64 // Count parameters from original tensor shapes
	var torchDtype string // Read from component config for quantization display

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
			quantizeMsg := ""
			if quantize != "" && component != "vae" {
				quantizeMsg = ", quantizing to " + quantize
			}
			fn(fmt.Sprintf("importing %s/%s (%d tensors%s)", component, entry.Name(), len(tensorNames), quantizeMsg))

			for _, tensorName := range tensorNames {
				td, err := extractor.GetTensor(tensorName)
				if err != nil {
					extractor.Close()
					return fmt.Errorf("failed to get tensor %s: %w", tensorName, err)
				}

				// Count parameters from original tensor shape
				if len(td.Shape) > 0 {
					numElements := int64(1)
					for _, dim := range td.Shape {
						numElements *= int64(dim)
					}
					totalParams += numElements
				}

				// Store as minimal safetensors format (88 bytes header overhead)
				// This enables native mmap loading via mlx_load_safetensors
				// Use path-style name: "component/tensor_name"
				fullName := component + "/" + tensorName

				// Determine quantization type for this tensor (empty string if not quantizing)
				quantizeType := ""
				if quantize != "" && ShouldQuantize(tensorName, component) && canQuantizeShape(td.Shape) {
					quantizeType = quantize
				}

				// createTensorLayer returns multiple layers if quantizing (weight + scales)
				newLayers, err := createTensorLayer(td.SafetensorsReader(), fullName, td.Dtype, td.Shape, quantizeType)
				if err != nil {
					extractor.Close()
					return fmt.Errorf("failed to create layer for %s: %w", fullName, err)
				}
				layers = append(layers, newLayers...)
			}

			extractor.Close()
		}
	}

	// Read torch_dtype from text_encoder config for quantization display
	if torchDtype == "" {
		textEncoderConfig := filepath.Join(modelDir, "text_encoder/config.json")
		if data, err := os.ReadFile(textEncoderConfig); err == nil {
			var cfg struct {
				TorchDtype string `json:"torch_dtype"`
			}
			if json.Unmarshal(data, &cfg) == nil && cfg.TorchDtype != "" {
				torchDtype = cfg.TorchDtype
			}
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

		// For model_index.json, normalize to Ollama format and add metadata
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

			// Add parameter count (counted from tensor shapes during import)
			cfg["parameter_count"] = totalParams

			// Add quantization info - use quantize type if set, otherwise torch_dtype
			if quantize != "" {
				cfg["quantization"] = strings.ToUpper(quantize)
			} else {
				cfg["quantization"] = torchDtype
			}

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

// canQuantizeShape returns true if a tensor shape is compatible with MLX quantization.
// MLX requires the last dimension to be divisible by the group size (32).
func canQuantizeShape(shape []int32) bool {
	if len(shape) < 2 {
		return false
	}
	return shape[len(shape)-1]%32 == 0
}
