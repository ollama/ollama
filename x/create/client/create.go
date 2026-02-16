// Package client provides client-side model creation for safetensors-based models.
//
// This package is in x/ because the safetensors model storage format is under development.
// It also exists to break an import cycle: server imports x/create, so x/create
// cannot import server. This sub-package can import server because server doesn't
// import it.
package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/imagegen/safetensors"
)

// MinOllamaVersion is the minimum Ollama version required for safetensors models.
const MinOllamaVersion = "0.14.0"

// ModelfileConfig holds configuration extracted from a Modelfile.
type ModelfileConfig struct {
	Template string
	System   string
	License  string
}

// CreateOptions holds all options for model creation.
type CreateOptions struct {
	ModelName string
	ModelDir  string
	Quantize  string           // "int4", "int8", "nvfp4", or "mxfp8" for quantization
	Modelfile *ModelfileConfig // template/system/license from Modelfile
}

// CreateModel imports a model from a local directory.
// This creates blobs and manifest directly on disk, bypassing the HTTP API.
// Automatically detects model type (safetensors LLM vs image gen) and routes accordingly.
func CreateModel(opts CreateOptions, p *progress.Progress) error {
	// Detect model type
	isSafetensors := create.IsSafetensorsModelDir(opts.ModelDir)
	isImageGen := create.IsTensorModelDir(opts.ModelDir)

	if !isSafetensors && !isImageGen {
		return fmt.Errorf("%s is not a supported model directory (needs config.json + *.safetensors or model_index.json)", opts.ModelDir)
	}

	// Determine model type settings
	var modelType, spinnerKey string
	var capabilities []string
	var parserName, rendererName string
	if isSafetensors {
		modelType = "safetensors model"
		spinnerKey = "create"
		capabilities = []string{"completion"}

		// Check if model supports thinking based on architecture
		if supportsThinking(opts.ModelDir) {
			capabilities = append(capabilities, "thinking")
		}

		// Set parser and renderer name based on architecture
		parserName = getParserName(opts.ModelDir)
		rendererName = getRendererName(opts.ModelDir)
	} else {
		modelType = "image generation model"
		spinnerKey = "imagegen"
		capabilities = []string{"image"}
	}

	// Set up progress spinner
	statusMsg := "importing " + modelType
	spinner := progress.NewSpinner(statusMsg)
	p.Add(spinnerKey, spinner)

	progressFn := func(msg string) {
		spinner.Stop()
		statusMsg = msg
		spinner = progress.NewSpinner(statusMsg)
		p.Add(spinnerKey, spinner)
	}

	// Create the model using shared callbacks
	var err error
	if isSafetensors {
		err = create.CreateSafetensorsModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			newLayerCreator(), newTensorLayerCreator(),
			newManifestWriter(opts, capabilities, parserName, rendererName),
			progressFn,
			newPackedTensorLayerCreator(),
		)
	} else {
		err = create.CreateImageGenModel(
			opts.ModelName, opts.ModelDir, opts.Quantize,
			newLayerCreator(), newTensorLayerCreator(),
			newManifestWriter(opts, capabilities, "", ""),
			progressFn,
		)
	}

	spinner.Stop()
	if err != nil {
		return err
	}

	fmt.Printf("Created %s '%s'\n", modelType, opts.ModelName)
	return nil
}

// newLayerCreator returns a LayerCreator callback for creating config/JSON layers.
func newLayerCreator() create.LayerCreator {
	return func(r io.Reader, mediaType, name string) (create.LayerInfo, error) {
		layer, err := manifest.NewLayer(r, mediaType)
		if err != nil {
			return create.LayerInfo{}, err
		}

		return create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		}, nil
	}
}

// newTensorLayerCreator returns a QuantizingTensorLayerCreator callback for creating tensor layers.
// When quantize is non-empty, returns multiple layers (weight + scales + optional qbias).
func newTensorLayerCreator() create.QuantizingTensorLayerCreator {
	return func(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
		if quantize != "" {
			return createQuantizedLayers(r, name, dtype, shape, quantize)
		}
		return createUnquantizedLayer(r, name)
	}
}

// createQuantizedLayers quantizes a tensor and returns a single combined layer.
// The combined blob contains data, scale, and optional bias tensors with metadata.
func createQuantizedLayers(r io.Reader, name, dtype string, shape []int32, quantize string) ([]create.LayerInfo, error) {
	if !QuantizeSupported() {
		return nil, fmt.Errorf("quantization requires MLX support")
	}

	// Quantize the tensor into a single combined blob
	blobData, err := quantizeTensor(r, name, dtype, shape, quantize)
	if err != nil {
		return nil, fmt.Errorf("failed to quantize %s: %w", name, err)
	}

	// Create single layer for the combined blob
	layer, err := manifest.NewLayer(bytes.NewReader(blobData), manifest.MediaTypeImageTensor)
	if err != nil {
		return nil, err
	}

	return []create.LayerInfo{
		{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		},
	}, nil
}

// createUnquantizedLayer creates a single tensor layer without quantization.
func createUnquantizedLayer(r io.Reader, name string) ([]create.LayerInfo, error) {
	layer, err := manifest.NewLayer(r, manifest.MediaTypeImageTensor)
	if err != nil {
		return nil, err
	}

	return []create.LayerInfo{
		{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      name,
		},
	}, nil
}

// newPackedTensorLayerCreator returns a PackedTensorLayerCreator callback for
// creating packed multi-tensor blob layers (used for expert groups).
func newPackedTensorLayerCreator() create.PackedTensorLayerCreator {
	return func(groupName string, tensors []create.PackedTensorInput) (create.LayerInfo, error) {
		// Check if any tensor in the group needs quantization
		hasQuantize := false
		for _, t := range tensors {
			if t.Quantize != "" {
				hasQuantize = true
				break
			}
		}

		var blobReader io.Reader
		if hasQuantize {
			if !QuantizeSupported() {
				return create.LayerInfo{}, fmt.Errorf("quantization requires MLX support")
			}
			blobData, err := quantizePackedGroup(tensors)
			if err != nil {
				return create.LayerInfo{}, fmt.Errorf("failed to quantize packed group %s: %w", groupName, err)
			}
			blobReader = bytes.NewReader(blobData)
		} else {
			// Build unquantized packed blob using streaming reader
			// Extract raw tensor data from safetensors-wrapped readers
			var tds []*safetensors.TensorData
			for _, t := range tensors {
				rawData, err := safetensors.ExtractRawFromSafetensors(t.Reader)
				if err != nil {
					return create.LayerInfo{}, fmt.Errorf("failed to extract tensor %s: %w", t.Name, err)
				}
				td := safetensors.NewTensorDataFromBytes(t.Name, t.Dtype, t.Shape, rawData)
				tds = append(tds, td)
			}
			blobReader = safetensors.BuildPackedSafetensorsReader(tds)
		}

		layer, err := manifest.NewLayer(blobReader, manifest.MediaTypeImageTensor)
		if err != nil {
			return create.LayerInfo{}, err
		}

		return create.LayerInfo{
			Digest:    layer.Digest,
			Size:      layer.Size,
			MediaType: layer.MediaType,
			Name:      groupName,
		}, nil
	}
}

// newManifestWriter returns a ManifestWriter callback for writing the model manifest.
func newManifestWriter(opts CreateOptions, capabilities []string, parserName, rendererName string) create.ManifestWriter {
	return func(modelName string, config create.LayerInfo, layers []create.LayerInfo) error {
		name := model.ParseName(modelName)
		if !name.IsValid() {
			return fmt.Errorf("invalid model name: %s", modelName)
		}

		// TODO: find a better way to detect image input support
		// For now, hardcode Flux2KleinPipeline as supporting vision (image input)
		caps := capabilities
		modelIndex := filepath.Join(opts.ModelDir, "model_index.json")
		if data, err := os.ReadFile(modelIndex); err == nil {
			var cfg struct {
				ClassName string `json:"_class_name"`
			}
			if json.Unmarshal(data, &cfg) == nil && cfg.ClassName == "Flux2KleinPipeline" {
				caps = append(caps, "vision")
			}
		}

		// Create config blob with version requirement
		configData := model.ConfigV2{
			ModelFormat:  "safetensors",
			Capabilities: caps,
			Requires:     MinOllamaVersion,
			Parser:       parserName,
			Renderer:     rendererName,
		}
		configJSON, err := json.Marshal(configData)
		if err != nil {
			return fmt.Errorf("failed to marshal config: %w", err)
		}

		// Create config layer blob
		configLayer, err := manifest.NewLayer(bytes.NewReader(configJSON), "application/vnd.docker.container.image.v1+json")
		if err != nil {
			return fmt.Errorf("failed to create config layer: %w", err)
		}

		// Convert LayerInfo to manifest.Layer
		manifestLayers := make([]manifest.Layer, 0, len(layers))
		for _, l := range layers {
			manifestLayers = append(manifestLayers, manifest.Layer{
				MediaType: l.MediaType,
				Digest:    l.Digest,
				Size:      l.Size,
				Name:      l.Name,
			})
		}

		// Add Modelfile layers if present
		if opts.Modelfile != nil {
			modelfileLayers, err := createModelfileLayers(opts.Modelfile)
			if err != nil {
				return err
			}
			manifestLayers = append(manifestLayers, modelfileLayers...)
		}

		return manifest.WriteManifest(name, configLayer, manifestLayers)
	}
}

// createModelfileLayers creates layers for template, system, and license from Modelfile config.
func createModelfileLayers(mf *ModelfileConfig) ([]manifest.Layer, error) {
	var layers []manifest.Layer

	if mf.Template != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.Template)), "application/vnd.ollama.image.template")
		if err != nil {
			return nil, fmt.Errorf("failed to create template layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.System != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.System)), "application/vnd.ollama.image.system")
		if err != nil {
			return nil, fmt.Errorf("failed to create system layer: %w", err)
		}
		layers = append(layers, layer)
	}

	if mf.License != "" {
		layer, err := manifest.NewLayer(bytes.NewReader([]byte(mf.License)), "application/vnd.ollama.image.license")
		if err != nil {
			return nil, fmt.Errorf("failed to create license layer: %w", err)
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

// supportsThinking checks if the model supports thinking mode based on its architecture.
// This reads the config.json from the model directory and checks the architectures field.
func supportsThinking(modelDir string) bool {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return false
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return false
	}

	// Check architectures that support thinking
	thinkingArchitectures := []string{
		"glm4moe",  // GLM-4 MoE models
		"deepseek", // DeepSeek models
		"qwen3",    // Qwen3 models
	}

	// Check the architecture list
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		for _, thinkArch := range thinkingArchitectures {
			if strings.Contains(archLower, thinkArch) {
				return true
			}
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		for _, thinkArch := range thinkingArchitectures {
			if strings.Contains(typeLower, thinkArch) {
				return true
			}
		}
	}

	return false
}

// getParserName returns the parser name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate parser.
func getParserName(modelDir string) string {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}

	// Check architectures for known parsers
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "glm4") || strings.Contains(archLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(archLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(archLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		if strings.Contains(typeLower, "glm4") || strings.Contains(typeLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(typeLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(typeLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	return ""
}

// getRendererName returns the renderer name for a model based on its architecture.
// This reads the config.json from the model directory and determines the appropriate renderer.
func getRendererName(modelDir string) string {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}

	var cfg struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}

	// Check architectures for known renderers
	for _, arch := range cfg.Architectures {
		archLower := strings.ToLower(arch)
		if strings.Contains(archLower, "glm4") || strings.Contains(archLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(archLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(archLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	// Also check model_type
	if cfg.ModelType != "" {
		typeLower := strings.ToLower(cfg.ModelType)
		if strings.Contains(typeLower, "glm4") || strings.Contains(typeLower, "glm-4") {
			return "glm-4.7"
		}
		if strings.Contains(typeLower, "deepseek") {
			return "deepseek3"
		}
		if strings.Contains(typeLower, "qwen3") {
			return "qwen3-coder"
		}
	}

	return ""
}
