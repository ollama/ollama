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
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	modelparsers "github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/safetensors"
)

// MinOllamaVersion is the minimum Ollama version required for safetensors models.
const MinOllamaVersion = "0.19.0"

// ModelfileConfig holds configuration extracted from a Modelfile.
type ModelfileConfig struct {
	Template   string
	System     string
	License    string
	Parser     string
	Renderer   string
	Parameters map[string]any
}

var ignoredModelfileParameters = []string{
	"penalize_newline",
	"low_vram",
	"f16_kv",
	"logits_all",
	"vocab_only",
	"use_mlock",
	"mirostat",
	"mirostat_tau",
	"mirostat_eta",
}

// ConfigFromModelfile extracts the model directory and x/create-specific
// Modelfile configuration from a parsed Modelfile.
func ConfigFromModelfile(modelfile *parser.Modelfile) (string, *ModelfileConfig, error) {
	var modelDir string
	mfConfig := &ModelfileConfig{}

	for _, cmd := range modelfile.Commands {
		switch cmd.Name {
		case "model":
			modelDir = cmd.Args
		case "template":
			mfConfig.Template = cmd.Args
		case "system":
			mfConfig.System = cmd.Args
		case "license":
			mfConfig.License = cmd.Args
		case "parser":
			mfConfig.Parser = cmd.Args
		case "renderer":
			mfConfig.Renderer = cmd.Args
		case "adapter", "message", "requires":
			continue
		default:
			if slices.Contains(ignoredModelfileParameters, cmd.Name) {
				continue
			}

			ps, err := api.FormatParams(map[string][]string{cmd.Name: {cmd.Args}})
			if err != nil {
				return "", nil, err
			}

			if mfConfig.Parameters == nil {
				mfConfig.Parameters = make(map[string]any)
			}

			for k, v := range ps {
				if ks, ok := mfConfig.Parameters[k].([]string); ok {
					mfConfig.Parameters[k] = append(ks, v.([]string)...)
				} else if vs, ok := v.([]string); ok {
					mfConfig.Parameters[k] = vs
				} else {
					mfConfig.Parameters[k] = v
				}
			}
		}
	}

	if modelDir == "" {
		modelDir = "."
	}

	return modelDir, mfConfig, nil
}

// CreateOptions holds all options for model creation.
type CreateOptions struct {
	ModelName string
	ModelDir  string
	Quantize  string           // "int4", "int8", "nvfp4", "mxfp4", or "mxfp8" for quantization
	Modelfile *ModelfileConfig // template/system/license/parser/renderer/parameters from Modelfile
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
		// Load config.json once and share it across capability / parser /
		// renderer detection.
		cfg := loadSafetensorsConfig(opts.ModelDir)
		parserName = getParserName(cfg)
		rendererName = getRendererName(cfg)
		capabilities = inferSafetensorsCapabilities(cfg, resolveParserName(opts.Modelfile, parserName))
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

// safetensorsConfig is the subset of HuggingFace config.json fields used for
// capability / parser / renderer detection at create time. Load once per
// create to avoid re-reading and re-parsing config.json for each predicate.
type safetensorsConfig struct {
	Architectures []string        `json:"architectures"`
	ModelType     string          `json:"model_type"`
	VisionConfig  *map[string]any `json:"vision_config"`
	AudioConfig   *map[string]any `json:"audio_config"`
}

// loadSafetensorsConfig reads modelDir/config.json. Returns nil if the file
// is missing or unparseable — callers should treat that as "unknown" and
// fall through to defaults.
func loadSafetensorsConfig(modelDir string) *safetensorsConfig {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nil
	}
	var cfg safetensorsConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil
	}
	return &cfg
}

// archAndTypeTokens returns the lowercased architectures (in declaration
// order) followed by the lowercased model_type. Used by parser/renderer
// dispatch so the matching logic walks architectures and model_type in one
// loop instead of duplicating the match block.
func (cfg *safetensorsConfig) archAndTypeTokens() []string {
	if cfg == nil {
		return nil
	}
	tokens := make([]string, 0, len(cfg.Architectures)+1)
	for _, arch := range cfg.Architectures {
		tokens = append(tokens, strings.ToLower(arch))
	}
	if cfg.ModelType != "" {
		tokens = append(tokens, strings.ToLower(cfg.ModelType))
	}
	return tokens
}

func inferSafetensorsCapabilities(cfg *safetensorsConfig, parserName string) []string {
	capabilities := []string{"completion"}

	// Qwen3.5 multimodal checkpoints use ConditionalGeneration architectures.
	if cfg != nil && cfg.VisionConfig != nil {
		capabilities = append(capabilities, "vision")
	}

	if cfg != nil && cfg.AudioConfig != nil {
		capabilities = append(capabilities, "audio")
	}

	var builtinParser modelparsers.Parser
	if parserName != "" {
		builtinParser = modelparsers.ParserForName(parserName)
	}

	if builtinParser != nil && builtinParser.HasToolSupport() {
		capabilities = append(capabilities, "tools")
	}

	if supportsThinking(cfg) || (builtinParser != nil && builtinParser.HasThinkingSupport()) {
		capabilities = append(capabilities, "thinking")
	}

	return capabilities
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
			blobData, err := quantizePackedGroup(groupName, tensors)
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
			FileType:     strings.ToLower(strings.TrimSpace(opts.Quantize)),
			Capabilities: caps,
			Requires:     MinOllamaVersion,
			Parser:       resolveParserName(opts.Modelfile, parserName),
			Renderer:     resolveRendererName(opts.Modelfile, rendererName),
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

func resolveParserName(mf *ModelfileConfig, inferred string) string {
	if mf != nil && mf.Parser != "" {
		return mf.Parser
	}

	return inferred
}

func resolveRendererName(mf *ModelfileConfig, inferred string) string {
	if mf != nil && mf.Renderer != "" {
		return mf.Renderer
	}

	return inferred
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

	if len(mf.Parameters) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(mf.Parameters); err != nil {
			return nil, fmt.Errorf("failed to encode parameters: %w", err)
		}

		layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return nil, fmt.Errorf("failed to create params layer: %w", err)
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

// supportsThinking reports whether the model's architecture or model_type
// matches one of the known thinking-capable families.
func supportsThinking(cfg *safetensorsConfig) bool {
	if cfg == nil {
		return false
	}
	thinkingArchitectures := []string{
		"glm4moe",  // GLM-4 MoE models
		"deepseek", // DeepSeek models
		"qwen3",    // Qwen3 models
	}
	for _, tok := range cfg.archAndTypeTokens() {
		for _, thinkArch := range thinkingArchitectures {
			if strings.Contains(tok, thinkArch) {
				return true
			}
		}
	}
	return false
}

// getParserName returns the parser name for a model based on its architecture.
func getParserName(cfg *safetensorsConfig) string {
	for _, tok := range cfg.archAndTypeTokens() {
		switch {
		case strings.Contains(tok, "laguna"):
			return "laguna"
		case strings.Contains(tok, "glm4"), strings.Contains(tok, "glm-4"):
			return "glm-4.7"
		case strings.Contains(tok, "deepseek"):
			return "deepseek3"
		case strings.Contains(tok, "gemma4"):
			return "gemma4"
		case strings.Contains(tok, "qwen3"):
			return "qwen3"
		}
	}
	return ""
}

// getRendererName returns the renderer name for a model based on its architecture.
func getRendererName(cfg *safetensorsConfig) string {
	for _, tok := range cfg.archAndTypeTokens() {
		switch {
		case strings.Contains(tok, "laguna"):
			return "laguna"
		case strings.Contains(tok, "gemma4"):
			return "gemma4"
		case strings.Contains(tok, "glm4"), strings.Contains(tok, "glm-4"):
			return "glm-4.7"
		case strings.Contains(tok, "deepseek"):
			return "deepseek3"
		case strings.Contains(tok, "qwen3"):
			return "qwen3-coder"
		}
	}
	return ""
}
