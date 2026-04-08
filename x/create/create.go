package create

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/safetensors"
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
// When quantize is non-empty (e.g., "int8"), returns multiple layers (weight + scales + biases).
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

// ShouldQuantizeTensor returns true if a tensor should be quantized based on name, shape, and quantize type.
// This is a more detailed check that also considers tensor dimensions.
// The quantize parameter specifies the quantization type (e.g., "int4", "nvfp4", "mxfp4", "int8", "mxfp8").
func ShouldQuantizeTensor(name string, shape []int32, quantize string) bool {
	return GetTensorQuantization(name, shape, quantize) != ""
}

// normalizeQuantType converts various quantization type aliases to canonical forms.
// Supports: q4/Q4/int4/INT4/fp4/FP4 -> int4, q8/Q8/int8/INT8/fp8/FP8 -> int8, nvfp4/NVFP4, mxfp4/MXFP4, mxfp8/MXFP8
func normalizeQuantType(quantize string) string {
	switch strings.ToUpper(quantize) {
	case "Q4", "INT4", "FP4":
		return "int4"
	case "Q8", "INT8", "FP8":
		return "int8"
	case "NVFP4":
		return "nvfp4"
	case "MXFP4":
		return "mxfp4"
	case "MXFP8":
		return "mxfp8"
	default:
		return quantize
	}
}

func isStackedExpertWeight(name string) bool {
	// Combined/stacked expert tensors may be emitted either as "...proj.weight" (per-expert)
	// or "...proj" (pre-stacked packed tensor).
	if strings.HasSuffix(name, ".bias") || strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".qbias") {
		return false
	}

	return strings.Contains(name, ".mlp.switch_mlp.") ||
		strings.Contains(name, ".mlp.experts.") ||
		strings.Contains(name, ".mlp.shared_experts.")
}

// GetTensorQuantization returns the appropriate quantization type for a tensor.
// Returns "" if the tensor should not be quantized.
// This implements mixed-precision quantization:
//   - Attention MLA weights (q_a, q_b, kv_a, kv_b): unquantized (most sensitive)
//   - Output projection, gate/up weights: int4 (less sensitive)
//   - Down projection weights: int8 (more sensitive, would be Q6 in GGML but no MLX kernel)
//   - Norms, embeddings, biases, routing gates: no quantization
func GetTensorQuantization(name string, shape []int32, quantize string) string {
	stackedExpert := isStackedExpertWeight(name)

	// Use basic name-based check first
	if !stackedExpert && !ShouldQuantize(name, "") {
		return ""
	}

	// Quantize standard linear weights (2D). Also allow stacked expert weights (3D),
	// e.g. qwen switch_mlp / experts combined tensors.
	if len(shape) != 2 && !(len(shape) == 3 && stackedExpert) {
		return ""
	}

	// Skip small tensors (less than 1024 elements) - not worth quantizing
	var elems int64 = 1
	for _, d := range shape {
		elems *= int64(d)
	}
	if elems < 1024 {
		return ""
	}

	// Normalize quantization type to canonical form
	quantNorm := normalizeQuantType(quantize)

	// MLX quantization requires last dimension to be divisible by group size
	// nvfp4: 16, mxfp4/mxfp8: 32, int4/int8: 64
	groupSize := int32(32)
	switch quantNorm {
	case "nvfp4":
		groupSize = 16
	case "int4", "int8":
		groupSize = 64
	}
	if shape[len(shape)-1]%groupSize != 0 {
		return ""
	}

	// Skip routing gate weights (should stay high precision)
	// In safetensors these are: mlp.gate.weight (not mlp.gate_proj.weight)
	if strings.Contains(name, "mlp.gate.weight") && !strings.Contains(name, "_proj") {
		return ""
	}

	// For non-affine modes, use the same quantization for all eligible tensors.
	if quantNorm == "nvfp4" || quantNorm == "mxfp4" || quantNorm == "mxfp8" {
		return quantNorm
	}

	// Attention MLA weights - keep unquantized (bf16)
	// These are highly sensitive: errors accumulate in the KV cache over time
	// q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj
	if strings.Contains(name, "q_a_proj") ||
		strings.Contains(name, "q_b_proj") ||
		strings.Contains(name, "kv_a_proj") ||
		strings.Contains(name, "kv_b_proj") {
		return "" // No quantization - keep bf16
	}

	// Down projection weights - use INT8 (would be Q6_K in GGML, but MLX has no Q6 kernel)
	// mlp.down_proj, mlp.experts.X.down_proj, mlp.shared_experts.down_proj
	if strings.Contains(name, "down_proj") {
		return "int8"
	}

	// Output projection, gate/up weights - use requested quantization (INT4)
	// o_proj, gate_proj, up_proj
	if strings.Contains(name, "o_proj") ||
		strings.Contains(name, "gate_proj") ||
		strings.Contains(name, "up_proj") {
		return quantNorm
	}

	// LM head - use requested quantization
	if strings.Contains(name, "lm_head") {
		return quantNorm
	}

	// Default to requested quantization for other weights
	return quantNorm
}

var expertLayerPrefixRegexp = regexp.MustCompile(`^(?:model\.language_model\.|language_model(?:\.model)?\.|model\.)?layers\.\d+$`)

// ExpertGroupPrefix returns the group prefix for expert tensors that should be packed together.
// For example:
//   - "model.layers.1.mlp.experts.0.down_proj.weight" -> "model.layers.1.mlp.experts"
//   - "model.layers.1.mlp.shared_experts.down_proj.weight" -> "model.layers.1.mlp.shared_experts"
//   - "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight" -> "language_model.model.layers.1.mlp.switch_mlp"
//   - "model.layers.0.mlp.down_proj.weight" -> "" (dense layer, no experts)
//   - "model.layers.1.mlp.gate.weight" -> "" (routing gate, not an expert)
func ExpertGroupPrefix(tensorName string) string {
	if !strings.HasSuffix(tensorName, ".weight") {
		return ""
	}

	for _, marker := range []string{
		".mlp.experts.",
		".mlp.shared_experts.",
		".mlp.switch_mlp.",
	} {
		idx := strings.Index(tensorName, marker)
		if idx == -1 {
			continue
		}

		layerPrefix := tensorName[:idx]
		if !expertLayerPrefixRegexp.MatchString(layerPrefix) {
			continue
		}

		return layerPrefix + strings.TrimSuffix(marker, ".")
	}

	return ""
}

// PackedTensorInput holds metadata for a tensor that will be packed into a multi-tensor blob.
type PackedTensorInput struct {
	Name     string
	Dtype    string
	Shape    []int32
	Quantize string    // per-tensor quantization type (may differ within group)
	Reader   io.Reader // safetensors-wrapped tensor data
}

// PackedTensorLayerCreator creates a single blob layer containing multiple packed tensors.
// groupName is the group prefix (e.g., "model.layers.1.mlp.experts").
type PackedTensorLayerCreator func(groupName string, tensors []PackedTensorInput) (LayerInfo, error)

type sourceQuantization struct {
	Bits            int     `json:"bits"`
	GroupSize       int     `json:"group_size"`
	Mode            string  `json:"mode"`
	QuantMethod     string  `json:"quant_method"`
	WeightBlockSize []int32 `json:"weight_block_size"`
}

type sourceModelConfig struct {
	ModelType          string             `json:"model_type"`
	Architectures      []string           `json:"architectures"`
	Quantization       sourceQuantization `json:"quantization"`
	QuantizationConfig sourceQuantization `json:"quantization_config"`
	TextConfig         struct {
		ModelType          string             `json:"model_type"`
		Quantization       sourceQuantization `json:"quantization"`
		QuantizationConfig sourceQuantization `json:"quantization_config"`
	} `json:"text_config"`
}

func readSourceModelConfig(modelDir string) (sourceModelConfig, error) {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return sourceModelConfig{}, err
	}

	var cfg sourceModelConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return sourceModelConfig{}, err
	}

	return cfg, nil
}

func (cfg sourceModelConfig) Architecture() string {
	if len(cfg.Architectures) > 0 && cfg.Architectures[0] != "" {
		return cfg.Architectures[0]
	}
	if cfg.ModelType != "" {
		return cfg.ModelType
	}
	return cfg.TextConfig.ModelType
}

func (cfg sourceModelConfig) QuantMetadata() map[string]string {
	// Use the first non-empty quantization config found
	var q sourceQuantization
	for _, candidate := range []sourceQuantization{
		cfg.Quantization,
		cfg.QuantizationConfig,
		cfg.TextConfig.Quantization,
		cfg.TextConfig.QuantizationConfig,
	} {
		if candidate.Bits != 0 {
			q = candidate
			break
		}
	}

	quantType := sourceQuantType(q.Mode, q.Bits)
	if quantType == "" {
		return nil
	}

	metadata := map[string]string{"quant_type": quantType}
	if q.GroupSize > 0 {
		metadata["group_size"] = strconv.Itoa(q.GroupSize)
	}
	return metadata
}

type sourceQuantizedKind string

const (
	sourceQuantizedKindNone         sourceQuantizedKind = ""
	sourceQuantizedKindPrequantized sourceQuantizedKind = "prequantized"
	sourceQuantizedKindHFFP8        sourceQuantizedKind = "hf_fp8"
)

func (cfg sourceModelConfig) quantizationConfigs() []sourceQuantization {
	return []sourceQuantization{
		cfg.Quantization,
		cfg.QuantizationConfig,
		cfg.TextConfig.Quantization,
		cfg.TextConfig.QuantizationConfig,
	}
}

func (cfg sourceModelConfig) HFFP8WeightBlockSize() (rows, cols int32, ok bool) {
	for _, q := range cfg.quantizationConfigs() {
		if !strings.EqualFold(q.QuantMethod, "fp8") || len(q.WeightBlockSize) != 2 {
			continue
		}
		return q.WeightBlockSize[0], q.WeightBlockSize[1], true
	}
	return 0, 0, false
}

func inspectSourceQuantization(modelDir string, cfg sourceModelConfig) (sourceQuantizedKind, error) {
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return sourceQuantizedKindNone, err
	}

	hasScaleInv := false
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") {
			continue
		}

		extractor, err := safetensors.OpenForExtraction(filepath.Join(modelDir, entry.Name()))
		if err != nil {
			return sourceQuantizedKindNone, err
		}

		for _, name := range extractor.ListTensors() {
			switch {
			case strings.HasSuffix(name, ".scales"):
				extractor.Close()
				return sourceQuantizedKindPrequantized, nil
			case strings.HasSuffix(name, ".weight_scale_inv"):
				hasScaleInv = true
			}
		}

		extractor.Close()
	}

	if hasScaleInv {
		if _, _, ok := cfg.HFFP8WeightBlockSize(); ok {
			return sourceQuantizedKindHFFP8, nil
		}
	}

	return sourceQuantizedKindNone, nil
}

func resolveEffectiveQuantization(cfg sourceModelConfig, sourceKind sourceQuantizedKind, requested string) (string, error) {
	switch sourceKind {
	case sourceQuantizedKindNone:
		return requested, nil
	case sourceQuantizedKindPrequantized:
		if requested != "" {
			return "", fmt.Errorf("cannot requantize already-quantized source model with --quantize %q", requested)
		}
		return "", nil
	case sourceQuantizedKindHFFP8:
		if requested != "" {
			return "", fmt.Errorf("cannot requantize already-quantized fp8 source model with --quantize %q", requested)
		}
		rows, cols, ok := cfg.HFFP8WeightBlockSize()
		if !ok {
			return "", fmt.Errorf("fp8 source model missing weight_block_size metadata")
		}
		if rows != 128 || cols != 128 {
			return "", fmt.Errorf("unsupported fp8 source block size %dx%d", rows, cols)
		}
		return "mxfp8", nil
	default:
		return "", fmt.Errorf("unsupported source quantization kind %q", sourceKind)
	}
}

type tensorImportTransform interface {
	skipTensor(name string) bool
	transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error)
	quantizationType(name string, shape []int32, quantize string) string
}

type noopImportTransform struct{}

func (noopImportTransform) skipTensor(string) bool { return false }

func (noopImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	return []*safetensors.TensorData{td}, nil
}

func (noopImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	return GetTensorQuantization(name, shape, quantize)
}

type tensorImportTransformFactory func(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error)

var tensorImportTransformRegistry = map[string]tensorImportTransformFactory{
	"Qwen3_5ForCausalLM":                   newQwen35ImportTransform,
	"Qwen3_5ForConditionalGeneration":      newQwen35ImportTransform,
	"Qwen3NextForCausalLM":                 newQwen35ImportTransform,
	"Qwen3NextForConditionalGeneration":    newQwen35ImportTransform,
	"Qwen3_5MoeForCausalLM":                newQwen35ImportTransform,
	"Qwen3_5MoeForConditionalGeneration":   newQwen35ImportTransform,
	"Qwen3NextMoeForCausalLM":              newQwen35ImportTransform,
	"Qwen3NextMoeForConditionalGeneration": newQwen35ImportTransform,
}

func newTensorImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	if factory, ok := tensorImportTransformRegistry[cfg.Architecture()]; ok {
		return factory(modelDir, cfg)
	}
	return noopImportTransform{}, nil
}

// CreateSafetensorsModel imports a standard safetensors model from a directory.
// This handles Hugging Face style models with config.json and *.safetensors files.
// Stores each tensor as a separate blob for fine-grained deduplication.
// Expert tensors are packed into per-layer blobs when createPackedLayer is non-nil.
// If quantize is non-empty (e.g., "int8"), eligible tensors will be quantized.
func CreateSafetensorsModel(modelName, modelDir, quantize string, createLayer LayerCreator, createTensorLayer QuantizingTensorLayerCreator, writeManifest ManifestWriter, fn func(status string), createPackedLayer ...PackedTensorLayerCreator) error {
	var layers []LayerInfo
	var configLayer LayerInfo
	sourceConfig, err := readSourceModelConfig(modelDir)
	if err != nil {
		return fmt.Errorf("failed to read source config.json: %w", err)
	}
	sourceQuantKind, err := inspectSourceQuantization(modelDir, sourceConfig)
	if err != nil {
		return fmt.Errorf("failed to inspect source quantization: %w", err)
	}
	effectiveQuantize, err := resolveEffectiveQuantization(sourceConfig, sourceQuantKind, quantize)
	if err != nil {
		return err
	}
	sourceQuantMetadata := sourceConfig.QuantMetadata()
	importTransform, err := newTensorImportTransform(modelDir, sourceConfig)
	if err != nil {
		return fmt.Errorf("failed to construct import transform for architecture %q: %w", sourceConfig.Architecture(), err)
	}

	// Resolve the optional packed layer creator
	var packedCreator PackedTensorLayerCreator
	if len(createPackedLayer) > 0 {
		packedCreator = createPackedLayer[0]
	}
	// Accumulate expert tensors by group prefix for packing.
	// Readers reference file-backed SectionReaders, so we keep extractors
	// open until each group is flushed to avoid buffering tensor data in memory.
	expertGroups := make(map[string][]PackedTensorInput)
	var expertGroupOrder []string

	// Track open extractors so we can close them after flushing groups
	var openExtractors []*safetensors.TensorExtractor

	closeExtractors := func() {
		for _, ext := range openExtractors {
			ext.Close()
		}
		openExtractors = nil
	}

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
			closeExtractors()
			return fmt.Errorf("failed to open %s: %w", stPath, err)
		}

		tensorNames := extractor.ListTensors()
		tensorSet := make(map[string]struct{}, len(tensorNames))
		for _, name := range tensorNames {
			tensorSet[name] = struct{}{}
		}
		quantizeMsg := ""
		if effectiveQuantize != "" {
			quantizeMsg = fmt.Sprintf(", quantizing to %s", effectiveQuantize)
		}
		fn(fmt.Sprintf("importing %s (%d tensors%s)", entry.Name(), len(tensorNames), quantizeMsg))

		// Track whether this extractor has expert tensors that need to stay open
		hasExpertTensors := false

		for _, tensorName := range tensorNames {
			if importTransform.skipTensor(tensorName) {
				continue
			}
			if shouldSkipSourceCompanion(tensorName, tensorSet) {
				continue
			}
			sourceFP8ScaleName, hasSourceFP8Scale := sourceFP8Companion(tensorName, tensorSet)

			td, err := extractor.GetTensor(tensorName)
			if err != nil {
				extractor.Close()
				closeExtractors()
				return fmt.Errorf("failed to get tensor %s: %w", tensorName, err)
			}

			if effectiveQuantize == "" {
				layer, ok, err := createPrequantizedLayer(extractor, td, tensorName, tensorSet, sourceQuantMetadata, createLayer)
				if err != nil {
					extractor.Close()
					closeExtractors()
					return err
				}
				if ok {
					layers = append(layers, layer)
					continue
				}
			}

			outputTensors, err := importTransform.transformTensor(td)
			if err != nil {
				extractor.Close()
				closeExtractors()
				return fmt.Errorf("failed to transform tensor %s: %w", tensorName, err)
			}

			for _, outTD := range outputTensors {
				// Determine quantization type for this tensor (empty string if not quantizing)
				// GetTensorQuantization handles mixed-precision (e.g., Q8 for attention, Q4 for FFN)
				quantizeType := ""
				switch {
				case sourceQuantKind == sourceQuantizedKindHFFP8 && hasSourceFP8Scale:
					quantizeType = "mxfp8"
				case sourceQuantKind == sourceQuantizedKindHFFP8:
					quantizeType = ""
				case effectiveQuantize != "":
					quantizeType = importTransform.quantizationType(outTD.Name, outTD.Shape, effectiveQuantize)
				}
				reader := outTD.SafetensorsReader()
				if hasSourceFP8Scale {
					if len(outputTensors) != 1 {
						extractor.Close()
						closeExtractors()
						return fmt.Errorf("source fp8 tensor %s rewrote into %d tensors; only 1:1 rewrites are supported", tensorName, len(outputTensors))
					}
					if quantizeType == "" {
						extractor.Close()
						closeExtractors()
						return fmt.Errorf("source fp8 tensor %s was not scheduled for mxfp8 conversion", tensorName)
					}
					scaleTD, err := extractor.GetTensor(sourceFP8ScaleName)
					if err != nil {
						extractor.Close()
						closeExtractors()
						return fmt.Errorf("failed to get fp8 scale tensor %s: %w", sourceFP8ScaleName, err)
					}
					reader = buildSourceFP8Reader(outTD, scaleTD.WithName(outTD.Name+".scale_inv"))
				}

				// Check if this tensor belongs to an expert group for packing
				groupPrefix := ""
				if packedCreator != nil {
					groupPrefix = ExpertGroupPrefix(outTD.Name)
				}

				if groupPrefix != "" {
					// Accumulate expert tensor for packed blob.
					// The Reader uses a file-backed SectionReader, so we must
					// keep the extractor open until this group is flushed.
					hasExpertTensors = true
					if _, exists := expertGroups[groupPrefix]; !exists {
						expertGroupOrder = append(expertGroupOrder, groupPrefix)
					}
					expertGroups[groupPrefix] = append(expertGroups[groupPrefix], PackedTensorInput{
						Name:     outTD.Name,
						Dtype:    outTD.Dtype,
						Shape:    outTD.Shape,
						Quantize: quantizeType,
						Reader:   reader,
					})
				} else {
					// Store as minimal safetensors format (88 bytes header overhead)
					// This enables native mmap loading via mlx_load_safetensors
					// createTensorLayer returns multiple layers if quantizing (weight + scales)
					newLayers, err := createTensorLayer(reader, outTD.Name, outTD.Dtype, outTD.Shape, quantizeType)
					if err != nil {
						extractor.Close()
						closeExtractors()
						return fmt.Errorf("failed to create layer for %s: %w", outTD.Name, err)
					}
					layers = append(layers, newLayers...)
				}
			}
		}

		if hasExpertTensors {
			// Keep extractor open - readers still reference its file handle
			openExtractors = append(openExtractors, extractor)
		} else {
			extractor.Close()
		}
	}

	// Process accumulated expert groups into packed blobs, then close extractors
	if packedCreator != nil {
		sort.Strings(expertGroupOrder)
		for _, groupName := range expertGroupOrder {
			tensors := expertGroups[groupName]
			fn(fmt.Sprintf("packing %s (%d tensors)", groupName, len(tensors)))
			layer, err := packedCreator(groupName, tensors)
			if err != nil {
				closeExtractors()
				return fmt.Errorf("failed to create packed layer for %s: %w", groupName, err)
			}
			layers = append(layers, layer)
		}
	}
	closeExtractors()

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

func shouldSkipSourceCompanion(name string, tensorSet map[string]struct{}) bool {
	switch {
	case strings.HasSuffix(name, ".scales"):
		_, ok := tensorSet[strings.TrimSuffix(name, ".scales")+".weight"]
		return ok
	case strings.HasSuffix(name, ".biases"):
		_, ok := tensorSet[strings.TrimSuffix(name, ".biases")+".weight"]
		return ok
	case strings.HasSuffix(name, ".weight_scale_inv"):
		_, ok := tensorSet[strings.TrimSuffix(name, "_scale_inv")]
		return ok
	default:
		return false
	}
}

func sourceFP8Companion(weightName string, tensorSet map[string]struct{}) (scaleName string, ok bool) {
	if !strings.HasSuffix(weightName, ".weight") {
		return "", false
	}

	scaleName = weightName + "_scale_inv"
	_, ok = tensorSet[scaleName]
	return scaleName, ok
}

func buildSourceFP8Reader(weightTD, scaleTD *safetensors.TensorData) io.Reader {
	return safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{weightTD, scaleTD})
}

func createPrequantizedLayer(
	extractor *safetensors.TensorExtractor,
	td *safetensors.TensorData,
	tensorName string,
	tensorSet map[string]struct{},
	metadata map[string]string,
	createLayer LayerCreator,
) (LayerInfo, bool, error) {
	scaleName, biasName, ok := prequantizedCompanions(tensorName, tensorSet)
	if !ok {
		return LayerInfo{}, false, nil
	}

	tensors := []*safetensors.TensorData{td.WithName(tensorName)}

	scaleTD, err := extractor.GetTensor(scaleName)
	if err != nil {
		return LayerInfo{}, false, fmt.Errorf("failed to get tensor %s: %w", scaleName, err)
	}
	tensors = append(tensors, scaleTD.WithName(tensorName+".scale"))

	if biasName != "" {
		biasTD, err := extractor.GetTensor(biasName)
		if err != nil {
			return LayerInfo{}, false, fmt.Errorf("failed to get tensor %s: %w", biasName, err)
		}
		tensors = append(tensors, biasTD.WithName(tensorName+".bias"))
	}

	layer, err := createLayer(
		safetensors.BuildPackedSafetensorsReaderWithMetadata(tensors, metadata),
		"application/vnd.ollama.image.tensor",
		tensorName,
	)
	if err != nil {
		return LayerInfo{}, false, fmt.Errorf("failed to create prequantized layer for %s: %w", tensorName, err)
	}
	return layer, true, nil
}

func prequantizedCompanions(weightName string, tensorSet map[string]struct{}) (scaleName, biasName string, ok bool) {
	if !strings.HasSuffix(weightName, ".weight") {
		return "", "", false
	}

	base := strings.TrimSuffix(weightName, ".weight")
	scaleName = base + ".scales"
	if _, ok := tensorSet[scaleName]; !ok {
		return "", "", false
	}

	biasName = base + ".biases"
	if _, ok := tensorSet[biasName]; !ok {
		biasName = ""
	}
	return scaleName, biasName, true
}
