package create

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
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

	// Skip audio encoder tensors (highly sensitive to quantization)
	if strings.Contains(name, "audio_tower") || strings.Contains(name, "embed_audio") {
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

// isAligned checks if a tensor's last dimension is divisible by the
// group size required for the given quantization type.
func isAligned(shape []int32, quantType string) bool {
	if len(shape) == 0 {
		return false
	}
	groupSize := int32(32)
	switch normalizeQuantType(quantType) {
	case "nvfp4":
		groupSize = 16
	case "int4", "int8":
		groupSize = 64
	}
	return shape[len(shape)-1]%groupSize == 0
}

func isStackedExpertWeight(name string) bool {
	// Combined/stacked expert tensors may be emitted either as "...proj.weight" (per-expert)
	// or "...proj" (pre-stacked packed tensor).
	if strings.HasSuffix(name, ".bias") || strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".qbias") {
		return false
	}

	return strings.Contains(name, ".mlp.switch_mlp.") ||
		strings.Contains(name, ".mlp.experts.") ||
		strings.Contains(name, ".mlp.shared_experts.") ||
		strings.Contains(name, ".moe.experts.")
}

func sourceFP8BF16PromotionQuantization(name string, shape []int32, requested string) string {
	quantNorm := normalizeQuantType(requested)
	if quantNorm == "" {
		return ""
	}

	switch quantNorm {
	case "nvfp4", "mxfp4", "mxfp8":
	default:
		return ""
	}

	if !sourceFP8CanPromoteBF16Weight(name, shape) {
		return ""
	}

	return "mxfp8"
}

func sourceFP8TensorQuantization(name string, shape []int32, requested string, fallback string) string {
	quantNorm := normalizeQuantType(requested)
	switch quantNorm {
	case "nvfp4", "mxfp4":
		if sourceFP8ShouldPromoteLowBitTensor(name, shape) {
			return "mxfp8"
		}
	}
	return fallback
}

func sourceFP8ShouldPromoteLowBitTensor(name string, shape []int32) bool {
	if len(shape) != 2 || !isAligned(shape, "mxfp8") {
		return false
	}

	return strings.Contains(name, "down_proj") ||
		strings.Contains(name, ".v_proj") ||
		strings.Contains(name, ".k_proj")
}

func sourceFP8CanPromoteBF16Weight(name string, shape []int32) bool {
	if !strings.HasSuffix(name, ".weight") || len(shape) != 2 {
		return false
	}

	var elems int64 = 1
	for _, d := range shape {
		elems *= int64(d)
	}
	if elems < 1024 {
		return false
	}

	if !isAligned(shape, "mxfp8") {
		return false
	}

	switch {
	case strings.Contains(name, "audio_tower") || strings.Contains(name, "embed_audio"):
		return false
	case strings.Contains(name, "norm") || strings.Contains(name, "ln_") || strings.Contains(name, "layernorm"):
		return false
	case strings.Contains(name, "router") || strings.Contains(name, "score_correction"):
		return false
	case strings.Contains(name, "mlp.gate.weight") && !strings.Contains(name, "_proj"):
		return false
	default:
		return true
	}
}

// GetTensorQuantization returns the appropriate quantization type for a tensor.
// Returns "" if the tensor should not be quantized.
// This implements mixed-precision quantization:
//   - v_proj, k_proj, down_proj: promoted to INT8 when base is INT4
//   - Norms, embeddings, biases, routing gates: no quantization
//   - All other eligible weights: use requested quantization type
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

	// Skip routing gate weights (should stay high precision)
	// In safetensors these are: mlp.gate.weight (not mlp.gate_proj.weight)
	if strings.Contains(name, "mlp.gate.weight") && !strings.Contains(name, "_proj") {
		return ""
	}

	// MLX quantization requires last dimension to be divisible by group size.
	if !isAligned(shape, quantNorm) {
		return ""
	}

	// For non-affine modes, use the same quantization for all eligible tensors.
	if quantNorm == "nvfp4" || quantNorm == "mxfp4" || quantNorm == "mxfp8" {
		return quantNorm
	}

	// Value projection weights directly determine attention output quality.
	// Down projection weights feed directly into the residual stream where
	// errors accumulate across layers. Both benefit from higher precision.
	// Promote to INT8 when base is INT4 (same affine mode, compatible with
	// GatherQMM for MoE expert tensors).
	if quantNorm == "int4" {
		if strings.Contains(name, ".v_proj") || strings.Contains(name, ".k_proj") || strings.Contains(name, "down_proj") {
			if isAligned(shape, "int8") {
				return "int8"
			}
		}
	}

	return quantNorm
}

var expertLayerPrefixRegexp = regexp.MustCompile(`^(?:model\.language_model\.|language_model(?:\.model)?\.|model\.)?layers\.\d+$`)
var prequantizedExpertSuffixRegexp = regexp.MustCompile(`^\.(\d+)\.(.+)$`)

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
		".moe.experts.",
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
	Format          string  `json:"format"`
	QuantMethod     string  `json:"quant_method"`
	WeightBlockSize []int32 `json:"weight_block_size"`
	ConfigGroups    map[string]struct {
		Format  string `json:"format"`
		Weights struct {
			BlockStructure []int32 `json:"block_structure"`
			NumBits        int     `json:"num_bits"`
			Type           string  `json:"type"`
		} `json:"weights"`
	} `json:"config_groups"`
}

type sourceModelConfig struct {
	ModelType          string             `json:"model_type"`
	Architectures      []string           `json:"architectures"`
	Quantization       sourceQuantization `json:"quantization"`
	QuantizationConfig sourceQuantization `json:"quantization_config"`
	CompressionConfig  sourceQuantization `json:"compression_config"`
	TextConfig         struct {
		ModelType          string             `json:"model_type"`
		Quantization       sourceQuantization `json:"quantization"`
		QuantizationConfig sourceQuantization `json:"quantization_config"`
		CompressionConfig  sourceQuantization `json:"compression_config"`
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
		cfg.CompressionConfig,
		cfg.TextConfig.Quantization,
		cfg.TextConfig.QuantizationConfig,
		cfg.TextConfig.CompressionConfig,
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
	sourceQuantizedKindSourceFP8    sourceQuantizedKind = "source_fp8"
)

func (cfg sourceModelConfig) quantizationConfigs() []sourceQuantization {
	return []sourceQuantization{
		cfg.Quantization,
		cfg.QuantizationConfig,
		cfg.CompressionConfig,
		cfg.TextConfig.Quantization,
		cfg.TextConfig.QuantizationConfig,
		cfg.TextConfig.CompressionConfig,
	}
}

func (cfg sourceModelConfig) HFFP8WeightBlockSize() (rows, cols int32, ok bool) {
	for _, q := range cfg.quantizationConfigs() {
		if !strings.EqualFold(q.QuantMethod, "fp8") || len(q.WeightBlockSize) != 2 {
			if !strings.EqualFold(q.QuantMethod, "compressed-tensors") && !strings.EqualFold(q.Format, "float-quantized") {
				continue
			}
			for _, group := range q.ConfigGroups {
				if !strings.EqualFold(group.Format, "float-quantized") || group.Weights.NumBits != 8 || !strings.EqualFold(group.Weights.Type, "float") || len(group.Weights.BlockStructure) != 2 {
					continue
				}
				return group.Weights.BlockStructure[0], group.Weights.BlockStructure[1], true
			}
			continue
		}
		return q.WeightBlockSize[0], q.WeightBlockSize[1], true
	}
	return 0, 0, false
}

func (cfg sourceModelConfig) hasPackedNVFP4Format() bool {
	for _, q := range cfg.quantizationConfigs() {
		if strings.EqualFold(q.Format, "nvfp4-pack-quantized") {
			return true
		}
	}
	return false
}

func inspectSourceQuantization(modelDir string, cfg sourceModelConfig) (sourceQuantizedKind, error) {
	// Check for NVIDIA ModelOpt hf_quant_config.json (NVFP4)
	if detectModelOptQuantization(modelDir) {
		return sourceQuantizedKindPrequantized, nil
	}

	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return sourceQuantizedKindNone, err
	}

	hasFP8Scale := false
	hasPackedNVFP4 := false
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
			case strings.HasSuffix(name, ".weight_packed"):
				hasPackedNVFP4 = true
			case strings.HasSuffix(name, ".weight_scale_inv"):
				hasFP8Scale = true
			case strings.HasSuffix(name, ".weight_scale"):
				hasFP8Scale = true
			}
		}

		extractor.Close()
	}

	if hasPackedNVFP4 && cfg.hasPackedNVFP4Format() {
		return sourceQuantizedKindPrequantized, nil
	}

	if hasFP8Scale {
		if _, _, ok := cfg.HFFP8WeightBlockSize(); ok {
			return sourceQuantizedKindSourceFP8, nil
		}
	}

	return sourceQuantizedKindNone, nil
}

// modelOptQuantConfig represents the hf_quant_config.json format from
// NVIDIA ModelOpt (TensorRT Model Optimizer).
type modelOptQuantConfig struct {
	Producer struct {
		Name    string `json:"name"`
		Version string `json:"version"`
	} `json:"producer"`
	Quantization struct {
		QuantAlgo      string   `json:"quant_algo"`
		GroupSize      int      `json:"group_size"`
		ExcludeModules []string `json:"exclude_modules"`
	} `json:"quantization"`
}

func detectModelOptQuantization(modelDir string) bool {
	data, err := os.ReadFile(filepath.Join(modelDir, "hf_quant_config.json"))
	if err != nil {
		return false
	}
	var cfg modelOptQuantConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return false
	}
	return strings.ToUpper(cfg.Quantization.QuantAlgo) == "NVFP4"
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
	case sourceQuantizedKindSourceFP8:
		rows, cols, ok := cfg.HFFP8WeightBlockSize()
		if !ok {
			return "", fmt.Errorf("fp8 source model missing weight_block_size metadata")
		}
		if rows != 128 || cols != 128 {
			return "", fmt.Errorf("unsupported fp8 source block size %dx%d", rows, cols)
		}
		if requested != "" {
			requested = normalizeQuantType(requested)
			switch requested {
			case "nvfp4", "mxfp4", "mxfp8":
				return requested, nil
			default:
				return "", fmt.Errorf("cannot convert already-quantized fp8 source model with --quantize %q", requested)
			}
		}
		return "mxfp8", nil
	default:
		return "", fmt.Errorf("unsupported source quantization kind %q", sourceKind)
	}
}

func importQuantizationStatus(sourceKind sourceQuantizedKind, effectiveQuantize string) string {
	if effectiveQuantize == "" {
		if sourceKind == sourceQuantizedKindPrequantized {
			return ", preserving source quantization"
		}
		return ""
	}
	switch sourceKind {
	case sourceQuantizedKindSourceFP8:
		return fmt.Sprintf(", converting source E4M3 block-FP8 to MLX %s", effectiveQuantize)
	default:
		return fmt.Sprintf(", quantizing to %s", effectiveQuantize)
	}
}

type tensorImportTransform interface {
	skipTensor(name string) bool
	transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error)
	quantizationType(name string, shape []int32, quantize string) string
}

type sourceFP8TensorImportTransform interface {
	sourceFP8TensorQuantization(name string, shape []int32, requested string, fallback string) string
	sourceFP8BF16Quantization(name string, shape []int32, requested string) string
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
	"Gemma4ForCausalLM":                    newGemma4ImportTransform,
	"Gemma4ForConditionalGeneration":       newGemma4ImportTransform,
	"LagunaForCausalLM":                    newLagunaImportTransform,
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
	sourceTensorFiles, err := readSourceTensorFiles(modelDir)
	if err != nil {
		return fmt.Errorf("failed to read source tensor index: %w", err)
	}
	importTransform, err := newTensorImportTransform(modelDir, sourceConfig)
	if err != nil {
		return fmt.Errorf("failed to construct import transform for architecture %q: %w", sourceConfig.Architecture(), err)
	}
	sourceFP8Transform, _ := importTransform.(sourceFP8TensorImportTransform)

	// Resolve the optional packed layer creator
	var packedCreator PackedTensorLayerCreator
	if len(createPackedLayer) > 0 {
		packedCreator = createPackedLayer[0]
	}
	// Accumulate expert tensors by group prefix for packing.
	// Readers reference file-backed SectionReaders, so we keep extractors
	// open until each group is flushed to avoid buffering tensor data in memory.
	expertGroups := make(map[string][]PackedTensorInput)
	prequantizedExpertGroups := make(map[string][]*safetensors.TensorData)
	var expertGroupOrder []string

	// Track open extractors so we can close them after flushing groups
	var openExtractors []*safetensors.TensorExtractor
	crossFileExtractors := make(map[string]*safetensors.TensorExtractor)

	closeExtractors := func() {
		for _, ext := range openExtractors {
			ext.Close()
		}
		openExtractors = nil
		for _, ext := range crossFileExtractors {
			ext.Close()
		}
		clear(crossFileExtractors)
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
		fn(fmt.Sprintf("importing %s (%d tensors%s)", entry.Name(), len(tensorNames), importQuantizationStatus(sourceQuantKind, effectiveQuantize)))

		// Track whether this extractor has expert tensors that need to stay open
		hasExpertTensors := false

		for _, tensorName := range tensorNames {
			if importTransform.skipTensor(tensorName) {
				continue
			}
			if shouldSkipSourceCompanion(tensorName, tensorSet, sourceTensorFiles) {
				continue
			}
			sourceFP8ScaleName, hasSourceFP8Scale := sourceFP8Companion(tensorName, tensorSet, sourceTensorFiles)

			td, err := extractor.GetTensor(tensorName)
			if err != nil {
				extractor.Close()
				closeExtractors()
				return fmt.Errorf("failed to get tensor %s: %w", tensorName, err)
			}

			if packedCreator != nil {
				if packedWeightName := strings.TrimSuffix(tensorName, "_packed"); packedWeightName != tensorName {
					groupPrefix := ExpertGroupPrefix(packedWeightName)
					if groupPrefix != "" {
						packedTensors, ok, err := packedNVFP4TensorData(modelDir, extractor, crossFileExtractors, td, tensorName, tensorSet, sourceTensorFiles)
						if err != nil {
							extractor.Close()
							closeExtractors()
							return err
						}
						if ok {
							hasExpertTensors = true
							if _, exists := prequantizedExpertGroups[groupPrefix]; !exists {
								expertGroupOrder = append(expertGroupOrder, groupPrefix)
							}
							prequantizedExpertGroups[groupPrefix] = append(prequantizedExpertGroups[groupPrefix], packedTensors...)
							continue
						}
					}
				}
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
				layer, ok, err = createPackedNVFP4Layer(modelDir, extractor, crossFileExtractors, td, tensorName, tensorSet, sourceTensorFiles, sourceQuantMetadata, createLayer)
				if err != nil {
					extractor.Close()
					closeExtractors()
					return err
				}
				if ok {
					layers = append(layers, layer)
					continue
				}
				// Try ModelOpt NVFP4 format (weight_scale + weight_scale_2)
				layer, ok, err = createModelOptFP4Layer(extractor, td, tensorName, tensorSet, sourceQuantMetadata, createLayer)
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
				case sourceQuantKind == sourceQuantizedKindSourceFP8 && hasSourceFP8Scale:
					quantizeType = importTransform.quantizationType(outTD.Name, outTD.Shape, effectiveQuantize)
					if quantizeType == "" && effectiveQuantize == "mxfp8" {
						// Source FP8 tensors are already quantized weights and small
						// synthetic tests may not pass the generic import size filter.
						quantizeType = "mxfp8"
					}
					if sourceFP8Transform != nil {
						quantizeType = sourceFP8Transform.sourceFP8TensorQuantization(outTD.Name, outTD.Shape, quantize, quantizeType)
					} else {
						quantizeType = sourceFP8TensorQuantization(outTD.Name, outTD.Shape, quantize, quantizeType)
					}
				case sourceQuantKind == sourceQuantizedKindSourceFP8:
					if sourceFP8Transform != nil {
						quantizeType = sourceFP8Transform.sourceFP8BF16Quantization(outTD.Name, outTD.Shape, quantize)
					} else {
						quantizeType = sourceFP8BF16PromotionQuantization(outTD.Name, outTD.Shape, quantize)
					}
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
						return fmt.Errorf("source fp8 tensor %s was not scheduled for %s conversion", tensorName, effectiveQuantize)
					}
					scaleTD, err := getTensorFromSource(modelDir, extractor, crossFileExtractors, sourceTensorFiles, sourceFP8ScaleName)
					if err != nil {
						extractor.Close()
						closeExtractors()
						return fmt.Errorf("failed to get fp8 scale tensor %s: %w", sourceFP8ScaleName, err)
					}
					reader = buildSourceFP8Reader(outTD, scaleTD)
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
			if tensors := prequantizedExpertGroups[groupName]; len(tensors) > 0 {
				layer, ok, err := createPackedNVFP4ExpertGroupLayer(groupName, tensors, createLayer)
				if err != nil {
					closeExtractors()
					return fmt.Errorf("failed to create packed prequantized layer for %s: %w", groupName, err)
				}
				if ok {
					layers = append(layers, layer)
					continue
				}
				layer, err = createLayer(
					safetensors.BuildPackedSafetensorsReaderWithMetadata(tensors, map[string]string{
						"quant_type": "nvfp4",
						"group_size": "16",
					}),
					"application/vnd.ollama.image.tensor",
					groupName,
				)
				if err != nil {
					closeExtractors()
					return fmt.Errorf("failed to create packed prequantized layer for %s: %w", groupName, err)
				}
				layers = append(layers, layer)
				continue
			}
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

func shouldSkipSourceCompanion(name string, tensorSet map[string]struct{}, sourceTensorFiles map[string]string) bool {
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
	case strings.HasSuffix(name, ".weight_scale"):
		base := strings.TrimSuffix(name, "_scale")
		if _, ok := tensorSet[base]; ok {
			return true
		}
		if _, ok := sourceTensorFiles[base+"_packed"]; ok {
			return true
		}
		_, ok := tensorSet[base+"_packed"]
		return ok
	// ModelOpt NVFP4 companion tensors
	case strings.HasSuffix(name, ".weight_scale_2"):
		_, ok := tensorSet[strings.TrimSuffix(name, "_scale_2")]
		return ok
	case strings.HasSuffix(name, ".input_scale"):
		// Activation scale for ModelOpt — not needed for weight-only inference
		base := strings.TrimSuffix(name, ".input_scale")
		_, ok := tensorSet[base+".weight"]
		return ok
	case strings.HasSuffix(name, ".weight_global_scale"):
		base := strings.TrimSuffix(name, ".weight_global_scale")
		if _, ok := sourceTensorFiles[base+".weight_packed"]; ok {
			return true
		}
		_, ok := tensorSet[base+".weight_packed"]
		return ok
	case strings.HasSuffix(name, ".input_global_scale"):
		base := strings.TrimSuffix(name, ".input_global_scale")
		if _, ok := sourceTensorFiles[base+".weight_packed"]; ok {
			return true
		}
		_, ok := tensorSet[base+".weight_packed"]
		return ok
	default:
		return false
	}
}

func sourceFP8Companion(weightName string, tensorSet map[string]struct{}, sourceTensorFiles map[string]string) (scaleName string, ok bool) {
	if !strings.HasSuffix(weightName, ".weight") {
		return "", false
	}

	scaleName = weightName + "_scale_inv"
	if _, ok = tensorSet[scaleName]; ok {
		return scaleName, true
	}
	if _, ok = sourceTensorFiles[scaleName]; ok {
		return scaleName, true
	}
	scaleName = weightName + "_scale"
	if _, ok = tensorSet[scaleName]; ok {
		return scaleName, true
	}
	_, ok = sourceTensorFiles[scaleName]
	return scaleName, ok
}

func buildSourceFP8Reader(weightTD, scaleTD *safetensors.TensorData) io.Reader {
	scaleName := weightTD.Name + ".scale_inv"
	if strings.HasSuffix(scaleTD.Name, "_scale") && !strings.HasSuffix(scaleTD.Name, "_scale_inv") {
		scaleName = weightTD.Name + ".scale"
	}
	return safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{weightTD, scaleTD.WithName(scaleName)})
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

// createModelOptFP4Layer creates a pre-quantized layer from NVIDIA ModelOpt
// NVFP4 tensors. The weight (U8) and scale (F8_E4M3 stored as uint8) are
// packed with the per-tensor global scale (weight_scale_2) into a single
// safetensors blob. The tensor names are mapped to our standard format:
//   - source.weight → tensorName (weight data, kept as-is)
//   - source.weight_scale → tensorName.scale (FP8 E4M3 bytes as uint8)
//   - source.weight_scale_2 → tensorName.global_scale (F32 scalar)
func createModelOptFP4Layer(
	extractor *safetensors.TensorExtractor,
	td *safetensors.TensorData,
	tensorName string,
	tensorSet map[string]struct{},
	metadata map[string]string,
	createLayer LayerCreator,
) (LayerInfo, bool, error) {
	scaleName, globalScaleName, ok := modelOptFP4Companions(tensorName, tensorSet)
	if !ok {
		return LayerInfo{}, false, nil
	}

	// NVIDIA packs FP4 as U8 (2 values/byte), MLX expects U32 (8 values/uint32).
	// Repack: view the U8 data as U32 (4 consecutive bytes → 1 uint32) and
	// adjust the shape from [out, in/2] to [out, in/8].
	weightTD := td.WithName(tensorName)
	if strings.ToUpper(weightTD.Dtype) == "U8" && len(weightTD.Shape) == 2 {
		weightTD.Dtype = "U32"
		weightTD.Shape = []int32{weightTD.Shape[0], weightTD.Shape[1] / 4}
	}
	tensors := []*safetensors.TensorData{weightTD}

	scaleTD, err := extractor.GetTensor(scaleName)
	if err != nil {
		return LayerInfo{}, false, fmt.Errorf("failed to get tensor %s: %w", scaleName, err)
	}
	// F8_E4M3 scales stored as uint8 — fix the dtype for our loader
	scaleRenamed := scaleTD.WithName(tensorName + ".scale")
	if strings.ToUpper(scaleRenamed.Dtype) == "F8_E4M3" {
		scaleRenamed.Dtype = "U8"
	}
	tensors = append(tensors, scaleRenamed)

	if globalScaleName != "" {
		gsTD, err := extractor.GetTensor(globalScaleName)
		if err != nil {
			return LayerInfo{}, false, fmt.Errorf("failed to get tensor %s: %w", globalScaleName, err)
		}
		gsTD, err = validateScalarFloat32TensorData(gsTD, tensorName+".global_scale")
		if err != nil {
			return LayerInfo{}, false, fmt.Errorf("failed to normalize tensor %s: %w", globalScaleName, err)
		}
		tensors = append(tensors, gsTD)
	}

	// Add nvfp4 quant metadata
	md := make(map[string]string)
	for k, v := range metadata {
		md[k] = v
	}
	md["quant_type"] = "nvfp4"

	layer, err := createLayer(
		safetensors.BuildPackedSafetensorsReaderWithMetadata(tensors, md),
		"application/vnd.ollama.image.tensor",
		tensorName,
	)
	if err != nil {
		return LayerInfo{}, false, fmt.Errorf("failed to create ModelOpt FP4 layer for %s: %w", tensorName, err)
	}
	return layer, true, nil
}

// createPackedNVFP4Layer creates a pre-quantized layer from packed NVFP4
// tensors that use the newer source layout:
//   - source.weight_packed -> tensorName (U32 repacked weight)
//   - source.weight_scale -> tensorName.scale
//   - source.weight_global_scale -> reciprocal stored as tensorName.global_scale
//   - source.input_global_scale -> ignored for weight-only inference
func createPackedNVFP4Layer(
	modelDir string,
	extractor *safetensors.TensorExtractor,
	crossFileExtractors map[string]*safetensors.TensorExtractor,
	td *safetensors.TensorData,
	tensorName string,
	tensorSet map[string]struct{},
	sourceTensorFiles map[string]string,
	metadata map[string]string,
	createLayer LayerCreator,
) (LayerInfo, bool, error) {
	weightName, scaleName, weightGlobalScaleName, _, ok := packedNVFP4Companions(tensorName, tensorSet, sourceTensorFiles)
	if !ok {
		return LayerInfo{}, false, nil
	}

	weightTD := td.WithName(weightName)
	if strings.ToUpper(weightTD.Dtype) == "U8" && len(weightTD.Shape) == 2 {
		weightTD.Dtype = "U32"
		weightTD.Shape = []int32{weightTD.Shape[0], weightTD.Shape[1] / 4}
	}
	tensors := []*safetensors.TensorData{weightTD}

	scaleTD, err := getTensorFromSource(modelDir, extractor, crossFileExtractors, sourceTensorFiles, scaleName)
	if err != nil {
		return LayerInfo{}, false, fmt.Errorf("failed to get tensor %s: %w", scaleName, err)
	}
	scaleRenamed := scaleTD.WithName(weightName + ".scale")
	if strings.ToUpper(scaleRenamed.Dtype) == "F8_E4M3" {
		scaleRenamed.Dtype = "U8"
	}
	tensors = append(tensors, scaleRenamed)

	if weightGlobalScaleName != "" {
		gsTD, err := getTensorFromSource(modelDir, extractor, crossFileExtractors, sourceTensorFiles, weightGlobalScaleName)
		if err != nil {
			return LayerInfo{}, false, fmt.Errorf("failed to get tensor %s: %w", weightGlobalScaleName, err)
		}
		gsTD, err = invertScalarFloat32TensorData(gsTD, weightName+".global_scale")
		if err != nil {
			return LayerInfo{}, false, fmt.Errorf("failed to normalize tensor %s: %w", weightGlobalScaleName, err)
		}
		tensors = append(tensors, gsTD)
	}

	md := make(map[string]string)
	for k, v := range metadata {
		md[k] = v
	}
	md["quant_type"] = "nvfp4"
	if _, ok := md["group_size"]; !ok {
		md["group_size"] = "16"
	}

	layer, err := createLayer(
		safetensors.BuildPackedSafetensorsReaderWithMetadata(tensors, md),
		"application/vnd.ollama.image.tensor",
		weightName,
	)
	if err != nil {
		return LayerInfo{}, false, fmt.Errorf("failed to create packed NVFP4 layer for %s: %w", tensorName, err)
	}
	return layer, true, nil
}

type stackedTempTensor struct {
	tensor *safetensors.TensorData
	file   *os.File
	path   string
}

func createPackedNVFP4ExpertGroupLayer(groupName string, tensors []*safetensors.TensorData, createLayer LayerCreator) (LayerInfo, bool, error) {
	stacked, metadata, ok, err := stackPackedNVFP4ExpertGroup(groupName, tensors)
	if err != nil || !ok {
		return LayerInfo{}, ok, err
	}
	defer func() {
		for _, td := range stacked {
			if td.file != nil {
				td.file.Close()
			}
			if td.path != "" {
				os.Remove(td.path)
			}
		}
	}()

	packed := make([]*safetensors.TensorData, 0, len(stacked))
	for _, td := range stacked {
		packed = append(packed, td.tensor)
	}
	layer, err := createLayer(
		safetensors.BuildPackedSafetensorsReaderWithMetadata(packed, metadata),
		"application/vnd.ollama.image.tensor",
		groupName,
	)
	if err != nil {
		return LayerInfo{}, true, err
	}
	return layer, true, nil
}

func stackPackedNVFP4ExpertGroup(groupName string, tensors []*safetensors.TensorData) ([]stackedTempTensor, map[string]string, bool, error) {
	if !strings.HasSuffix(groupName, ".experts") {
		return nil, nil, false, nil
	}

	type namedExpertTensor struct {
		expert int
		name   string
		td     *safetensors.TensorData
	}

	grouped := make(map[string][]namedExpertTensor)
	for _, td := range tensors {
		suffix := strings.TrimPrefix(td.Name, groupName)
		m := prequantizedExpertSuffixRegexp.FindStringSubmatch(suffix)
		if m == nil {
			return nil, nil, false, nil
		}
		expert, err := strconv.Atoi(m[1])
		if err != nil {
			return nil, nil, false, fmt.Errorf("invalid expert index in %q: %w", td.Name, err)
		}
		grouped[m[2]] = append(grouped[m[2]], namedExpertTensor{
			expert: expert,
			name:   td.Name,
			td:     td,
		})
	}
	if len(grouped) == 0 {
		return nil, nil, false, nil
	}

	groupBase := strings.TrimSuffix(groupName, ".experts") + ".switch_mlp."
	names := make([]string, 0, len(grouped))
	for name := range grouped {
		names = append(names, name)
	}
	sort.Strings(names)

	var stacked []stackedTempTensor
	metadata := map[string]string{
		"quant_type": "nvfp4",
		"group_size": "16",
	}
	cleanup := func() {
		for _, td := range stacked {
			if td.file != nil {
				td.file.Close()
			}
			if td.path != "" {
				os.Remove(td.path)
			}
		}
	}

	for _, name := range names {
		if strings.HasSuffix(name, ".input_global_scale") {
			continue
		}
		experts := grouped[name]
		sort.Slice(experts, func(i, j int) bool { return experts[i].expert < experts[j].expert })
		if len(experts) == 0 {
			continue
		}

		stackedName := groupBase + name
		baseShape := append([]int32(nil), experts[0].td.Shape...)
		stackedShape := make([]int32, 0, len(baseShape)+1)
		stackedShape = append(stackedShape, int32(len(experts)))
		switch {
		case strings.HasSuffix(name, ".global_scale"), strings.HasSuffix(name, ".input_global_scale"):
			stackedShape = append(stackedShape, 1, 1)
		default:
			stackedShape = append(stackedShape, baseShape...)
		}

		f, err := os.CreateTemp("", "ollama-packed-nvfp4-*.bin")
		if err != nil {
			cleanup()
			return nil, nil, false, fmt.Errorf("create temp tensor for %s: %w", stackedName, err)
		}

		var size int64
		for _, expert := range experts {
			if expert.td.Dtype != experts[0].td.Dtype || !slices.Equal(expert.td.Shape, experts[0].td.Shape) {
				f.Close()
				os.Remove(f.Name())
				cleanup()
				return nil, nil, false, fmt.Errorf("mismatched expert tensor layout in %s", stackedName)
			}
			written, err := io.Copy(f, expert.td.Reader())
			if err != nil {
				f.Close()
				os.Remove(f.Name())
				cleanup()
				return nil, nil, false, fmt.Errorf("stack tensor %s: %w", expert.name, err)
			}
			size += written
		}

		stacked = append(stacked, stackedTempTensor{
			tensor: safetensors.NewTensorDataFromReaderAt(stackedName, experts[0].td.Dtype, stackedShape, f, size),
			file:   f,
			path:   f.Name(),
		})

		if strings.HasSuffix(name, ".weight") {
			metadata[stackedName+".quant_type"] = "nvfp4"
			metadata[stackedName+".group_size"] = "16"
		}
	}

	return stacked, metadata, true, nil
}

func packedNVFP4TensorData(
	modelDir string,
	extractor *safetensors.TensorExtractor,
	crossFileExtractors map[string]*safetensors.TensorExtractor,
	td *safetensors.TensorData,
	tensorName string,
	tensorSet map[string]struct{},
	sourceTensorFiles map[string]string,
) ([]*safetensors.TensorData, bool, error) {
	weightName, scaleName, weightGlobalScaleName, _, ok := packedNVFP4Companions(tensorName, tensorSet, sourceTensorFiles)
	if !ok {
		return nil, false, nil
	}

	weightTD := td.WithName(weightName)
	if strings.ToUpper(weightTD.Dtype) == "U8" && len(weightTD.Shape) == 2 {
		weightTD.Dtype = "U32"
		weightTD.Shape = []int32{weightTD.Shape[0], weightTD.Shape[1] / 4}
	}
	tensors := []*safetensors.TensorData{weightTD}

	scaleTD, err := getTensorFromSource(modelDir, extractor, crossFileExtractors, sourceTensorFiles, scaleName)
	if err != nil {
		return nil, false, fmt.Errorf("failed to get tensor %s: %w", scaleName, err)
	}
	scaleRenamed := scaleTD.WithName(weightName + ".scale")
	if strings.ToUpper(scaleRenamed.Dtype) == "F8_E4M3" {
		scaleRenamed.Dtype = "U8"
	}
	tensors = append(tensors, scaleRenamed)

	if weightGlobalScaleName != "" {
		gsTD, err := getTensorFromSource(modelDir, extractor, crossFileExtractors, sourceTensorFiles, weightGlobalScaleName)
		if err != nil {
			return nil, false, fmt.Errorf("failed to get tensor %s: %w", weightGlobalScaleName, err)
		}
		gsTD, err = invertScalarFloat32TensorData(gsTD, weightName+".global_scale")
		if err != nil {
			return nil, false, fmt.Errorf("failed to normalize tensor %s: %w", weightGlobalScaleName, err)
		}
		tensors = append(tensors, gsTD)
	}

	return tensors, true, nil
}

func validateScalarFloat32TensorData(td *safetensors.TensorData, name string) (*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	if strings.ToUpper(td.Dtype) != "F32" {
		return nil, fmt.Errorf("expected F32 tensor, got %s", td.Dtype)
	}
	n := int32(1)
	for _, dim := range td.Shape {
		n *= dim
	}
	if n != 1 {
		return nil, fmt.Errorf("expected scalar F32 tensor, got shape %v", td.Shape)
	}
	return td.WithName(name), nil
}

func invertScalarFloat32TensorData(td *safetensors.TensorData, name string) (*safetensors.TensorData, error) {
	td, err := validateScalarFloat32TensorData(td, name)
	if err != nil {
		return nil, err
	}
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, err
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("invalid F32 tensor byte length %d", len(raw))
	}
	out := make([]byte, len(raw))
	for i := 0; i < len(raw); i += 4 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[i : i+4]))
		if v == 0 {
			return nil, fmt.Errorf("cannot invert zero F32 scale")
		}
		binary.LittleEndian.PutUint32(out[i:i+4], math.Float32bits(1/v))
	}
	return safetensors.NewTensorDataFromBytes(name, td.Dtype, td.Shape, out), nil
}

// modelOptFP4Companions finds the companion tensors for a ModelOpt NVFP4
// quantized weight: weight_scale (per-group FP8 E4M3 scales) and optional
// weight_scale_2 (per-tensor global scale).
func modelOptFP4Companions(weightName string, tensorSet map[string]struct{}) (scaleName, globalScaleName string, ok bool) {
	if !strings.HasSuffix(weightName, ".weight") {
		return "", "", false
	}

	scaleName = weightName + "_scale"
	if _, ok := tensorSet[scaleName]; !ok {
		return "", "", false
	}

	globalScaleName = weightName + "_scale_2"
	if _, ok := tensorSet[globalScaleName]; !ok {
		globalScaleName = ""
	}
	return scaleName, globalScaleName, true
}

func packedNVFP4Companions(weightPackedName string, tensorSet map[string]struct{}, sourceTensorFiles map[string]string) (weightName, scaleName, weightGlobalScaleName, inputGlobalScaleName string, ok bool) {
	if !strings.HasSuffix(weightPackedName, ".weight_packed") {
		return "", "", "", "", false
	}

	weightName = strings.TrimSuffix(weightPackedName, "_packed")
	scaleName = strings.TrimSuffix(weightPackedName, "_packed") + "_scale"
	if _, ok := tensorSet[scaleName]; !ok {
		if _, ok := sourceTensorFiles[scaleName]; !ok {
			return "", "", "", "", false
		}
	}

	weightGlobalScaleName = strings.TrimSuffix(weightPackedName, "_packed") + "_global_scale"
	if _, ok := tensorSet[weightGlobalScaleName]; !ok {
		if _, ok := sourceTensorFiles[weightGlobalScaleName]; !ok {
			weightGlobalScaleName = ""
		}
	}

	inputGlobalScaleName = strings.TrimSuffix(weightPackedName, ".weight_packed") + ".input_global_scale"
	if _, ok := tensorSet[inputGlobalScaleName]; !ok {
		if _, ok := sourceTensorFiles[inputGlobalScaleName]; !ok {
			inputGlobalScaleName = ""
		}
	}

	return weightName, scaleName, weightGlobalScaleName, inputGlobalScaleName, true
}

func readSourceTensorFiles(modelDir string) (map[string]string, error) {
	indexPath := filepath.Join(modelDir, "model.safetensors.index.json")
	data, err := os.ReadFile(indexPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, err
	}
	return index.WeightMap, nil
}

func getTensorFromSource(modelDir string, current *safetensors.TensorExtractor, cache map[string]*safetensors.TensorExtractor, sourceTensorFiles map[string]string, name string) (*safetensors.TensorData, error) {
	if td, err := current.GetTensor(name); err == nil {
		return td, nil
	}
	if sourceTensorFiles == nil {
		return nil, fmt.Errorf("tensor %s not found in current shard and no source index available", name)
	}
	fileName, ok := sourceTensorFiles[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s not found in source index", name)
	}
	ext := cache[fileName]
	if ext == nil {
		path := filepath.Join(modelDir, fileName)
		var err error
		ext, err = safetensors.OpenForExtraction(path)
		if err != nil {
			return nil, err
		}
		cache[fileName] = ext
	}
	return ext.GetTensor(name)
}
