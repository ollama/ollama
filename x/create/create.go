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

// IsSafetensorsLLMModel checks if a model is a safetensors LLM model
// (has completion capability, not image generation).
func IsSafetensorsLLMModel(modelName string) bool {
	config, err := loadModelConfig(modelName)
	if err != nil {
		return false
	}
	return config.ModelFormat == "safetensors" && slices.Contains(config.Capabilities, "completion")
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

	// ".experts." covers the common case (.mlp.experts., .moe.experts.) as well
	// as gemma's bare "...layers.N.experts.gate_up_proj" (no .mlp/.moe prefix).
	return strings.Contains(name, ".experts.") ||
		strings.Contains(name, ".mlp.switch_mlp.") ||
		strings.Contains(name, ".mlp.shared_experts.")
}

// isRoutingGate reports the small MoE routing/gate weights that select the
// active experts. Quantization noise there can flip expert selection, so they
// are kept at source precision regardless of architecture.
func isRoutingGate(name string) bool {
	return strings.HasSuffix(name, ".mlp.gate.weight") ||
		strings.HasSuffix(name, ".shared_expert_gate.weight") ||
		strings.HasSuffix(name, ".router.proj.weight")
}

// GetTensorQuantization returns the appropriate quantization type for a tensor.
// Returns "" if the tensor should not be quantized.
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

	// Routing gates are tiny and selection-sensitive — keep them at source precision.
	if isRoutingGate(name) {
		return ""
	}

	// lm_head is too sensitive for the fp quant modes; keep it at source precision.
	if strings.HasSuffix(name, "lm_head.weight") && (quantNorm == "nvfp4" || quantNorm == "mxfp4" || quantNorm == "mxfp8") {
		return ""
	}

	// MLX quantization requires last dimension to be divisible by group size.
	if !isAligned(shape, quantNorm) {
		return ""
	}

	// Promote sensitive projections to 8-bit; fp4 skips experts since their kernels take a single mode.
	if quantNorm == "int4" || ((quantNorm == "nvfp4" || quantNorm == "mxfp4") && !stackedExpert) {
		if strings.Contains(name, ".v_proj") || strings.Contains(name, ".k_proj") || strings.Contains(name, "down_proj") {
			if e := eightBit(quantNorm); isAligned(shape, e) {
				return e
			}
		}
	}

	return quantNorm
}

var (
	expertLayerPrefixRegexp = regexp.MustCompile(`^(?:model\.language_model\.|language_model(?:\.model)?\.|model\.)?layers\.\d+$`)
)

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

// readSourceModelConfig parses config.json into the shared sourceModelConfig
// and returns the raw bytes alongside it. The raw bytes are retained on the
// Inventory so architecture-specific factories can parse their own fields
// without re-opening the file.
func readSourceModelConfig(modelDir string) (sourceModelConfig, json.RawMessage, error) {
	configPath := filepath.Join(modelDir, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return sourceModelConfig{}, nil, err
	}

	var cfg sourceModelConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return sourceModelConfig{}, nil, err
	}

	return cfg, data, nil
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
	for _, candidate := range cfg.quantizationConfigs() {
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

type tensorImportTransformFactory func(rawConfig json.RawMessage) (quantizePolicy, error)

var tensorImportTransformRegistry = map[string]tensorImportTransformFactory{
	"Qwen3_5ForCausalLM":                    newQwen35ImportTransform,
	"Qwen3_5ForConditionalGeneration":       newQwen35ImportTransform,
	"Qwen3NextForCausalLM":                  newQwen35ImportTransform,
	"Qwen3NextForConditionalGeneration":     newQwen35ImportTransform,
	"Qwen3_5MoeForCausalLM":                 newQwen35ImportTransform,
	"Qwen3_5MoeForConditionalGeneration":    newQwen35ImportTransform,
	"Qwen3NextMoeForCausalLM":               newQwen35ImportTransform,
	"Qwen3NextMoeForConditionalGeneration":  newQwen35ImportTransform,
	"Gemma4ForCausalLM":                     newGemma4ImportTransform,
	"Gemma4ForConditionalGeneration":        newGemma4ImportTransform,
	"Gemma4UnifiedForCausalLM":              newGemma4ImportTransform,
	"Gemma4UnifiedForConditionalGeneration": newGemma4ImportTransform,
	"gemma4_unified":                        newGemma4ImportTransform,
	"gemma4_unified_text":                   newGemma4ImportTransform,
	"LagunaForCausalLM":                     newLagunaImportTransform,
	"Cohere2MoeForCausalLM":                 newCohere2MoeImportTransform,
	"Gemma4AssistantForCausalLM":            newGemma4ImportTransform,
	"Gemma4UnifiedAssistantForCausalLM":     newGemma4ImportTransform,
	"gemma4_unified_assistant":              newGemma4ImportTransform,
}

func newTensorImportTransform(inv Inventory) (quantizePolicy, error) {
	if factory, ok := tensorImportTransformRegistry[inv.Config.Architecture()]; ok {
		return factory(inv.RawConfig)
	}
	return defaultQuantPolicy{}, nil
}

func buildSourceFP8Reader(weightTD, scaleTD *safetensors.TensorData) io.Reader {
	scaleName := weightTD.Name + ".scale_inv"
	if strings.HasSuffix(scaleTD.Name, "_scale") && !strings.HasSuffix(scaleTD.Name, "_scale_inv") {
		scaleName = weightTD.Name + ".scale"
	}
	return safetensors.BuildPackedSafetensorsReader([]*safetensors.TensorData{weightTD, scaleTD.WithName(scaleName)})
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
