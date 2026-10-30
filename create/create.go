package create

import (
	"cmp"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx/quant"
	"github.com/ollama/ollama/types/model"
)

// SafetensorsMinOllamaVersion is the minimum Ollama version required for
// safetensors-backed models.
const SafetensorsMinOllamaVersion = "0.19.0"

// phononMinOllamaVersion is the first Ollama release that runs Phonon speech
// models.
const phononMinOllamaVersion = "0.40.9"

// IsSafetensorsLLMModel checks for a safetensors completion or decision model.
func IsSafetensorsLLMModel(modelName string) bool {
	name := model.ParseName(modelName)
	if !name.IsValid() {
		return false
	}
	m, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return false
	}
	f, err := m.Config.Open()
	if err != nil {
		return false
	}
	defer f.Close()

	var config model.ConfigV2
	if err := json.NewDecoder(f).Decode(&config); err != nil {
		return false
	}
	return config.ModelFormat == "safetensors" && (slices.Contains(config.Capabilities, "completion") || slices.Contains(config.Capabilities, "decision"))
}

// IsSafetensorsModelDir checks for a supported model config and safetensors weights.
func IsSafetensorsModelDir(dir string) bool {
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		if !os.IsNotExist(err) {
			return false
		}
		if _, err := os.Stat(filepath.Join(dir, "rl_agent_config.json")); err != nil {
			return false
		}
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

// ManifestInfo is the data a manifest writer needs after the import pipeline
// has planned and written the source model.
type ManifestInfo struct {
	ModelConfig model.ConfigV2
	ConfigLayer LayerInfo
	Layers      []LayerInfo
	Class       Classification
}

// ManifestWriter writes the manifest file.
type ManifestWriter func(ctx context.Context, modelName string, info ManifestInfo) error

func shouldQuantize(name string) bool {
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
		strings.Contains(name, ".mlp.shared_experts.") ||
		strings.Contains(name, ".mixer.shared_experts.")
}

// isRoutingGate reports the small MoE routing/gate weights that select the
// active experts. Quantization noise there can flip expert selection, so they
// are kept at source precision regardless of architecture.
func isRoutingGate(name string) bool {
	return strings.HasSuffix(name, ".mlp.gate.weight") ||
		strings.HasSuffix(name, ".mixer.gate.weight") ||
		strings.HasSuffix(name, ".shared_expert_gate.weight") ||
		strings.HasSuffix(name, ".router.proj.weight")
}

// GetTensorQuantization returns the appropriate quantization type for a tensor.
// Returns "" if the tensor should not be quantized.
func GetTensorQuantization(name string, shape []int32, quantize string) string {
	stackedExpert := isStackedExpertWeight(name)

	// Use basic name-based check first
	if !stackedExpert && !shouldQuantize(name) {
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

	// lm_head is too sensitive for 4-bit types; the 8-bit type in the requested
	// family keeps quality close to bf16 while saving decode bandwidth.
	if strings.HasSuffix(name, "lm_head.weight") {
		if e := eightBit(quantNorm); isAligned(shape, e) {
			return e
		}
		return ""
	}

	// Vision components are too quantization-sensitive; keep source precision.
	if isVision(name) {
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

var expertLayerPrefixRegexp = regexp.MustCompile(`^(?:model\.language_model\.backbone\.|model\.language_model\.|language_model(?:\.model)?\.|language_model\.backbone\.|model\.|backbone\.|mtp\.)?layers\.\d+$`)

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
		".mixer.experts.",
		".mixer.shared_experts.",
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
	// Modules holds mlx-lm's per-module entries, keyed by module path, which
	// mixed-precision recipes write next to the defaults. Other producers'
	// objects can land here too; it is only meaningful when quantType names
	// an MLX type.
	Modules map[string]moduleQuantization `json:"-"`
}

// moduleQuantization is one mlx-lm per-module entry. mlx.nn passes it to
// to_quantized as keyword arguments, so a field it omits or nulls takes MLX's
// default for the mode, not the config's.
type moduleQuantization struct {
	Bits      int    `json:"bits"`
	GroupSize int    `json:"group_size"`
	Mode      string `json:"mode"`
	// Unreadable marks an entry whose fields aren't the integers and string
	// mlx.nn expects, so its settings are unknown.
	Unreadable bool `json:"-"`
}

func (q *sourceQuantization) UnmarshalJSON(data []byte) error {
	// Named so decode errors read "quantization.<field>" whichever block
	// holds the field. Decoding through this method drops the path above it,
	// such as text_config or compression_config.
	type quantization sourceQuantization
	if err := json.Unmarshal(data, (*quantization)(q)); err != nil {
		return err
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}
	for path, raw := range entries {
		// mlx.nn quantizes a module whose entry is a non-empty object. true takes
		// the config's defaults, and null, false and {} leave the module
		// unquantized, so none of those are entries.
		var fields map[string]json.RawMessage
		if json.Unmarshal(raw, &fields) != nil || len(fields) == 0 {
			continue
		}
		var m moduleQuantization
		if json.Unmarshal(raw, &m) != nil {
			m = moduleQuantization{Unreadable: true}
		}
		if q.Modules == nil {
			q.Modules = make(map[string]moduleQuantization)
		}
		q.Modules[path] = m
	}
	return nil
}

// mixedPrecision reports whether any per-module entry quantizes its module
// differently from the default. QuantMetadata calls it for every weight, so it
// stops at the first such entry.
func (q sourceQuantization) mixedPrecision() bool {
	for _, m := range q.Modules {
		if q.overrides(m) {
			return true
		}
	}
	return false
}

// firstMixedModule returns, in path order, the first module whose entry
// quantizes it differently from the default, or "" if none does.
func (q sourceQuantization) firstMixedModule() string {
	for _, path := range slices.Sorted(maps.Keys(q.Modules)) {
		if q.overrides(q.Modules[path]) {
			return path
		}
	}
	return ""
}

// overrides reports whether m quantizes its module differently from the
// default. An entry that restates the default field for field needs no
// resolving.
func (q sourceQuantization) overrides(m moduleQuantization) bool {
	defaults := q.defaults()
	if m == defaults {
		return false
	}
	quantType, groupSize := m.settings()
	wantType, wantGroupSize := defaults.settings()
	return quantType != wantType || groupSize != wantGroupSize
}

// defaults returns the entry that restates the config's default quantization,
// which MLX gives modules without an entry.
func (q sourceQuantization) defaults() moduleQuantization {
	return moduleQuantization{Bits: q.Bits, GroupSize: q.GroupSize, Mode: q.Mode}
}

// settings returns the quantization type and group size MLX gives the module,
// with "" for a type the engine does not recognize.
func (m moduleQuantization) settings() (quantType string, groupSize int) {
	if m.Unreadable {
		return "", 0
	}
	// MLX's affine default is 4 bits; the other modes fix their width.
	quantType = sourceQuantization{Bits: cmp.Or(m.Bits, 4), Mode: m.Mode}.quantType()
	defaultGroupSize, _, _ := quant.Params(quantType)
	return quantType, cmp.Or(m.GroupSize, defaultGroupSize)
}

type sourceModelConfig struct {
	ModelType     string          `json:"model_type"`
	Architectures []string        `json:"architectures"`
	VisionConfig  *map[string]any `json:"vision_config"`
	AudioConfig   *map[string]any `json:"audio_config"`
	HasVision     bool            `json:"has_vision"`
	SoundConfig   *map[string]any `json:"sound_config"`
	LLMConfig     struct {
		ModelType string `json:"model_type"`
	} `json:"llm_config"`
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
	if os.IsNotExist(err) {
		return readLayaConfig(modelDir)
	}
	if err != nil {
		return sourceModelConfig{}, nil, fmt.Errorf("read %s: %w", configPath, err)
	}

	var cfg sourceModelConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return sourceModelConfig{}, nil, fmt.Errorf("parse %s: %w", configPath, err)
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

// quantType returns the canonical quantization type the config declares, or
// "" if it declares none the engine recognizes.
func (q sourceQuantization) quantType() string {
	switch mode := strings.ToLower(q.Mode); mode {
	case "":
		// mlx-lm omits the mode for its default, affine. Other producers
		// identify themselves with a quant_method instead.
		if q.QuantMethod != "" {
			return ""
		}
		fallthrough
	case "affine":
		return quant.Canonical(fmt.Sprintf("int%d", q.Bits))
	case "nvfp4", "mxfp8", "mxfp4":
		return mode
	}
	return ""
}

// declaredQuantization returns the first quantization config that names a bit
// width.
func (cfg sourceModelConfig) declaredQuantization() sourceQuantization {
	for _, q := range cfg.quantizationConfigs() {
		if q.Bits != 0 {
			return q
		}
	}
	return sourceQuantization{}
}

func (cfg sourceModelConfig) QuantMetadata() map[string]string {
	q := cfg.declaredQuantization()
	quantType := q.quantType()
	if quantType == "" {
		return nil
	}
	// The loader trusts the recorded type over tensor shapes, so recording the
	// default would mislabel modules whose entries differ from it. Such a config
	// without a mode records nothing, and the loader infers each weight from its
	// shapes as it always has; checkModuleQuantization refuses the ones whose
	// layouts it can't read that way. Configs that state a mode still record the
	// default, pending support for per-module entries.
	if q.Mode == "" && q.mixedPrecision() {
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
	"Qwen3_5Model":                          newPplxImportTransform,
	"KevForDecision":                        newQwen35DecisionImportTransform,
	"StrandsDeciderForDecision":             newQwen35DecisionImportTransform,
	"ClefForDecision":                       newQwen35DecisionImportTransform,
	"Qwen3_5ForCausalLM":                    newQwen35ImportTransform,
	"Qwen3_5ForConditionalGeneration":       newQwen35ImportTransform,
	"Qwen3NextForCausalLM":                  newQwen35ImportTransform,
	"Qwen3NextForConditionalGeneration":     newQwen35ImportTransform,
	"Qwen3_5MoeForCausalLM":                 newQwen35ImportTransform,
	"Qwen3_5MoeForConditionalGeneration":    newQwen35ImportTransform,
	"Qwen3NextMoeForCausalLM":               newQwen35ImportTransform,
	"Qwen3NextMoeForConditionalGeneration":  newQwen35ImportTransform,
	"Qwen4ExpForConditionalGeneration":      newQwen4ExpImportTransform,
	"Gemma4ForCausalLM":                     newGemma4ImportTransform,
	"Gemma4ForConditionalGeneration":        newGemma4ImportTransform,
	"Gemma4UnifiedForCausalLM":              newGemma4ImportTransform,
	"Gemma4UnifiedForConditionalGeneration": newGemma4ImportTransform,
	"gemma4_unified":                        newGemma4ImportTransform,
	"gemma4_unified_text":                   newGemma4ImportTransform,
	"LagunaForCausalLM":                     newLagunaImportTransform,
	"MuseGlimmerForConditionalGeneration":   newGlimmerImportTransform,
	"Cohere2MoeForCausalLM":                 newCohere2MoeImportTransform,
	"Gemma4AssistantForCausalLM":            newGemma4ImportTransform,
	"Gemma4UnifiedAssistantForCausalLM":     newGemma4ImportTransform,
	"gemma4_unified_assistant":              newGemma4ImportTransform,
	"NemotronH_Nano_VL_V2":                  newNemotronHImportTransform,
	"NemotronH_Nano_Omni_Reasoning_V3":      newNemotronHImportTransform,
	"NemotronH_Omni_Reasoning_V3":           newNemotronHImportTransform,
	"NemotronHForCausalLM":                  newNemotronHImportTransform,
	"EmbeddingGemma2Model":                  newGemma4EmbeddingImportTransform,
	"Lfm3VlForConditionalGeneration":        newLFM3ImportTransform,
	"Lfm3OmniForConditionalGeneration":      newLFM3OmniImportTransform,
	"Lfm3AudioForConditionalGeneration":     newLFM3AudioImportTransform,
	"D1Model":                               newLFM3ImportTransform,
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
	validated, err := validateFloat32TensorData(td, name)
	if err != nil || validated == nil {
		if err != nil && td != nil && strings.EqualFold(td.Dtype, "F32") {
			return nil, fmt.Errorf("expected scalar F32 tensor, got shape %v", td.Shape)
		}
		return validated, err
	}
	n := int64(1)
	for _, dim := range validated.Shape {
		n *= int64(dim)
	}
	if n != 1 {
		return nil, fmt.Errorf("expected scalar F32 tensor, got shape %v", validated.Shape)
	}
	return validated, nil
}

func validateFloat32TensorData(td *safetensors.TensorData, name string) (*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	if strings.ToUpper(td.Dtype) != "F32" {
		return nil, fmt.Errorf("expected F32 tensor, got %s", td.Dtype)
	}
	n := int64(1)
	for _, dim := range td.Shape {
		if dim <= 0 || n > math.MaxInt64/int64(dim) {
			return nil, fmt.Errorf("expected non-empty F32 tensor, got shape %v", td.Shape)
		}
		n *= int64(dim)
	}
	if n > math.MaxInt64/4 || td.Size != n*4 {
		return nil, fmt.Errorf("expected non-empty F32 tensor, got shape %v", td.Shape)
	}
	return td.WithName(name), nil
}

func invertFloat32TensorData(td *safetensors.TensorData, name string) (*safetensors.TensorData, error) {
	td, err := validateFloat32TensorData(td, name)
	if err != nil || td == nil {
		return td, err
	}
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) != td.Size {
		return nil, fmt.Errorf("invalid F32 tensor byte length %d, expected %d", len(raw), td.Size)
	}
	out := make([]byte, len(raw))
	for i := 0; i < len(raw); i += 4 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[i : i+4]))
		inverse := 1 / v
		if !(v > 0) || math.IsInf(float64(v), 0) || math.IsInf(float64(inverse), 0) {
			return nil, fmt.Errorf("cannot invert F32 scale %v: expected a finite positive scale with a finite reciprocal", v)
		}
		binary.LittleEndian.PutUint32(out[i:i+4], math.Float32bits(inverse))
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
