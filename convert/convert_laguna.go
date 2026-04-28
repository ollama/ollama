package convert

import (
	"cmp"
	"encoding/json"
	"fmt"
	iofs "io/fs"
	"math"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type lagunaModel struct {
	ModelParameters

	NumHiddenLayers       uint32           `json:"num_hidden_layers"`
	HiddenSize            uint32           `json:"hidden_size"`
	IntermediateSize      uint32           `json:"intermediate_size"`
	NumAttentionHeads     uint32           `json:"num_attention_heads"`
	NumKeyValueHeads      uint32           `json:"num_key_value_heads"`
	HeadDim               uint32           `json:"head_dim"`
	RMSNormEPS            float32          `json:"rms_norm_eps"`
	MaxPositionEmbeddings uint32           `json:"max_position_embeddings"`
	SlidingWindow         uint32           `json:"sliding_window"`
	PartialRotaryFactor   float32          `json:"partial_rotary_factor"`
	Gating                lagunaGatingMode `json:"gating"`
	QKNormType            string           `json:"qk_norm_type"`

	LayerTypes                []string `json:"layer_types"`
	NumAttentionHeadsPerLayer []uint32 `json:"num_attention_heads_per_layer"`

	NumExperts                   uint32   `json:"num_experts"`
	NumExpertsPerTok             uint32   `json:"num_experts_per_tok"`
	MoEIntermediateSize          uint32   `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize uint32   `json:"shared_expert_intermediate_size"`
	NormTopKProb                 bool     `json:"norm_topk_prob"`
	MoeRoutedScalingFactor       float32  `json:"moe_routed_scaling_factor"`
	MoERouterUseSigmoid          bool     `json:"moe_router_use_sigmoid"`
	MoEApplyRouterWeightOnInput  bool     `json:"moe_apply_router_weight_on_input"`
	DecoderSparseStep            uint32   `json:"decoder_sparse_step"`
	MLPOnlyLayers                []uint32 `json:"mlp_only_layers"`
	MLPLayerTypes                []string `json:"mlp_layer_types"`

	RopeParameters    lagunaRopeParameters `json:"rope_parameters"`
	SwaRopeParameters lagunaRopeParameters `json:"swa_rope_parameters"`

	SwaAttentionSinkEnabled bool `json:"swa_attention_sink_enabled"`
}

type lagunaGatingMode string

type lagunaRopeParameters struct {
	RopeTheta                     float32 `json:"rope_theta"`
	RopeType                      string  `json:"rope_type"`
	Type                          string  `json:"type"`
	Factor                        float32 `json:"factor"`
	OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`
	BetaSlow                      float32 `json:"beta_slow"`
	BetaFast                      float32 `json:"beta_fast"`
	AttentionFactor               float32 `json:"attention_factor"`
	PartialRotaryFactor           float32 `json:"partial_rotary_factor"`
}

type lagunaRopeConfig struct {
	flat    lagunaRopeParameters
	full    lagunaRopeParameters
	sliding lagunaRopeParameters
	nested  bool
}

func (g *lagunaGatingMode) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		*g = lagunaGatingMode(s)
		return nil
	}

	var enabled bool
	if err := json.Unmarshal(b, &enabled); err == nil {
		if enabled {
			*g = "true"
		} else {
			*g = "false"
		}
		return nil
	}

	if string(b) == "null" {
		return nil
	}
	return fmt.Errorf("unsupported Laguna gating JSON value %s", string(b))
}

func (g lagunaGatingMode) perHead() bool {
	return strings.EqualFold(string(g), "per-head") || strings.EqualFold(string(g), "true")
}

func (r *lagunaRopeConfig) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		return nil
	}

	var probe map[string]json.RawMessage
	if err := json.Unmarshal(b, &probe); err != nil {
		return err
	}

	if len(probe) == 0 {
		return nil
	}

	if raw, ok := probe["full_attention"]; ok {
		r.nested = true
		if err := json.Unmarshal(raw, &r.full); err != nil {
			return err
		}
		if raw = probe["sliding_attention"]; raw != nil {
			if err := json.Unmarshal(raw, &r.sliding); err != nil {
				return err
			}
		}
		return nil
	}

	if raw, ok := probe["global_attention"]; ok {
		r.nested = true
		if err := json.Unmarshal(raw, &r.full); err != nil {
			return err
		}
		if raw = probe["sliding_attention"]; raw != nil {
			if err := json.Unmarshal(raw, &r.sliding); err != nil {
				return err
			}
		}
		return nil
	}

	return json.Unmarshal(b, &r.flat)
}

func (r lagunaRopeConfig) fullParams() lagunaRopeParameters {
	if r.nested {
		return r.full
	}
	return r.flat
}

func (r lagunaRopeConfig) slidingParams() (lagunaRopeParameters, bool) {
	if !r.nested {
		return lagunaRopeParameters{}, false
	}
	return r.sliding, true
}

func (r lagunaRopeParameters) ropeType() string {
	return cmp.Or(r.RopeType, r.Type)
}

func (r lagunaRopeParameters) withDefaultPartialRotaryFactor(v float32) lagunaRopeParameters {
	if r.PartialRotaryFactor == 0 {
		r.PartialRotaryFactor = v
	}
	return r
}

func (r lagunaRopeParameters) empty() bool {
	return r == (lagunaRopeParameters{})
}

type rawLagunaModel struct {
	ModelParameters

	NumHiddenLayers       uint32           `json:"num_hidden_layers"`
	HiddenSize            uint32           `json:"hidden_size"`
	IntermediateSize      uint32           `json:"intermediate_size"`
	NumAttentionHeads     uint32           `json:"num_attention_heads"`
	NumKeyValueHeads      uint32           `json:"num_key_value_heads"`
	HeadDim               uint32           `json:"head_dim"`
	RMSNormEPS            float32          `json:"rms_norm_eps"`
	MaxPositionEmbeddings uint32           `json:"max_position_embeddings"`
	SlidingWindow         uint32           `json:"sliding_window"`
	PartialRotaryFactor   float32          `json:"partial_rotary_factor"`
	Gating                lagunaGatingMode `json:"gating"`
	QKNormType            string           `json:"qk_norm_type"`

	LayerTypes                []string `json:"layer_types"`
	NumAttentionHeadsPerLayer []uint32 `json:"num_attention_heads_per_layer"`

	NumExperts                   uint32   `json:"num_experts"`
	NumExpertsPerTok             uint32   `json:"num_experts_per_tok"`
	MoEIntermediateSize          uint32   `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize uint32   `json:"shared_expert_intermediate_size"`
	NormTopKProb                 *bool    `json:"norm_topk_prob"`
	MoeRoutedScalingFactor       float32  `json:"moe_routed_scaling_factor"`
	MoERouterUseSigmoid          *bool    `json:"moe_router_use_sigmoid"`
	MoEApplyRouterWeightOnInput  bool     `json:"moe_apply_router_weight_on_input"`
	DecoderSparseStep            uint32   `json:"decoder_sparse_step"`
	MLPOnlyLayers                []uint32 `json:"mlp_only_layers"`
	MLPLayerTypes                []string `json:"mlp_layer_types"`

	RopeParameters    lagunaRopeConfig     `json:"rope_parameters"`
	SwaRopeParameters lagunaRopeParameters `json:"swa_rope_parameters"`

	SwaAttentionSinkEnabled bool `json:"swa_attention_sink_enabled"`
}

func (p *lagunaModel) UnmarshalJSON(b []byte) error {
	var raw rawLagunaModel
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}

	mlpOnlyLayers, err := lagunaDenseLayers(raw.MLPOnlyLayers, raw.MLPLayerTypes)
	if err != nil {
		return err
	}

	fullRope := raw.RopeParameters.fullParams().withDefaultPartialRotaryFactor(cmp.Or(raw.PartialRotaryFactor, float32(1)))
	swaRope := raw.SwaRopeParameters
	if nestedSwa, ok := raw.RopeParameters.slidingParams(); ok && !nestedSwa.empty() {
		swaRope = nestedSwa
	}
	swaRope = swaRope.withDefaultPartialRotaryFactor(cmp.Or(fullRope.PartialRotaryFactor, float32(1)))

	*p = lagunaModel{
		ModelParameters:              raw.ModelParameters,
		NumHiddenLayers:              raw.NumHiddenLayers,
		HiddenSize:                   raw.HiddenSize,
		IntermediateSize:             raw.IntermediateSize,
		NumAttentionHeads:            raw.NumAttentionHeads,
		NumKeyValueHeads:             raw.NumKeyValueHeads,
		HeadDim:                      raw.HeadDim,
		RMSNormEPS:                   raw.RMSNormEPS,
		MaxPositionEmbeddings:        raw.MaxPositionEmbeddings,
		SlidingWindow:                raw.SlidingWindow,
		PartialRotaryFactor:          cmp.Or(raw.PartialRotaryFactor, fullRope.PartialRotaryFactor),
		Gating:                       raw.Gating,
		QKNormType:                   cmp.Or(raw.QKNormType, "rmsnorm"),
		LayerTypes:                   raw.LayerTypes,
		NumAttentionHeadsPerLayer:    raw.NumAttentionHeadsPerLayer,
		NumExperts:                   raw.NumExperts,
		NumExpertsPerTok:             raw.NumExpertsPerTok,
		MoEIntermediateSize:          raw.MoEIntermediateSize,
		SharedExpertIntermediateSize: raw.SharedExpertIntermediateSize,
		NormTopKProb:                 defaultBool(raw.NormTopKProb, true),
		MoeRoutedScalingFactor:       raw.MoeRoutedScalingFactor,
		MoERouterUseSigmoid:          defaultBool(raw.MoERouterUseSigmoid, true),
		MoEApplyRouterWeightOnInput:  raw.MoEApplyRouterWeightOnInput,
		DecoderSparseStep:            raw.DecoderSparseStep,
		MLPOnlyLayers:                mlpOnlyLayers,
		MLPLayerTypes:                raw.MLPLayerTypes,
		RopeParameters:               fullRope,
		SwaRopeParameters:            swaRope,
		SwaAttentionSinkEnabled:      raw.SwaAttentionSinkEnabled,
	}
	return nil
}

func defaultBool(v *bool, fallback bool) bool {
	if v == nil {
		return fallback
	}
	return *v
}

const (
	lagunaGatingFuncSoftmax uint32 = 1
	lagunaGatingFuncSigmoid uint32 = 2

	lagunaLayerTypeGlobal  uint32 = 0
	lagunaLayerTypeSliding uint32 = 1
)

func (p *lagunaModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "laguna"
	// Laguna's chat template and built-in renderer both emit the leading
	// special token explicitly. Auto-prepending BOS here would duplicate it.
	kv["tokenizer.ggml.add_bos_token"] = false
	kv["tokenizer.ggml.pre"] = "laguna"
	// Laguna does not need tokenizer.chat_template at runtime: Ollama create
	// sets the Laguna renderer/parser from the architecture, and the renderer
	// owns prompt formatting.
	delete(kv, "tokenizer.chat_template")

	kv["laguna.block_count"] = p.NumHiddenLayers
	kv["laguna.context_length"] = p.MaxPositionEmbeddings
	kv["laguna.embedding_length"] = p.HiddenSize
	kv["laguna.feed_forward_length"] = p.IntermediateSize

	if len(p.NumAttentionHeadsPerLayer) == int(p.NumHiddenLayers) {
		kv["laguna.attention.head_count"] = p.NumAttentionHeadsPerLayer
	} else {
		kv["laguna.attention.head_count"] = p.NumAttentionHeads
	}
	kv["laguna.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["laguna.attention.key_length"] = p.HeadDim
	kv["laguna.attention.value_length"] = p.HeadDim
	kv["laguna.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["laguna.attention.sliding_window"] = p.SlidingWindow
	kv["laguna.attention.sink_enabled"] = p.SwaAttentionSinkEnabled

	if len(p.LayerTypes) > 0 {
		encoded := make([]uint32, len(p.LayerTypes))
		slidingPattern := make([]bool, len(p.LayerTypes))
		for i, layerType := range p.LayerTypes {
			if lagunaLayerIsSliding(layerType) {
				encoded[i] = lagunaLayerTypeSliding
				slidingPattern[i] = true
			} else {
				encoded[i] = lagunaLayerTypeGlobal
			}
		}
		kv["laguna.attention.layer_types"] = encoded
		kv["laguna.attention.sliding_window_pattern"] = slidingPattern
	}

	if p.Gating.perHead() {
		kv["laguna.attention.gating_type"] = uint32(1)
	} else {
		kv["laguna.attention.gating_type"] = uint32(0)
	}
	kv["laguna.attention.qk_norm"] = p.QKNormType == "rmsnorm"

	kv["laguna.expert_count"] = p.NumExperts
	kv["laguna.expert_used_count"] = p.NumExpertsPerTok
	kv["laguna.expert_feed_forward_length"] = p.MoEIntermediateSize
	kv["laguna.expert_shared_feed_forward_length"] = p.SharedExpertIntermediateSize
	kv["laguna.expert_shared_count"] = uint32(1)
	kv["laguna.expert_weights_norm"] = p.NormTopKProb
	kv["laguna.expert_weights_scale"] = p.MoeRoutedScalingFactor
	kv["laguna.expert_gating_func"] = lagunaMoeGatingFunc(p.MoERouterUseSigmoid)
	kv["laguna.decoder_sparse_step"] = cmp.Or(p.DecoderSparseStep, uint32(1))

	if leading, ok := lagunaLeadingDensePrefix(p.MLPOnlyLayers); ok {
		kv["laguna.leading_dense_block_count"] = leading
	}
	if len(p.MLPOnlyLayers) > 0 {
		kv["laguna.dense_layers"] = p.MLPOnlyLayers
	}

	ropeType := p.RopeParameters.ropeType()
	kv["laguna.rope.freq_base"] = cmp.Or(p.RopeParameters.RopeTheta, float32(10000))
	kv["laguna.rope.scaling.type"] = ropeType
	ropeFactor := cmp.Or(p.RopeParameters.Factor, float32(1))
	kv["laguna.rope.scaling.factor"] = ropeFactor
	kv["laguna.rope.scaling.original_context_length"] = p.RopeParameters.OriginalMaxPositionEmbeddings
	kv["laguna.rope.scaling.beta_fast"] = p.RopeParameters.BetaFast
	kv["laguna.rope.scaling.beta_slow"] = p.RopeParameters.BetaSlow
	kv["laguna.rope.scaling.attn_factor"] = lagunaAttentionFactor(ropeType, ropeFactor, p.RopeParameters.AttentionFactor)
	kv["laguna.rope.partial_rotary_factor"] = cmp.Or(p.PartialRotaryFactor, float32(1))

	swaRopeType := p.SwaRopeParameters.ropeType()
	kv["laguna.rope.swa.freq_base"] = cmp.Or(p.SwaRopeParameters.RopeTheta, float32(10000))
	kv["laguna.rope.swa.scaling.type"] = cmp.Or(swaRopeType, "linear")
	kv["laguna.rope.swa.scaling.factor"] = cmp.Or(p.SwaRopeParameters.Factor, float32(1))
	kv["laguna.rope.swa.partial_rotary_factor"] = cmp.Or(p.SwaRopeParameters.PartialRotaryFactor, float32(1))

	headDim := p.HeadDim
	if headDim == 0 && p.NumAttentionHeads > 0 {
		headDim = p.HiddenSize / p.NumAttentionHeads
	}
	kv["laguna.rope.dimension_count"] = lagunaRopeDim(headDim, cmp.Or(p.PartialRotaryFactor, float32(1)))
	kv["laguna.rope.swa.dimension_count"] = lagunaRopeDim(headDim, cmp.Or(p.SwaRopeParameters.PartialRotaryFactor, float32(1)))

	return kv
}

func (p *lagunaModel) parseMore(_ iofs.FS) error {
	return p.validate()
}

func (p *lagunaModel) validate() error {
	if p.NumHiddenLayers == 0 {
		return fmt.Errorf("laguna: num_hidden_layers must be set")
	}
	if p.HiddenSize == 0 {
		return fmt.Errorf("laguna: hidden_size must be set")
	}
	if p.HeadDim == 0 {
		return fmt.Errorf("laguna: head_dim must be set")
	}
	if p.NumKeyValueHeads == 0 {
		return fmt.Errorf("laguna: num_key_value_heads must be set")
	}
	if p.SwaAttentionSinkEnabled {
		return fmt.Errorf("laguna: unsupported swa_attention_sink_enabled=true")
	}
	if !p.Gating.perHead() {
		return fmt.Errorf("laguna: unsupported attention gating %q: only gating=\"per-head\" is supported", p.Gating)
	}
	if p.QKNormType != "rmsnorm" {
		return fmt.Errorf("laguna: unsupported qk_norm_type %q: only rmsnorm is supported", p.QKNormType)
	}
	if !p.MoERouterUseSigmoid {
		return fmt.Errorf("laguna: unsupported moe_router_use_sigmoid=false")
	}
	if p.MoEApplyRouterWeightOnInput {
		return fmt.Errorf("laguna: unsupported moe_apply_router_weight_on_input=true")
	}
	if p.DecoderSparseStep != 0 && p.DecoderSparseStep != 1 {
		return fmt.Errorf("laguna: unsupported decoder_sparse_step=%d: only 1 is supported", p.DecoderSparseStep)
	}
	if len(p.MLPOnlyLayers) != 1 || p.MLPOnlyLayers[0] != 0 {
		return fmt.Errorf("laguna: unsupported mlp_only_layers=%v: only [0] is supported", p.MLPOnlyLayers)
	}
	if p.NumExperts == 0 {
		return fmt.Errorf("laguna: num_experts must be set")
	}
	if p.NumExpertsPerTok == 0 {
		return fmt.Errorf("laguna: num_experts_per_tok must be set")
	}
	if p.MoEIntermediateSize == 0 {
		return fmt.Errorf("laguna: moe_intermediate_size must be set")
	}
	if p.SharedExpertIntermediateSize == 0 {
		return fmt.Errorf("laguna: shared_expert_intermediate_size must be set")
	}

	if len(p.LayerTypes) > 0 && len(p.LayerTypes) != int(p.NumHiddenLayers) {
		return fmt.Errorf("laguna: layer_types has %d entries, expected %d", len(p.LayerTypes), p.NumHiddenLayers)
	}
	for i, layerType := range p.LayerTypes {
		if !lagunaLayerIsGlobal(layerType) && !lagunaLayerIsSliding(layerType) {
			return fmt.Errorf("laguna: unsupported layer_types[%d]=%q", i, layerType)
		}
	}
	if len(p.NumAttentionHeadsPerLayer) > 0 && len(p.NumAttentionHeadsPerLayer) != int(p.NumHiddenLayers) {
		return fmt.Errorf("laguna: num_attention_heads_per_layer has %d entries, expected %d", len(p.NumAttentionHeadsPerLayer), p.NumHiddenLayers)
	}
	if len(p.NumAttentionHeadsPerLayer) == 0 && p.NumAttentionHeads == 0 {
		return fmt.Errorf("laguna: num_attention_heads or num_attention_heads_per_layer must be set")
	}
	for i, heads := range p.NumAttentionHeadsPerLayer {
		if heads == 0 {
			return fmt.Errorf("laguna: num_attention_heads_per_layer[%d] must be non-zero", i)
		}
	}

	return nil
}

func (p *lagunaModel) numHeadsForLayer(layer uint32) uint32 {
	if len(p.NumAttentionHeadsPerLayer) > int(layer) && p.NumAttentionHeadsPerLayer[layer] > 0 {
		return p.NumAttentionHeadsPerLayer[layer]
	}
	return p.NumAttentionHeads
}

func (p *lagunaModel) layerUsesMoE(layer uint32) bool {
	for _, denseLayer := range p.MLPOnlyLayers {
		if denseLayer == layer {
			return false
		}
	}
	step := cmp.Or(p.DecoderSparseStep, uint32(1))
	return p.NumExperts > 0 && (layer+1)%step == 0
}

func (p *lagunaModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"post_attention_layernorm", "ffn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"self_attn.g_proj", "attn_g",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_norm", "attn_k_norm",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",
		"mlp.gate.weight", "ffn_gate_inp.weight",
		"mlp.experts.e_score_correction_bias", "exp_probs_b.bias",
		"mlp.shared_expert.gate_proj", "ffn_gate_shexp",
		"mlp.shared_expert.up_proj", "ffn_up_shexp",
		"mlp.shared_expert.down_proj", "ffn_down_shexp",
		"mlp.experts.*.gate_proj", "ffn_gate_exps",
		"mlp.experts.*.up_proj", "ffn_up_exps",
		"mlp.experts.*.down_proj", "ffn_down_exps",
	}
}

func (p *lagunaModel) Tensors(ts []Tensor) []*ggml.Tensor {
	// Current Laguna drops store routed MoE experts as separate per-expert
	// tensors. GGUF stores each projection as one stacked tensor. If future
	// drops change expert naming or layout, update these patterns with a
	// focused conversion test using the new tensor names.
	merges := make([]merge, 0, p.NumHiddenLayers*3)
	for i := range p.NumHiddenLayers {
		merges = append(merges,
			merge{
				fmt.Sprintf("blk.%d.mlp.experts.*.gate_proj.weight", i),
				fmt.Sprintf("blk.%d.ffn_gate_exps.weight", i),
			},
			merge{
				fmt.Sprintf("blk.%d.mlp.experts.*.up_proj.weight", i),
				fmt.Sprintf("blk.%d.ffn_up_exps.weight", i),
			},
			merge{
				fmt.Sprintf("blk.%d.mlp.experts.*.down_proj.weight", i),
				fmt.Sprintf("blk.%d.ffn_down_exps.weight", i),
			},
		)
	}

	out, rest := mergeTensors(ts, merges...)
	for _, t := range rest {
		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}
	return out
}

func (p *lagunaModel) specialTokenTypes() []string {
	return []string{"bos", "eos", "pad", "unk"}
}

func lagunaLayerIsSliding(layerType string) bool {
	return strings.EqualFold(layerType, "sliding_attention")
}

func lagunaLayerIsGlobal(layerType string) bool {
	return strings.EqualFold(layerType, "full_attention") || strings.EqualFold(layerType, "global_attention")
}

func lagunaLeadingDensePrefix(layers []uint32) (uint32, bool) {
	for i, v := range layers {
		if v != uint32(i) {
			return 0, false
		}
	}
	return uint32(len(layers)), true
}

func lagunaDenseLayers(mlpOnlyLayers []uint32, mlpLayerTypes []string) ([]uint32, error) {
	if len(mlpOnlyLayers) > 0 {
		return mlpOnlyLayers, nil
	}
	if len(mlpLayerTypes) == 0 {
		return nil, nil
	}

	denseLayers := make([]uint32, 0, len(mlpLayerTypes))
	for i, layerType := range mlpLayerTypes {
		switch {
		case strings.EqualFold(layerType, "dense"):
			denseLayers = append(denseLayers, uint32(i))
		case strings.EqualFold(layerType, "sparse"):
		default:
			return nil, fmt.Errorf("laguna: unsupported mlp_layer_types[%d]=%q", i, layerType)
		}
	}
	return denseLayers, nil
}

func lagunaMoeGatingFunc(useSigmoid bool) uint32 {
	if useSigmoid {
		return lagunaGatingFuncSigmoid
	}
	return lagunaGatingFuncSoftmax
}

func lagunaAttentionFactor(ropeType string, scaleFactor, attentionFactor float32) float32 {
	if attentionFactor != 0 {
		return attentionFactor
	}
	if strings.EqualFold(ropeType, "yarn") && scaleFactor > 1 {
		return float32(0.1*math.Log(float64(scaleFactor)) + 1)
	}
	return 1
}

func lagunaRopeDim(headDim uint32, partialRotaryFactor float32) uint32 {
	if headDim == 0 {
		return 0
	}
	dim := uint32(float32(headDim) * partialRotaryFactor)
	if dim == 0 || dim > headDim {
		dim = headDim
	}
	if dim%2 != 0 {
		dim--
	}
	if dim == 0 {
		return headDim
	}
	return dim
}

var (
	_ ModelConverter = (*lagunaModel)(nil)
	_ moreParser     = (*lagunaModel)(nil)
)
