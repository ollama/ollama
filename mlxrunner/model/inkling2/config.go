package inkling2

import "fmt"

// Bound allocations driven by checkpoint metadata before weights are loaded.
const maxLayerCount = 1024

type textConfig struct {
	NumLayers               int         `json:"num_hidden_layers"`
	LocalLayers             []int       `json:"local_layer_ids"`
	ResidualBlockSize       int         `json:"n_layers_per_attn_res_block"`
	NormalizeResidualValues bool        `json:"attn_res_normalize_values"`
	Loop                    *loopConfig `json:"loop_config"`

	HiddenSize    int `json:"hidden_size"`
	VocabSize     int `json:"vocab_size"`
	UnpaddedVocab int `json:"unpadded_vocab_size"`
	Context       int `json:"model_max_length"`
	HeadDim       int `json:"head_dim"`
	Heads         int `json:"num_attention_heads"`
	KVHeads       int `json:"num_key_value_heads"`
	LocalHeadDim  int `json:"swa_head_dim"`
	LocalHeads    int `json:"swa_num_attention_heads"`
	LocalKVHeads  int `json:"swa_num_key_value_heads"`
	RelativeDim   int `json:"d_rel"`
	RelativeRange int `json:"rel_extent"`
	Window        int `json:"sliding_window_size"`
	ConvTaps      int `json:"sconv_kernel_size"`
	Experts       int `json:"n_routed_experts"`
	SharedExperts int `json:"n_shared_experts"`
	TopK          int `json:"num_experts_per_tok"`
	ExpertWidth   int `json:"intermediate_size"`

	NormEps      float32 `json:"rms_norm_eps"`
	QKClip       float32 `json:"qk_l2_clip_threshold"`
	RelativeClip float32 `json:"rel_l2_clip_threshold"`
	LogFloor     float32 `json:"log_scaling_n_floor"`
	LogAlpha     float32 `json:"log_scaling_alpha"`
	RouteScale   float32 `json:"route_scale"`
	GateCap      float32 `json:"mlp_gate_softcap"`
	UpCap        float32 `json:"mlp_up_softcap"`
	LogitCap     float32 `json:"final_logit_softcapping"`
	GateBias     bool    `json:"use_gate_bias"`
	ValueNorm    bool    `json:"use_value_norm_no_gamma"`
	QueryBias    bool    `json:"q_bias"`
	OutputBias   bool    `json:"o_bias"`
}

func (c textConfig) validate() error {
	if c.HiddenSize <= 0 || c.UnpaddedVocab <= 0 || c.UnpaddedVocab > c.VocabSize || c.Context <= 0 {
		return fmt.Errorf("invalid Inkling2 hidden size, vocabulary or context")
	}
	if c.HeadDim <= 0 || c.KVHeads <= 0 || c.Heads <= 0 || c.Heads%c.KVHeads != 0 ||
		c.LocalHeadDim <= 0 || c.LocalKVHeads <= 0 || c.LocalHeads <= 0 || c.LocalHeads%c.LocalKVHeads != 0 {
		return fmt.Errorf("invalid Inkling2 attention dimensions")
	}
	if c.ConvTaps <= 0 || c.RelativeDim <= 0 || c.RelativeRange <= 0 || c.Window <= 0 || c.NormEps <= 0 || c.QKClip <= 0 || c.RelativeClip <= 0 {
		return fmt.Errorf("invalid Inkling2 convolution, attention or normalization settings")
	}
	if c.TopK <= 0 || c.TopK > c.Experts || c.ExpertWidth <= 0 || c.RouteScale <= 0 || c.GateCap < 0 || c.UpCap < 0 || c.LogitCap < 0 {
		return fmt.Errorf("invalid Inkling2 expert routing or softcap settings")
	}
	if c.SharedExperts != 0 || c.QueryBias || c.OutputBias || !c.ValueNorm {
		return fmt.Errorf("unsupported Inkling2 expert, attention bias or value normalization configuration")
	}
	return nil
}

type loopConfig struct {
	Variable bool `json:"is_varloop"`
	Start    int  `json:"core_start"`
	End      int  `json:"core_end"`
	Passes   int  `json:"num_passes"`
	GateRank int  `json:"gate_rank"`
}

// A layer visit owns its own attention and convolution cache slots. Several
// visits can use the same physical layer's weights.
type layerVisit struct {
	layer int
	pass  int
	local bool
}

// executionPlan orders visits as the decoder executes them. Indices in the
// plan also identify the target activations consumed by the draft model.
func (c textConfig) executionPlan() ([]layerVisit, error) {
	if c.NumLayers <= 0 || c.NumLayers > maxLayerCount || c.ResidualBlockSize <= 0 || c.ResidualBlockSize > c.NumLayers {
		return nil, fmt.Errorf("invalid Inkling2 decoder depth %d or residual block size %d", c.NumLayers, c.ResidualBlockSize)
	}
	end := c.NumLayers
	visits := c.NumLayers
	if loop := c.Loop; loop != nil {
		if loop.Variable {
			return nil, fmt.Errorf("variable Inkling2 loops are unsupported")
		}
		if loop.Start < 0 || loop.End > c.NumLayers || loop.Start >= loop.End || loop.Passes < 2 || loop.GateRank <= 0 {
			return nil, fmt.Errorf("invalid Inkling2 loop: layers [%d,%d), passes %d, gate rank %d", loop.Start, loop.End, loop.Passes, loop.GateRank)
		}
		if loop.Start%c.ResidualBlockSize != 0 || loop.End%c.ResidualBlockSize != 0 || !c.NormalizeResidualValues {
			return nil, fmt.Errorf("Inkling2 looping requires aligned residual blocks and normalized residual values")
		}
		width := loop.End - loop.Start
		if loop.Passes-1 > (maxLayerCount-c.NumLayers)/width {
			return nil, fmt.Errorf("Inkling2 loop with %d passes exceeds the %d-visit limit", loop.Passes, maxLayerCount)
		}
		visits += (loop.Passes - 1) * width
		end = loop.End
	}

	local := make([]bool, c.NumLayers)
	for _, layer := range c.LocalLayers {
		if layer < 0 || layer >= c.NumLayers || local[layer] {
			return nil, fmt.Errorf("invalid or duplicate Inkling2 local attention layer %d", layer)
		}
		local[layer] = true
	}

	plan := make([]layerVisit, 0, visits)
	for layer := range end {
		plan = append(plan, layerVisit{layer: layer, local: local[layer]})
	}
	if loop := c.Loop; loop != nil {
		for pass := 1; pass < loop.Passes; pass++ {
			for layer := loop.Start; layer < loop.End; layer++ {
				plan = append(plan, layerVisit{layer: layer, pass: pass, local: local[layer]})
			}
		}
	}
	for layer := end; layer < c.NumLayers; layer++ {
		plan = append(plan, layerVisit{layer: layer, local: local[layer]})
	}
	return plan, nil
}
