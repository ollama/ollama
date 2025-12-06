package convert

import (
	"cmp"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type rnj1Model struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`
	HeadDim               uint32  `json:"head_dim"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeScaling           *struct {
		Type                          string  `json:"rope_type"`
		Factor                        float32 `json:"factor"`
		OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`
		ExtrapolationFactor           float32 `json:"extrapolation_factor"`
		BetaFast                      float32 `json:"beta_fast"`
		BetaSlow                      float32 `json:"beta_slow"`
	} `json:"rope_scaling"`
}

var _ ModelConverter = (*rnj1Model)(nil)

func (p *rnj1Model) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "rnj1"

	kv["rnj1.context_length"] = cmp.Or(p.MaxPositionEmbeddings, uint32(32768))
	kv["rnj1.embedding_length"] = p.HiddenSize
	kv["rnj1.block_count"] = p.HiddenLayers
	kv["rnj1.feed_forward_length"] = p.IntermediateSize
	kv["rnj1.attention.head_count"] = cmp.Or(p.NumAttentionHeads, uint32(8))
	kv["rnj1.attention.head_count_kv"] = cmp.Or(p.NumKeyValueHeads, uint32(4))
	kv["rnj1.attention.layer_norm_rms_epsilon"] = cmp.Or(p.RMSNormEPS, float32(1e-6))
	kv["rnj1.attention.key_length"] = cmp.Or(p.HeadDim, uint32(128))
	kv["rnj1.attention.value_length"] = cmp.Or(p.HeadDim, uint32(128))
	kv["rnj1.rope.freq_base"] = cmp.Or(p.RopeTheta, float32(10000.0))

	// Handle YARN rope scaling if present
	if p.RopeScaling != nil && p.RopeScaling.Type == "yarn" && p.RopeScaling.Factor > 0 {
		kv["rnj1.rope.scaling.type"] = "yarn"
		kv["rnj1.rope.scaling.factor"] = p.RopeScaling.Factor
		kv["rnj1.rope.scaling.original_context_length"] = p.RopeScaling.OriginalMaxPositionEmbeddings
		kv["rnj1.rope.scaling.extrapolation_factor"] = p.RopeScaling.ExtrapolationFactor
		kv["rnj1.rope.scaling.beta_fast"] = p.RopeScaling.BetaFast
		kv["rnj1.rope.scaling.beta_slow"] = p.RopeScaling.BetaSlow
	}

	return kv
}

func (p *rnj1Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		// RNJ1 uses Gemma3RMSNorm which adds 1.0 to the norm weight
		if strings.HasSuffix(t.Name(), "_norm.weight") {
			t.SetRepacker(p.addOne)
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *rnj1Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"language_model.", "",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "post_attention_norm",
		"pre_feedforward_layernorm", "ffn_norm",
		"post_feedforward_layernorm", "post_ffw_norm",
	}
}

func (*rnj1Model) addOne(_ string, data []float32, shape []uint64) ([]float32, error) {
	n := tensor.New(tensor.WithShape(int(shape[0])), tensor.WithBacking(data))
	ones := tensor.Ones(tensor.Float32, int(shape[0]))

	n, err := n.Add(ones)
	if err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 0)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}
