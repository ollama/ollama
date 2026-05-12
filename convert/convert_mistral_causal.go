package convert

import (
	"cmp"
	"fmt"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type mistral3CausalModel struct {
	ModelParameters

	NumHiddenLayers       uint32  `json:"num_hidden_layers"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RopeTheta             float32 `json:"rope_theta"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`
	HeadDim               uint32  `json:"head_dim"`
	SlidingWindow         *uint32 `json:"sliding_window"`
	HiddenAct             string  `json:"hidden_act"`
	VocabSize             uint32  `json:"vocab_size"`
	RopeParameters        struct {
		BetaFast                  float32  `json:"beta_fast"`
		BetaSlow                  float32  `json:"beta_slow"`
		Factor                    float32  `json:"factor"`
		Llama4ScalingBeta         *float32 `json:"llama_4_scaling_beta"`
		OrigMaxPositionEmbeddings uint32   `json:"original_max_position_embeddings"`
		RopeType                  string   `json:"rope_type"`
		RopeTheta                 float32  `json:"rope_theta"`
		Mscale                    *float32 `json:"mscale"`
		MscaleAllDim              *float32 `json:"mscale_all_dim"`
	} `json:"rope_parameters"`
}

func (p *mistral3CausalModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "mistral3"
	kv["mistral3.vocab_size"] = p.VocabSize

	// Text configuration
	kv["mistral3.block_count"] = p.NumHiddenLayers
	kv["mistral3.context_length"] = p.MaxPositionEmbeddings
	kv["mistral3.embedding_length"] = p.HiddenSize
	kv["mistral3.feed_forward_length"] = p.IntermediateSize
	kv["mistral3.attention.head_count"] = p.NumAttentionHeads
	kv["mistral3.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["mistral3.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["mistral3.attention.key_length"] = p.HeadDim
	kv["mistral3.attention.value_length"] = p.HeadDim
	kv["mistral3.rope.dimension_count"] = cmp.Or(p.HeadDim, p.HiddenSize/p.NumAttentionHeads)
	kv["mistral3.rope.freq_base"] = cmp.Or(p.RopeTheta, p.RopeParameters.RopeTheta)
	kv["mistral3.rope.scaling.factor"] = p.RopeParameters.Factor
	kv["mistral3.rope.scaling.type"] = p.RopeParameters.RopeType
	kv["mistral3.rope.scaling.beta_fast"] = p.RopeParameters.BetaFast
	kv["mistral3.rope.scaling.beta_slow"] = p.RopeParameters.BetaSlow

	if p.RopeParameters.Mscale != nil {
		kv["mistral3.rope.scaling.mscale"] = *p.RopeParameters.Mscale
	}

	if p.RopeParameters.MscaleAllDim != nil {
		kv["mistral3.rope.scaling.mscale_all_dim"] = *p.RopeParameters.MscaleAllDim
	}

	if p.RopeParameters.OrigMaxPositionEmbeddings > 0 {
		kv["mistral3.rope.scaling.original_context_length"] = p.RopeParameters.OrigMaxPositionEmbeddings
		kv["mistral3.rope.scaling_beta"] = *p.RopeParameters.Llama4ScalingBeta
	}

	if p.RopeParameters.Llama4ScalingBeta != nil {
		kv["mistral3.rope.scaling_beta"] = *p.RopeParameters.Llama4ScalingBeta
	}

	return kv
}

func (p *mistral3CausalModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		if !strings.HasPrefix(t.Name(), "v.") {
			if strings.HasSuffix(t.Name(), ".attn_q.weight") ||
				strings.HasSuffix(t.Name(), ".attn_k.weight") {
				t.SetRepacker(p.repack)
			}
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

func (p *mistral3CausalModel) Replacements() []string {
	return []string{
		"model.norm", "output_norm",
		"model.", "",
		"layers", "blk",
		"transformer.layers", "blk",
		"vision_tower", "v",
		"ln_pre", "encoder_norm",
		"input_layernorm", "attn_norm",
		"post_attention_layernorm", "ffn_norm",
		"embed_tokens", "token_embd",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"attention.q_proj", "attn_q",
		"attention.k_proj", "attn_k",
		"attention.v_proj", "attn_v",
		"attention.o_proj", "attn_output",
		"attention_norm", "attn_norm",
		"feed_forward.gate_proj", "ffn_gate",
		"feed_forward.down_proj", "ffn_down",
		"feed_forward.up_proj", "ffn_up",
		"multi_modal_projector", "mm",
		"ffn_norm", "ffn_norm",
		"lm_head", "output",
	}
}

func (p *mistral3CausalModel) repack(name string, data []float32, shape []uint64) ([]float32, error) {
	var dims []int
	for _, dim := range shape {
		dims = append(dims, int(dim))
	}

	var heads uint32
	if strings.HasSuffix(name, ".attn_q.weight") {
		heads = p.NumAttentionHeads
	} else if strings.HasSuffix(name, ".attn_k.weight") {
		heads = cmp.Or(p.NumKeyValueHeads, p.NumAttentionHeads)
	} else {
		return nil, fmt.Errorf("unknown tensor for repack: %s", name)
	}

	n := tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
	if err := n.Reshape(append([]int{int(heads), 2, dims[0] / int(heads) / 2}, dims[1:]...)...); err != nil {
		return nil, err
	}

	if err := n.T(0, 2, 1, 3); err != nil {
		return nil, err
	}

	if err := n.Reshape(dims...); err != nil {
		return nil, err
	}

	if err := n.Transpose(); err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 1)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}
