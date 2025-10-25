package convert

import (
	"strings"
	"sync"

	"github.com/ollama/ollama/fs/ggml"
)

type minicpm3Model struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	QLoraRank             uint32  `json:"q_lora_rank"`
	KVLoraRank            uint32  `json:"kv_lora_rank"`
	QKNopeHeadDim         uint32  `json:"qk_nope_head_dim"`
	QKRopeHeadDim         uint32  `json:"qk_rope_head_dim"`
	RopeScaling struct {
		Type        string     `json:"type"`
		Factor      float32    `json:"factor"`
		LongFactor  ropeFactor `json:"long_factor"`
		ShortFactor ropeFactor `json:"short_factor"`
	} `json:"rope_scaling"`
}

var _ ModelConverter = (*minicpm3Model)(nil)

func (m *minicpm3Model) KV(t *Tokenizer) ggml.KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "minicpm3"
	kv["minicpm3.block_count"] = m.HiddenLayers
	kv["minicpm3.context_length"] = m.MaxPositionEmbeddings
	kv["minicpm3.embedding_length"] = m.HiddenSize
	kv["minicpm3.feed_forward_length"] = m.IntermediateSize
	kv["minicpm3.attention.head_count"] = m.NumAttentionHeads
	kv["minicpm3.attention.head_count_kv"] = m.NumKeyValueHeads
	kv["minicpm3.attention.layer_norm_rms_epsilon"] = m.RMSNormEPS
	if m.RopeTheta > 0 {
		kv["minicpm3.rope.freq_base"] = m.RopeTheta
	} else {
		kv["minicpm3.rope.freq_base"] = float32(1000000)
	}

	// LoRA parameters for MiniCPM3's attention mechanism
	kv["minicpm3.attention.q_lora_rank"] = m.QLoraRank
	kv["minicpm3.attention.kv_lora_rank"] = m.KVLoraRank

	// Handle RoPE scaling if present
	if m.RopeScaling.Type != "" {
		kv["minicpm3.rope.scaling.type"] = m.RopeScaling.Type
		if m.RopeScaling.Factor > 0 {
			kv["minicpm3.rope.scaling.factor"] = m.RopeScaling.Factor
		} else {
			kv["minicpm3.rope.scaling.factor"] = float32(1.0)
		}
	}

	return kv
}

func (m *minicpm3Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var addRopeFactors sync.Once

	out := make([]*ggml.Tensor, 0, len(ts)+2)
	for _, t := range ts {
		if strings.HasPrefix(t.Name(), "blk.0.") {
			addRopeFactors.Do(func() {
				out = append(out, &ggml.Tensor{
					Name:     "rope_factors_long.weight",
					Kind:     0,
					Shape:    []uint64{uint64(len(m.RopeScaling.LongFactor))},
					WriterTo: m.RopeScaling.LongFactor,
				}, &ggml.Tensor{
					Name:     "rope_factors_short.weight",
					Kind:     0,
					Shape:    []uint64{uint64(len(m.RopeScaling.ShortFactor))},
					WriterTo: m.RopeScaling.ShortFactor,
				})
			})
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

func (m *minicpm3Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_a_proj", "attn_q_a",
		"self_attn.q_a_layernorm", "attn_q_a_norm",
		"self_attn.q_b_proj", "attn_q_b",
		"self_attn.kv_a_proj_with_mqa", "attn_kv_a_mqa",
		"self_attn.kv_a_layernorm", "attn_kv_a_norm",
		"self_attn.kv_b_proj", "attn_kv_b",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
		"model.norm", "output_norm",
	}
}
