package convert

import (
	"bytes"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen25VLModel struct {
	ModelParameters
	HiddenSize            uint32  `json:"hidden_size"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	RopeTheta             float32 `json:"rope_theta"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`

	VisionModel struct {
	} `json:"vision_config"`
}

var _ ModelConverter = (*qwen25VLModel)(nil)

func (q *qwen25VLModel) KV(t *Tokenizer) ggml.KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen25vl"
	kv["qwen25vl.block_count"] = q.HiddenLayers
	kv["qwen25vl.context_length"] = q.MaxPositionEmbeddings
	kv["qwen25vl.embedding_length"] = q.HiddenSize
	kv["qwen25vl.feed_forward_length"] = q.IntermediateSize
	kv["qwen25vl.attention.head_count"] = q.NumAttentionHeads
	kv["qwen25vl.attention.head_count_kv"] = q.NumKeyValueHeads
	kv["qwen25vl.rope.freq_base"] = q.RopeTheta
	kv["qwen25vl.attention.layer_norm_rms_epsilon"] = q.RMSNormEPS

	return kv
}

func (q *qwen25VLModel) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	for _, t := range ts {
		if strings.HasSuffix(t.Name(), "patch_embed.proj.weight") {
			var buf bytes.Buffer
			t.WriteTo(&buf)
			newTensors := splitPatchEmbed(buf, t.Kind(), t.Shape())
			out = append(out, newTensors...)
		} else {
			out = append(out, ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
	}

	return out
}

func (p *qwen25VLModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"visual.blocks", "v.blk",
		"input_layernorm", "attn_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.q_proj", "attn_q",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
		"model.norm", "output_norm",
	}
}
