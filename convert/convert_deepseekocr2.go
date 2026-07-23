package convert

import (
	"fmt"

	"github.com/ollama/ollama/fs/ggml"
)

type deepseekocr2 struct {
	ModelParameters
	LanguageConfig struct {
		MaxPositionEmbeddings uint32 `json:"max_position_embeddings"`
		HiddenSize            uint32 `json:"hidden_size"`
		HiddenLayers          uint32 `json:"num_hidden_layers"`
		IntermediateSize      uint32 `json:"intermediate_size"`
		NumAttentionHeads     uint32 `json:"num_attention_heads"`
		NumKeyValueHeads      uint32 `json:"num_key_value_heads"`
		NumRoutedExperts      uint32 `json:"n_routed_experts"`
		NumSharedExperts      uint32 `json:"n_shared_experts"`
		NumExpertsPerToken    uint32 `json:"num_experts_per_tok"`
		FirstKDenseReplace    uint32 `json:"first_k_dense_replace"`
	} `json:"language_config"`

	VisionConfig struct {
		ImageSize uint32 `json:"image_size"`
		Width     struct {
			Qwen2 struct {
				Dim uint32 `json:"dim"`
			} `json:"qwen2-0-5b"`
			Sam struct {
				GlobalAttentionIndexes []int32 `json:"global_attn_indexes"`
				Heads                  uint32  `json:"heads"`
				Layers                 uint32  `json:"layers"`
				Width                  uint32  `json:"width"`
			} `json:"sam_vit_b"`
		}
	} `json:"vision_config"`
}

func (m *deepseekocr2) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "deepseekocr2"
	kv["block_count"] = m.LanguageConfig.HiddenLayers
	kv["context_length"] = m.LanguageConfig.MaxPositionEmbeddings
	kv["embedding_length"] = m.LanguageConfig.HiddenSize
	kv["feed_forward_length"] = m.LanguageConfig.IntermediateSize
	kv["attention.head_count"] = m.LanguageConfig.NumAttentionHeads
	kv["attention.head_count_kv"] = m.LanguageConfig.NumKeyValueHeads
	kv["expert_count"] = m.LanguageConfig.NumRoutedExperts
	kv["expert_used_count"] = m.LanguageConfig.NumExpertsPerToken
	kv["leading_dense_block_count"] = m.LanguageConfig.FirstKDenseReplace

	// Vision capability marker (enables CLI image support)
	kv["vision.block_count"] = uint32(24)

	// Qwen2 vision encoder config
	kv["qwen2.embedding_length"] = m.VisionConfig.Width.Qwen2.Dim
	// Qwen2 has fixed architecture: 24 layers, 14 heads, 2 KV heads, 4864 intermediate
	kv["qwen2.block_count"] = uint32(24)
	kv["qwen2.head_count"] = uint32(14)
	kv["qwen2.head_count_kv"] = uint32(2)
	kv["qwen2.intermediate_size"] = uint32(4864)

	// SAM config (same as v1)
	kv["sam.block_count"] = m.VisionConfig.Width.Sam.Layers
	kv["sam.embedding_length"] = m.VisionConfig.Width.Sam.Width
	kv["sam.head_count"] = m.VisionConfig.Width.Sam.Heads
	kv["sam.global_attention_indexes"] = m.VisionConfig.Width.Sam.GlobalAttentionIndexes
	return kv
}

func (m *deepseekocr2) Tensors(s []Tensor) (out []*ggml.Tensor) {
	merges := make([]merge, m.LanguageConfig.HiddenLayers*3)
	for i := range m.LanguageConfig.HiddenLayers {
		merges[i*3+0] = merge{
			fmt.Sprintf("blk.%d.mlp.experts.*.gate_proj.weight", i),
			fmt.Sprintf("blk.%d.ffn_gate_exps.weight", i),
		}
		merges[i*3+1] = merge{
			fmt.Sprintf("blk.%d.mlp.experts.*.up_proj.weight", i),
			fmt.Sprintf("blk.%d.ffn_up_exps.weight", i),
		}
		merges[i*3+2] = merge{
			fmt.Sprintf("blk.%d.mlp.experts.*.down_proj.weight", i),
			fmt.Sprintf("blk.%d.ffn_down_exps.weight", i),
		}
	}

	out, s = mergeTensors(s, merges...)
	for _, t := range s {
		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}
	return out
}

func (m *deepseekocr2) Replacements() []string {
	return []string{
		// Text model (LLM)
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"post_attention_layernorm", "ffn_norm",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",
		"mlp.gate", "ffn_gate_inp",
		"mlp.shared_experts.gate_proj", "ffn_gate_shexp",
		"mlp.shared_experts.up_proj", "ffn_up_shexp",
		"mlp.shared_experts.down_proj", "ffn_down_shexp",
		"model.norm", "output_norm",
		"lm_head", "output",

		// Qwen2 vision encoder (replaces CLIP in v1)
		// IMPORTANT: "model.qwen2_model.model.model.norm" must come BEFORE "model.norm"
		// to prevent "model.norm" from matching inside the Qwen2 path
		"model.qwen2_model.model.model.norm", "q.output_norm",
		"model.qwen2_model.model.model.layers", "q.blk",
		"model.qwen2_model.query_768", "q.query_768",
		"model.qwen2_model.query_1024", "q.query_1024",

		// Projector
		"model.projector", "mm",
		//nolint:misspell // this misspelling is upstream. fixing it breaks the model
		"model.view_seperator", "mm.view_seperator",

		// SAM (same as v1)
		"model.sam_model.patch_embed.proj", "s.patch_embd",
		"model.sam_model.pos_embed", "s.position_embd",
		"model.sam_model.blocks", "s.blk",
		"model.sam_model.neck", "s.neck",
		"model.sam_model.net_", "s.net_",
	}
}
