package convert

import (
	"fmt"

	"github.com/ollama/ollama/fs/ggml"
)

type deepseekocr struct {
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
			Vision struct {
				Heads     uint32 `json:"heads"`
				ImageSize uint32 `json:"image_size"`
				Layers    uint32 `json:"layers"`
				PatchSize uint32 `json:"patch_size"`
				Width     uint32 `json:"width"`
			} `json:"clip-l-14-224"`
			Sam struct {
				GlobalAttentionIndexes []int32 `json:"global_attn_indexes"`
				Heads                  uint32  `json:"heads"`
				Layers                 uint32  `json:"layers"`
				Width                  uint32  `json:"width"`
			} `json:"sam_vit_b"`
		}
	} `json:"vision_config"`
}

func (m *deepseekocr) KV(t *Tokenizer) KV {
	kv := m.ModelParameters.KV(t)
	kv["general.architecture"] = "deepseekocr"
	kv["block_count"] = m.LanguageConfig.HiddenLayers
	kv["context_length"] = m.LanguageConfig.MaxPositionEmbeddings
	kv["embedding_length"] = m.LanguageConfig.HiddenSize
	kv["feed_forward_length"] = m.LanguageConfig.IntermediateSize
	kv["attention.head_count"] = m.LanguageConfig.NumAttentionHeads
	kv["attention.head_count_kv"] = m.LanguageConfig.NumKeyValueHeads
	kv["expert_count"] = m.LanguageConfig.NumRoutedExperts
	kv["expert_used_count"] = m.LanguageConfig.NumExpertsPerToken
	kv["leading_dense_block_count"] = m.LanguageConfig.FirstKDenseReplace

	kv["vision.block_count"] = m.VisionConfig.Width.Vision.Layers
	kv["vision.embedding_length"] = m.VisionConfig.Width.Vision.Width
	kv["vision.head_count"] = m.VisionConfig.Width.Vision.Heads
	kv["vision.image_size"] = m.VisionConfig.Width.Vision.ImageSize
	kv["vision.patch_size"] = m.VisionConfig.Width.Vision.PatchSize

	kv["sam.block_count"] = m.VisionConfig.Width.Sam.Layers
	kv["sam.embedding_length"] = m.VisionConfig.Width.Sam.Width
	kv["sam.head_count"] = m.VisionConfig.Width.Sam.Heads
	kv["sam.global_attention_indexes"] = m.VisionConfig.Width.Sam.GlobalAttentionIndexes
	return kv
}

func (m *deepseekocr) Tensors(s []Tensor) (out []*ggml.Tensor) {
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

func (m *deepseekocr) Replacements() []string {
	return []string{
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

		"model.vision_model", "v",
		"embeddings.patch_embedding", "patch_embd",
		"embeddings.class_embedding", "class_embd",
		"embeddings.position_embedding", "position_embd",
		"transformer.layers", "blk",

		"model.projector", "mm",
		"model.image_newline", "mm.image_newline",
		//nolint:misspell // this misspelling is upstream. fixing it breaks the model
		"model.view_seperator", "mm.view_seperator",

		"model.sam_model.patch_embed.proj", "s.patch_embd",
		"model.sam_model.pos_embed", "s.position_embd",
		"model.sam_model.blocks", "s.blk",
		"model.sam_model.neck", "s.neck",
		"model.sam_model.net_", "s.net_",
	}
}
