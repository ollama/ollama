package convert

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

var _ MultimodalConverter = (*deepseekocr)(nil)

func isDeepseekOCRVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.") || strings.HasPrefix(name, "s.")
}

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
		MoeIntermediateSize   uint32 `json:"moe_intermediate_size"`
		FirstKDenseReplace    uint32 `json:"first_k_dense_replace"`
		NGroup                uint32 `json:"n_group"`
		TopKGroup             uint32 `json:"topk_group"`
		QKRopeHeadDim         uint32 `json:"qk_rope_head_dim"`
		VocabSize             uint32 `json:"vocab_size"`
	} `json:"language_config"`

	ProjectorConfig struct {
		InputDim uint32 `json:"input_dim"`
		NEmbed   uint32 `json:"n_embed"`
	} `json:"projector_config"`

	VisionConfig struct {
		ImageSize uint32  `json:"image_size"`
		MlpRatio  float32 `json:"mlp_ratio"`
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
	kv["general.architecture"] = "deepseek2-ocr"
	kv["block_count"] = m.LanguageConfig.HiddenLayers
	kv["context_length"] = m.LanguageConfig.MaxPositionEmbeddings
	kv["embedding_length"] = m.LanguageConfig.HiddenSize
	kv["feed_forward_length"] = m.LanguageConfig.IntermediateSize
	kv["attention.head_count"] = m.LanguageConfig.NumAttentionHeads
	kv["attention.head_count_kv"] = m.LanguageConfig.NumKeyValueHeads
	kv["attention.layer_norm_rms_epsilon"] = float32(1e-6)
	kv["expert_count"] = m.LanguageConfig.NumRoutedExperts
	kv["expert_feed_forward_length"] = m.LanguageConfig.MoeIntermediateSize
	kv["expert_used_count"] = m.LanguageConfig.NumExpertsPerToken
	kv["leading_dense_block_count"] = m.LanguageConfig.FirstKDenseReplace
	kv["expert_shared_count"] = m.LanguageConfig.NumSharedExperts
	kv["expert_group_count"] = m.LanguageConfig.NGroup
	kv["expert_group_used_count"] = m.LanguageConfig.TopKGroup
	kv["rope.dimension_count"] = m.LanguageConfig.QKRopeHeadDim
	kv["vocab_size"] = m.LanguageConfig.VocabSize
	return kv
}

// ProjectorKV returns KV metadata for the deepseek-ocr vision projector.
func (m *deepseekocr) ProjectorKV(t *Tokenizer) KV {
	return KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "deepseekocr",
		"clip.has_vision_encoder":                  true,
		"clip.use_gelu":                            true,
		"clip.vision.block_count":                  m.VisionConfig.Width.Vision.Layers,
		"clip.vision.embedding_length":             m.VisionConfig.Width.Vision.Width,
		"clip.vision.feed_forward_length":          uint32(64),
		"clip.vision.attention.head_count":         m.VisionConfig.Width.Vision.Heads,
		"clip.vision.attention.layer_norm_epsilon": float32(1e-6),
		"clip.vision.image_size":                   m.VisionConfig.Width.Vision.ImageSize,
		"clip.vision.image_mean":                   []float32{0.5, 0.5, 0.5},
		"clip.vision.image_std":                    []float32{0.5, 0.5, 0.5},
		"clip.vision.patch_size":                   m.VisionConfig.Width.Vision.PatchSize,
		"clip.vision.projection_dim":               m.ProjectorConfig.NEmbed,
		"clip.vision.projector.scale_factor":       uint32(1),
		"clip.vision.window_size":                  uint32(14),
		"clip.vision.sam.block_count":              m.VisionConfig.Width.Sam.Layers,
		"clip.vision.sam.embedding_length":         m.VisionConfig.Width.Sam.Width,
		"clip.vision.sam.head_count":               m.VisionConfig.Width.Sam.Heads,
	}
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

// TextTensors returns only text model tensors (no vision/SAM/projector).
func (m *deepseekocr) TextTensors(ts []Tensor, t *Tokenizer) []*ggml.Tensor {
	var textOnly []Tensor
	for _, tensor := range ts {
		if !isDeepseekOCRVisionTensor(tensor.Name()) {
			textOnly = append(textOnly, tensor)
		}
	}
	return m.Tensors(textOnly)
}

// deepseekOCRProjectorReplacer maps our tensor names to what llama-server expects.
var deepseekOCRProjectorReplacer = strings.NewReplacer(
	// Vision transformer block renames (v.blk.*)
	"self_attn.out_proj", "attn_out",
	"self_attn.qkv_proj", "attn_qkv",
	"layer_norm1", "ln1",
	"layer_norm2", "ln2",
	"mlp.fc1", "ffn_up",
	"mlp.fc2", "ffn_down",
	// Vision pre-layernorm
	"pre_layrnorm", "pre_ln",
	// SAM tensors: s.* → v.sam.*
	"s.blk.", "v.sam.blk.",
	"s.patch_embd.", "v.sam.patch_embd.",
	"s.position_embd", "v.sam.pos_embd.weight",
	"s.neck.", "v.sam.neck.",
	"s.net_", "v.sam.net_",
	// SAM attention
	"attn.proj.", "attn.out.",
	"attn.rel_pos_h", "attn.pos_h.weight",
	"attn.rel_pos_w", "attn.pos_w.weight",
	// SAM norms
	".norm1.", ".pre_ln.",
	".norm2.", ".post_ln.",
	// Projector
	"mm.layers.", "mm.model.fc.",
	"mm.image_newline", "v.image_newline",
	"mm.view_separator", "v.view_separator",
)

// ProjectorTensors returns only vision/SAM/projector tensors with names
// remapped for llama-server's clip/mtmd system.
func (m *deepseekocr) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		if !isDeepseekOCRVisionTensor(t.Name()) {
			continue
		}
		name := deepseekOCRProjectorReplacer.Replace(t.Name())
		out = append(out, &ggml.Tensor{
			Name:     name,
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
