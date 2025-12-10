package convert

import (
	"cmp"
	"fmt"
	"log/slog"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type mistralLarge3Model struct {
	ModelParameters
	Dim            uint32  `json:"dim"`
	NumLayers      uint32  `json:"n_layers"`
	HeadDim        uint32  `json:"head_dim"`
	HiddenDim      uint32  `json:"hidden_dim"`
	NumHeads       uint32  `json:"n_heads"`
	NumKVHeads     uint32  `json:"n_kv_heads"`
	RopeTheta      float32 `json:"rope_theta"`
	NormEps        float32 `json:"norm_eps"`
	VocabSize      uint32  `json:"vocab_size"`
	TiedEmbeddings bool    `json:"tied_embeddings"`
	MaxPosEmbed    uint32  `json:"max_position_embeddings"`
	MaxSeqLen      uint32  `json:"max_seq_len"`

	// LoRA attention parameters (DeepSeek-style)
	QLoraRank     uint32 `json:"q_lora_rank"`
	QKRopeHeadDim uint32 `json:"qk_rope_head_dim"`
	QKNopeHeadDim uint32 `json:"qk_nope_head_dim"`
	KVLoraRank    uint32 `json:"kv_lora_rank"`
	VHeadDim      uint32 `json:"v_head_dim"`

	// ROPE scaling configurations
	Llama4Scaling struct {
		OrigMaxPosEmbed uint32  `json:"original_max_position_embeddings"`
		Beta            float32 `json:"beta"`
	} `json:"llama_4_scaling"`

	Yarn struct {
		OrigMaxPosEmbed uint32  `json:"original_max_position_embeddings"`
		Factor          float32 `json:"factor"`
		ApplyScale      bool    `json:"apply_scale"`
		Beta            float32 `json:"beta"`
		Alpha           float32 `json:"alpha"`
	} `json:"yarn"`

	// MOE configuration
	MOE struct {
		ExpertParallel        uint32  `json:"expert_parallel"`
		ExpertModelParallel   uint32  `json:"expert_model_parallel"`
		RouteEveryN           uint32  `json:"route_every_n"`
		FirstKDenseReplace    uint32  `json:"first_k_dense_replace"`
		NumExperts            uint32  `json:"num_experts"`
		NumExpertsPerTok      uint32  `json:"num_experts_per_tok"`
		NumExpertGroups       uint32  `json:"num_expert_groups"`
		NumExpertGroupsPerTok uint32  `json:"num_expert_groups_per_tok"`
		RoutedScale           float32 `json:"routed_scale"`
		ExpertHiddenDim       uint32  `json:"expert_hidden_dim"`
		NumSharedExperts      uint32  `json:"num_shared_experts"`
	} `json:"moe"`

	// Vision encoder configuration
	VisionEncoder struct {
		ImageTokenID               uint32  `json:"image_token_id"`
		ImageBreakTokenID          uint32  `json:"image_break_token_id"`
		ImageEndTokenID            uint32  `json:"image_end_token_id"`
		IntermediateSize           uint32  `json:"intermediate_size"`
		NumHiddenLayers            uint32  `json:"num_hidden_layers"`
		NumAttentionHeads          uint32  `json:"num_attention_heads"`
		MMProjectorID              string  `json:"mm_projector_id"`
		SpatialMergeSize           uint32  `json:"spatial_merge_size"`
		HiddenSize                 uint32  `json:"hidden_size"`
		NumChannels                uint32  `json:"num_channels"`
		ImageSize                  uint32  `json:"image_size"`
		MaxImageSize               uint32  `json:"max_image_size"`
		PatchSize                  uint32  `json:"patch_size"`
		RopeTheta                  float32 `json:"rope_theta"`
		AddPreMMProjectorLayerNorm bool    `json:"add_pre_mm_projector_layer_norm"`
		AdapterBias                bool    `json:"adapter_bias"`
	} `json:"vision_encoder"`
}

func (p *mistralLarge3Model) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "deepseek2" // Use deepseek2 architecture for runtime compatibility
	kv["general.type"] = "model"

	// Basic model parameters (using deepseek2 keys for compatibility)
	kv["deepseek2.vocab_size"] = p.VocabSize
	kv["deepseek2.block_count"] = p.NumLayers
	kv["deepseek2.context_length"] = cmp.Or(p.MaxPosEmbed, p.MaxSeqLen)
	kv["deepseek2.embedding_length"] = p.Dim
	kv["deepseek2.feed_forward_length"] = p.HiddenDim

	// Attention configuration
	kv["deepseek2.attention.head_count"] = p.NumHeads
	kv["deepseek2.attention.head_count_kv"] = p.NumKVHeads
	kv["deepseek2.attention.layer_norm_rms_epsilon"] = p.NormEps
	kv["deepseek2.attention.key_length"] = p.QKNopeHeadDim + p.QKRopeHeadDim
	kv["deepseek2.attention.value_length"] = p.VHeadDim

	// LoRA attention parameters
	kv["deepseek2.attention.q_lora_rank"] = p.QLoraRank
	kv["deepseek2.attention.kv_lora_rank"] = p.KVLoraRank

	// ROPE configuration
	kv["deepseek2.rope.dimension_count"] = p.QKRopeHeadDim
	kv["deepseek2.rope.freq_base"] = cmp.Or(p.RopeTheta, 10000.0)

	// ROPE scaling - map to deepseek2 format
	if p.Yarn.OrigMaxPosEmbed > 0 {
		kv["deepseek2.rope.scaling.factor"] = p.Yarn.Factor
		kv["deepseek2.rope.scaling.original_context_length"] = p.Yarn.OrigMaxPosEmbed
		kv["deepseek2.rope.scaling.type"] = "yarn"
		kv["deepseek2.rope.scaling.yarn_log_multiplier"] = float32(0.1) // mscale_all_dim * 0.1 as in llama.cpp
	}

	// MOE configuration
	if p.MOE.NumExperts > 0 {
		kv["deepseek2.expert_count"] = p.MOE.NumExperts
		kv["deepseek2.expert_used_count"] = p.MOE.NumExpertsPerTok
		kv["deepseek2.expert_shared_count"] = p.MOE.NumSharedExperts
		kv["deepseek2.expert_feed_forward_length"] = p.MOE.ExpertHiddenDim
		kv["deepseek2.expert_weights_scale"] = p.MOE.RoutedScale
		kv["deepseek2.leading_dense_block_count"] = p.MOE.FirstKDenseReplace
		kv["deepseek2.expert_weights_norm"] = true
		kv["deepseek2.expert_gating_func"] = uint32(1) // softmax
	}

	// Vision encoder configuration (if supported by deepseek2 runtime)
	if p.VisionEncoder.HiddenSize > 0 {
		kv["deepseek2.vision.block_count"] = p.VisionEncoder.NumHiddenLayers
		kv["deepseek2.vision.embedding_length"] = p.VisionEncoder.HiddenSize
		kv["deepseek2.vision.feed_forward_length"] = p.VisionEncoder.IntermediateSize
		kv["deepseek2.vision.attention.head_count"] = p.VisionEncoder.NumAttentionHeads
		kv["deepseek2.vision.image_size"] = p.VisionEncoder.ImageSize
		kv["deepseek2.vision.patch_size"] = p.VisionEncoder.PatchSize
		kv["deepseek2.vision.num_channels"] = p.VisionEncoder.NumChannels

		// Multimodal configuration
		kv["deepseek2.image_token_id"] = p.VisionEncoder.ImageTokenID
		kv["deepseek2.image_break_token_id"] = p.VisionEncoder.ImageBreakTokenID
		kv["deepseek2.image_end_token_id"] = p.VisionEncoder.ImageEndTokenID
		kv["deepseek2.spatial_merge_size"] = p.VisionEncoder.SpatialMergeSize
	}

	// Set tokenizer type - use default for Mistral models
	kv["tokenizer.ggml.pre"] = "tekken" // Let it use the default tokenizer preprocessing

	return kv
}

func (p *mistralLarge3Model) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}

func (p *mistralLarge3Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"tok_embeddings", "token_embd", // Mistral Large uses tok_embeddings instead of model.embed_tokens
		"norm", "output_norm",
		"language_model.", "",
		"layers", "blk", // Mistral 3 Large uses "layers" instead of "model.layers"
		"attention_norm", "attn_norm",

		// LoRA attention mappings (Mistral 3 Large style)
		"attention.wkv_a_with_mqa", "attn_kv_a_mqa",
		"attention.kv_a_norm", "attn_kv_a_norm",
		"attention.wkv_b", "attn_kv_b",
		"attention.wq_a", "attn_q_a",
		"attention.q_a_norm", "attn_q_a_norm",
		"attention.wq_b", "attn_q_b",
		"attention.wo", "attn_output",

		"ffn_norm", "ffn_norm", // Keep ffn_norm as is

		// MOE mappings for Mistral 3 Large
		"shared_experts.w2", "ffn_down_shexp",
		"shared_experts.w1", "ffn_gate_shexp",
		"shared_experts.w3", "ffn_up_shexp",
		"experts.*.w1", "ffn_gate_exps", // Will be merged in Tensors()
		"experts.*.w2", "ffn_down_exps", // Will be merged in Tensors()
		"experts.*.w3", "ffn_up_exps", // Will be merged in Tensors()
		"gate", "ffn_gate_inp",

		// Standard feed forward mappings (for non-MOE layers)
		"feed_forward.w1", "ffn_gate",
		"feed_forward.w2", "ffn_down",
		"feed_forward.w3", "ffn_up",

		// Mistral-specific tensor renaming
		".qscale_act", ".input_scale",
		".qscale_weight", ".weight_scale",

		// Vision encoder mappings - do we even need this?
		"vision_tower", "v",
		"ln_pre", "encoder_norm",
		"attention.q_proj", "attn_q",
		"attention.k_proj", "attn_k",
		"attention.v_proj", "attn_v",
		"attention.o_proj", "attn_output",
		"attention_norm", "attn_norm",
		"feed_forward.gate_proj", "ffn_gate",
		"feed_forward.down_proj", "ffn_down",
		"feed_forward.up_proj", "ffn_up",

		"multi_modal_projector", "mm",
		"patch_merger.merging_layer", "mm.patch_merger",
		"pre_mm_projector_norm", "mm.pre_norm",
		"vision_language_adapter.w_in", "mm.w_in",
		"vision_language_adapter.w_out", "mm.w_out",
	}
}

func (p *mistralLarge3Model) Tensors(s []Tensor) (out []*ggml.Tensor) {
	// Create merges for MOE expert tensors
	if p.MOE.NumExperts > 0 {
		merges := make([]merge, p.NumLayers*3)
		for i := range p.NumLayers {
			merges[i*3+0] = merge{
				fmt.Sprintf("blk.%d.experts.*.w1.weight", i),
				fmt.Sprintf("blk.%d.ffn_gate_exps.weight", i),
			}
			merges[i*3+1] = merge{
				fmt.Sprintf("blk.%d.experts.*.w3.weight", i),
				fmt.Sprintf("blk.%d.ffn_up_exps.weight", i),
			}
			merges[i*3+2] = merge{
				fmt.Sprintf("blk.%d.experts.*.w2.weight", i),
				fmt.Sprintf("blk.%d.ffn_down_exps.weight", i),
			}
		}
		out, s = mergeTensors(s, merges...)
	}

	skipLayer := func(n string, minValue uint32) bool {
		re := regexp.MustCompile(`^blk\.(\d+)`)
		matches := re.FindStringSubmatch(n)
		if matches == nil {
			return false
		}

		blkNum, err := strconv.Atoi(matches[1])
		if err != nil {
			return false
		}

		return uint32(blkNum) >= minValue
	}

	// Function to check if tensor should be skipped (vision components)
	skipVisionTensor := func(name string) bool {
		return strings.HasPrefix(name, "vision_") ||
			strings.HasPrefix(name, "patch_merger.") ||
			strings.Contains(name, "mm_projector")
	}

	for _, t := range s {
		name := t.Name()

		// Skip vision tensors (handled separately or not needed)
		if skipVisionTensor(name) {
			slog.Debug("skipping vision tensor", "name", name)
			continue
		}

		// Skip any additional layers beyond expected count
		if skipLayer(name, p.NumLayers) {
			slog.Debug("skipping extra layer", "name", name)
			continue
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}
	return out
}
