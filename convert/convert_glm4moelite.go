package convert

import (
	"cmp"
	"fmt"
	"log/slog"
	"regexp"
	"strconv"

	"github.com/ollama/ollama/fs/ggml"
)

type glm4MoeLiteModel struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`

	RopeTheta     float32 `json:"rope_theta"`
	QKNopeHeadDim uint32  `json:"qk_nope_head_dim"`
	QKRopeHeadDim uint32  `json:"qk_rope_head_dim"`
	KVLoraRank    uint32  `json:"kv_lora_rank"`
	QLoraRank     uint32  `json:"q_lora_rank"`
	VHeadDim      uint32  `json:"v_head_dim"`

	ExpertCount            uint32  `json:"n_routed_experts"`
	ExpertSharedCount      uint32  `json:"n_shared_experts"`
	ExpertIntermediateSize uint32  `json:"moe_intermediate_size"`
	ExpertUsedCount        uint32  `json:"num_experts_per_tok"`
	ExpertWeightsNorm      bool    `json:"norm_topk_prob"`
	ExpertWeightsScale     float32 `json:"routed_scaling_factor"`

	LeadingDenseBlockCount uint32 `json:"first_k_dense_replace"`
}

func (p *glm4MoeLiteModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "glm4moelite"
	kv["general.type"] = "model"
	kv["glm4moelite.block_count"] = p.HiddenLayers

	numHeads := p.NumAttentionHeads
	numKVHeads := p.NumKeyValueHeads

	kv["glm4moelite.attention.head_count"] = numHeads
	kv["glm4moelite.attention.head_count_kv"] = numKVHeads
	kv["glm4moelite.attention.key_length"] = p.QKNopeHeadDim + p.QKRopeHeadDim
	kv["glm4moelite.attention.kv_lora_rank"] = p.KVLoraRank
	kv["glm4moelite.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["glm4moelite.attention.q_lora_rank"] = p.QLoraRank
	kv["glm4moelite.attention.value_length"] = p.VHeadDim
	kv["glm4moelite.context_length"] = p.MaxPositionEmbeddings
	kv["glm4moelite.embedding_length"] = p.HiddenSize
	kv["glm4moelite.expert_count"] = p.ExpertCount
	kv["glm4moelite.expert_feed_forward_length"] = p.ExpertIntermediateSize
	kv["glm4moelite.expert_shared_count"] = p.ExpertSharedCount

	// GLM-4.7 MOE Lite uses sigmoid gating
	kv["glm4moelite.expert_gating_func"] = uint32(2) // sigmoid
	kv["glm4moelite.expert_used_count"] = p.ExpertUsedCount
	kv["glm4moelite.expert_weights_norm"] = p.ExpertWeightsNorm
	kv["glm4moelite.expert_weights_scale"] = p.ExpertWeightsScale
	kv["glm4moelite.feed_forward_length"] = p.IntermediateSize
	kv["glm4moelite.leading_dense_block_count"] = p.LeadingDenseBlockCount

	kv["glm4moelite.rope.dimension_count"] = p.QKRopeHeadDim
	kv["glm4moelite.rope.freq_base"] = cmp.Or(p.RopeTheta, float32(1000000.0))

	kv["tokenizer.ggml.pre"] = "glm4"

	return kv
}

func (p *glm4MoeLiteModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.kv_a_proj_with_mqa", "attn_kv_a_mqa",
		"self_attn.kv_a_layernorm", "attn_kv_a_norm",
		"self_attn.kv_b_proj", "attn_kv_b",
		"self_attn.q_a_proj", "attn_q_a",
		"self_attn.q_a_layernorm", "attn_q_a_norm",
		"self_attn.q_b_proj", "attn_q_b",
		"self_attn.o_proj", "attn_output",
		"post_attention_layernorm", "ffn_norm",
		"mlp.shared_experts.down_proj", "ffn_down_shexp",
		"mlp.shared_experts.gate_proj", "ffn_gate_shexp",
		"mlp.shared_experts.up_proj", "ffn_up_shexp",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"mlp.gate.e_score_correction_bias", "exp_probs_b.bias",
		"mlp.gate", "ffn_gate_inp",
	}
}

func (p *glm4MoeLiteModel) Tensors(s []Tensor) (out []*ggml.Tensor) {
	merges := make([]merge, p.HiddenLayers*3)
	for i := range p.HiddenLayers {
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

	out, s = mergeTensors(s, merges...)
	for _, t := range s {
		// skip any additional layers (such as the Multi-Token Prediction layer)
		if skipLayer(t.Name(), p.HiddenLayers) {
			slog.Debug("skipping layer", "name", t.Name())
			continue
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
