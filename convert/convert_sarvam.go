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

// sarvamMLAModel handles conversion of sarvamai/sarvam-105b (SarvamMLAForCausalLM).
// This model uses Multi-head Latent Attention (MLA) like DeepSeek V3 and maps to the
// "deepseek2" GGML architecture. Key difference from DeepSeek V3: no Q compression
// (q_lora_rank=0), uses self_attn.q_proj directly.
type sarvamMLAModel struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`

	RopeTheta     float32 `json:"rope_theta"`
	DefaultTheta  float32 `json:"default_theta"`
	QKNopeHeadDim uint32  `json:"qk_nope_head_dim"`
	QKRopeHeadDim uint32  `json:"qk_rope_head_dim"`
	KVLoraRank    uint32  `json:"kv_lora_rank"`
	VHeadDim      uint32  `json:"v_head_dim"`

	ExpertCount            uint32  `json:"num_experts"`
	ExpertSharedCount      uint32  `json:"num_shared_experts"`
	ExpertIntermediateSize uint32  `json:"moe_intermediate_size"`
	ExpertUsedCount        uint32  `json:"num_experts_per_tok"`
	ExpertWeightsScale     float32 `json:"routed_scaling_factor"`

	LeadingDenseBlockCount uint32 `json:"first_k_dense_replace"`

	RopeScaling struct {
		Factor                        float32 `json:"factor"`
		OriginalMaxPositionEmbeddings uint32  `json:"original_max_position_embeddings"`
		Type                          string  `json:"type"`
		MScaleAllDim                  float32 `json:"mscale_all_dim"`
	} `json:"rope_scaling"`
}

func (p *sarvamMLAModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "deepseek2"
	kv["general.type"] = "model"
	kv["deepseek2.block_count"] = p.HiddenLayers

	kv["deepseek2.attention.head_count"] = p.NumAttentionHeads
	kv["deepseek2.attention.head_count_kv"] = p.NumAttentionHeads // MLA uses all heads
	kv["deepseek2.attention.key_length"] = p.QKNopeHeadDim + p.QKRopeHeadDim
	kv["deepseek2.attention.kv_lora_rank"] = p.KVLoraRank
	kv["deepseek2.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["deepseek2.attention.q_lora_rank"] = uint32(0) // no Q compression
	kv["deepseek2.attention.value_length"] = p.VHeadDim
	kv["deepseek2.context_length"] = p.MaxPositionEmbeddings
	kv["deepseek2.embedding_length"] = p.HiddenSize
	kv["deepseek2.expert_count"] = p.ExpertCount
	kv["deepseek2.expert_feed_forward_length"] = p.ExpertIntermediateSize
	kv["deepseek2.expert_shared_count"] = p.ExpertSharedCount

	// sigmoid scoring (e_score_correction_bias implies sigmoid)
	kv["deepseek2.expert_gating_func"] = uint32(2)
	kv["deepseek2.expert_used_count"] = p.ExpertUsedCount
	kv["deepseek2.expert_weights_norm"] = false
	kv["deepseek2.expert_weights_scale"] = p.ExpertWeightsScale
	kv["deepseek2.feed_forward_length"] = p.IntermediateSize
	kv["deepseek2.leading_dense_block_count"] = p.LeadingDenseBlockCount

	kv["deepseek2.rope.dimension_count"] = p.QKRopeHeadDim
	kv["deepseek2.rope.freq_base"] = cmp.Or(p.RopeTheta, p.DefaultTheta, float32(10000.0))
	kv["deepseek2.rope.scaling.factor"] = p.RopeScaling.Factor
	kv["deepseek2.rope.scaling.original_context_length"] = p.RopeScaling.OriginalMaxPositionEmbeddings

	ropeType := p.RopeScaling.Type
	if ropeType == "deepseek_yarn" {
		ropeType = "yarn"
	}
	kv["deepseek2.rope.scaling.type"] = ropeType
	kv["deepseek2.rope.scaling.yarn_log_multiplier"] = 0.1 * p.RopeScaling.MScaleAllDim

	kv["tokenizer.ggml.pre"] = "deepseek-v3"

	return kv
}

func (p *sarvamMLAModel) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.kv_a_proj_with_mqa", "attn_kv_a_mqa",
		"self_attn.kv_a_layernorm", "attn_kv_a_norm",
		"self_attn.kv_b_proj", "attn_kv_b",
		"self_attn.q_proj", "attn_q",
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

func (p *sarvamMLAModel) Tensors(s []Tensor) (out []*ggml.Tensor) {
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

// sarvamMoEModel handles conversion of sarvamai/sarvam-30b (SarvamMoEForCausalLM).
// This model uses standard GQA (not MLA) with fused QKV, QK normalization, and
// DeepSeek-style MoE with shared experts, expert bias, and sigmoid routing.
type sarvamMoEModel struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	HeadDim               uint32  `json:"head_dim"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`

	RopeTheta float32 `json:"rope_theta"`

	ExpertCount                        uint32  `json:"num_experts"`
	ExpertSharedCount                  uint32  `json:"num_shared_experts"`
	ExpertIntermediateSize             uint32  `json:"moe_intermediate_size"`
	SharedExpertIntermediateSize       uint32  `json:"moe_shared_expert_intermediate_size"`
	ExpertUsedCount                    uint32  `json:"num_experts_per_tok"`
	NormTopkProb                       bool    `json:"norm_topk_prob"`
	ExpertWeightsScale                 float32 `json:"routed_scaling_factor"`
	LeadingDenseBlockCount             uint32  `json:"first_k_dense_replace"`
}

func (p *sarvamMoEModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "sarvam_moe"
	kv["general.type"] = "model"

	kv["sarvam_moe.block_count"] = p.HiddenLayers
	kv["sarvam_moe.context_length"] = p.MaxPositionEmbeddings
	kv["sarvam_moe.embedding_length"] = p.HiddenSize
	kv["sarvam_moe.feed_forward_length"] = p.IntermediateSize

	kv["sarvam_moe.attention.head_count"] = p.NumAttentionHeads
	kv["sarvam_moe.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["sarvam_moe.attention.key_length"] = p.HeadDim
	kv["sarvam_moe.attention.value_length"] = p.HeadDim
	kv["sarvam_moe.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS

	kv["sarvam_moe.expert_count"] = p.ExpertCount
	kv["sarvam_moe.expert_used_count"] = p.ExpertUsedCount
	kv["sarvam_moe.expert_feed_forward_length"] = p.ExpertIntermediateSize
	kv["sarvam_moe.expert_shared_count"] = p.ExpertSharedCount
	kv["sarvam_moe.expert_shared_feed_forward_length"] = cmp.Or(p.SharedExpertIntermediateSize, p.ExpertIntermediateSize)
	kv["sarvam_moe.expert_weights_norm"] = p.NormTopkProb
	kv["sarvam_moe.expert_weights_scale"] = p.ExpertWeightsScale
	kv["sarvam_moe.expert_gating_func"] = uint32(2) // sigmoid
	kv["sarvam_moe.leading_dense_block_count"] = p.LeadingDenseBlockCount

	kv["sarvam_moe.rope.freq_base"] = cmp.Or(p.RopeTheta, float32(10000.0))
	kv["sarvam_moe.rope.dimension_count"] = p.HeadDim

	kv["tokenizer.ggml.pre"] = "deepseek-v3"

	return kv
}

func (p *sarvamMoEModel) Replacements() []string {
	return []string{
		"model.word_embeddings", "token_embd",
		"lm_head", "output",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"attention.query_key_value", "attn_qkv",
		"attention.dense", "attn_output",
		"attention.query_layernorm", "attn_q_norm",
		"attention.key_layernorm", "attn_k_norm",
		"post_attention_layernorm", "ffn_norm",
		"mlp.shared_experts.down_proj", "ffn_down_shexp",
		"mlp.shared_experts.gate_proj", "ffn_gate_shexp",
		"mlp.shared_experts.up_proj", "ffn_up_shexp",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"mlp.gate.expert_bias", "exp_probs_b.bias",
		"mlp.gate", "ffn_gate_inp",
	}
}

func (p *sarvamMoEModel) Tensors(s []Tensor) (out []*ggml.Tensor) {
	// Merge expert tensors for MoE layers
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

	qSize := uint64(p.NumAttentionHeads) * uint64(p.HeadDim)
	kSize := uint64(p.NumKeyValueHeads) * uint64(p.HeadDim)
	vSize := uint64(p.NumKeyValueHeads) * uint64(p.HeadDim)

	out, s = mergeTensors(s, merges...)
	for _, t := range s {
		if skipLayer(t.Name(), p.HiddenLayers) {
			slog.Debug("skipping layer", "name", t.Name())
			continue
		}

		// Split fused QKV tensor into separate Q, K, V
		if strings.HasSuffix(t.Name(), ".attn_qkv.weight") {
			for tt := range splitDim(t, 0,
				split{
					Replacer: strings.NewReplacer("attn_qkv", "attn_q"),
					dim:      int(qSize),
				},
				split{
					Replacer: strings.NewReplacer("attn_qkv", "attn_k"),
					dim:      int(kSize),
				},
				split{
					Replacer: strings.NewReplacer("attn_qkv", "attn_v"),
					dim:      int(vSize),
				},
			) {
				out = append(out, tt)
			}
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


