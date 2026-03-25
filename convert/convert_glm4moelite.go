package convert

import (
	"cmp"
	"fmt"
	"log/slog"
	"regexp"
	"strconv"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

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

	ExpertGroupCount     uint32 `json:"n_group"`
	ExpertGroupUsedCount uint32 `json:"topk_group"`
}

func (p *glm4MoeLiteModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "deepseek2"
	kv["general.type"] = "model"
	kv["deepseek2.block_count"] = p.HiddenLayers

	numHeads := p.NumAttentionHeads

	kv["deepseek2.attention.head_count"] = numHeads
	// deepseek2 uses MLA which is MQA (1 KV head) — upstream sets this to 1 regardless
	// of the HF num_key_value_heads. All attention heads share the compressed KV.
	kv["deepseek2.attention.head_count_kv"] = uint32(1)
	// MLA key/value lengths: upstream convention is key_length = kv_lora_rank + qk_rope
	// (the "absorbed" MLA head dim used at runtime) and key_length_mla = qk_nope + qk_rope
	// (the traditional head dim, used for q_b shape computation).
	kv["deepseek2.attention.key_length"] = p.KVLoraRank + p.QKRopeHeadDim
	kv["deepseek2.attention.kv_lora_rank"] = p.KVLoraRank
	kv["deepseek2.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	kv["deepseek2.attention.q_lora_rank"] = p.QLoraRank
	kv["deepseek2.attention.value_length"] = p.KVLoraRank
	kv["deepseek2.context_length"] = p.MaxPositionEmbeddings
	kv["deepseek2.embedding_length"] = p.HiddenSize
	kv["deepseek2.expert_count"] = p.ExpertCount
	kv["deepseek2.expert_feed_forward_length"] = p.ExpertIntermediateSize
	kv["deepseek2.expert_shared_count"] = p.ExpertSharedCount

	kv["deepseek2.expert_group_count"] = cmp.Or(p.ExpertGroupCount, 1)
	kv["deepseek2.expert_group_used_count"] = cmp.Or(p.ExpertGroupUsedCount, 1)
	kv["deepseek2.expert_used_count"] = p.ExpertUsedCount
	kv["deepseek2.vocab_size"] = p.VocabSize
	kv["deepseek2.expert_weights_norm"] = p.ExpertWeightsNorm
	kv["deepseek2.expert_weights_scale"] = p.ExpertWeightsScale
	kv["deepseek2.feed_forward_length"] = p.IntermediateSize
	kv["deepseek2.leading_dense_block_count"] = p.LeadingDenseBlockCount

	kv["deepseek2.rope.dimension_count"] = p.QKRopeHeadDim
	kv["deepseek2.rope.freq_base"] = cmp.Or(p.RopeTheta, float32(1000000.0))

	kv["deepseek2.attention.key_length_mla"] = p.QKNopeHeadDim + p.QKRopeHeadDim
	kv["deepseek2.attention.value_length_mla"] = p.VHeadDim

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

// repackKVB extracts K or V from the combined KV_B tensor for MLA absorption.
// K output row-major: [n_head, kv_lora_rank, qk_nope] -> GGML ne[]={qk_nope, kv_lora_rank, n_head}
// V output row-major: [n_head, v_head, kv_lora_rank] -> GGML ne[]={kv_lora_rank, v_head, n_head}
func (p *glm4MoeLiteModel) repackKVB(extractK bool, kvFirst bool, numHeads int) Repacker {
	qkNope := int(p.QKNopeHeadDim)
	vHeadDim := int(p.VHeadDim)
	kvLoraRank := int(p.KVLoraRank)
	kvPerHead := qkNope + vHeadDim

	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		dims := make([]int, len(shape))
		for i := range shape {
			dims[i] = int(shape[i])
		}

		var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
		var err error

		// Normalize to [n_head * (qk_nope + v_head), kv_lora_rank] layout
		if kvFirst {
			tt, err = tensor.Transpose(tt, 1, 0)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
		}

		// Reshape to [n_head, qk_nope + v_head, kv_lora_rank]
		if err := tt.Reshape(numHeads, kvPerHead, kvLoraRank); err != nil {
			return nil, err
		}

		if extractK {
			// Slice K: [n_head, qk_nope, kv_lora_rank]
			tt, err = tt.Slice(nil, tensor.S(0, qkNope), nil)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
			// Transpose to [n_head, kv_lora_rank, qk_nope]
			tt, err = tensor.Transpose(tt, 0, 2, 1)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
		} else {
			// Slice V: [n_head, v_head, kv_lora_rank] - already correct layout
			tt, err = tt.Slice(nil, tensor.S(qkNope, kvPerHead), nil)
			if err != nil {
				return nil, err
			}
			tt = tensor.Materialize(tt)
		}

		if err := tt.Reshape(tt.Shape().TotalSize()); err != nil {
			return nil, err
		}
		return native.VectorF32(tt.(*tensor.Dense))
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

		// Split attn_kv_b into separate attn_k_b and attn_v_b for MLA absorption
		if strings.HasSuffix(t.Name(), ".attn_kv_b.weight") {
			qkNope := int(p.QKNopeHeadDim)
			vHeadDim := int(p.VHeadDim)
			kvLoraRank := int(p.KVLoraRank)
			kvPerHead := qkNope + vHeadDim
			numHeads := int(p.NumAttentionHeads)
			kvFirst := true
			if len(t.Shape()) == 2 {
				switch {
				case int(t.Shape()[0]) == kvLoraRank:
					if kvPerHead > 0 && int(t.Shape()[1])%kvPerHead == 0 {
						numHeads = int(t.Shape()[1]) / kvPerHead
					}
					kvFirst = true
				case int(t.Shape()[1]) == kvLoraRank:
					if kvPerHead > 0 && int(t.Shape()[0])%kvPerHead == 0 {
						numHeads = int(t.Shape()[0]) / kvPerHead
					}
					kvFirst = false
				default:
					slog.Warn("deepseek2: unexpected attn_kv_b layout", "name", t.Name(), "shape", t.Shape())
				}
			}

			kTensor := t.Clone()
			kTensor.SetRepacker(p.repackKVB(true, kvFirst, numHeads))
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(t.Name(), "attn_kv_b", "attn_k_b", 1),
				Kind:     t.Kind(),
				Shape:    []uint64{uint64(numHeads), uint64(kvLoraRank), uint64(qkNope)},
				WriterTo: kTensor,
			})

			vTensor := t.Clone()
			vTensor.SetRepacker(p.repackKVB(false, kvFirst, numHeads))
			out = append(out, &ggml.Tensor{
				Name:     strings.Replace(t.Name(), "attn_kv_b", "attn_v_b", 1),
				Kind:     t.Kind(),
				Shape:    []uint64{uint64(numHeads), uint64(vHeadDim), uint64(kvLoraRank)},
				WriterTo: vTensor,
			})
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
