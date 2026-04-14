package convert

import (
	"cmp"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type hybridPattern string

func (p *hybridPattern) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		*p = ""
		return nil
	}

	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		*p = hybridPattern(strings.TrimSpace(single))
		return nil
	}

	var parts []string
	if err := json.Unmarshal(data, &parts); err == nil {
		*p = hybridPattern(strings.Join(parts, ""))
		return nil
	}

	return fmt.Errorf("hybrid_override_pattern must be a string or string array")
}

type nemotronHModel struct {
	ModelParameters
	MaxPositionEmbeddings uint32        `json:"max_position_embeddings"`
	HiddenSize            uint32        `json:"hidden_size"`
	NumHiddenLayers       uint32        `json:"num_hidden_layers"`
	NumAttentionHeads     uint32        `json:"num_attention_heads"`
	NumKeyValueHeads      uint32        `json:"num_key_value_heads"`
	HeadDim               uint32        `json:"head_dim"`
	LayerNormEpsilon      float32       `json:"layer_norm_epsilon"`
	NormEpsilon           float32       `json:"norm_eps"`
	RopeTheta             float32       `json:"rope_theta"`
	PartialRotaryFactor   float32       `json:"partial_rotary_factor"`
	ConvKernel            uint32        `json:"conv_kernel"`
	SSMStateSize          uint32        `json:"ssm_state_size"`
	MambaNumHeads         uint32        `json:"mamba_num_heads"`
	MambaHeadDim          uint32        `json:"mamba_head_dim"`
	NGroups               uint32        `json:"n_groups"`
	IntermediateSize      uint32        `json:"intermediate_size"`
	HybridOverridePattern hybridPattern `json:"hybrid_override_pattern"`

	// MoE
	NumExperts                  uint32  `json:"num_experts"`
	NumSharedExperts            uint32  `json:"num_shared_experts"`
	NRoutedExperts              uint32  `json:"n_routed_experts"`
	NSharedExperts              uint32  `json:"n_shared_experts"`
	NumExpertsPerTok            uint32  `json:"num_experts_per_tok"`
	MoEIntermediateSize         uint32  `json:"moe_intermediate_size"`
	MoESharedExpertIntermediate uint32  `json:"moe_shared_expert_intermediate_size"`
	NormTopKProb                bool    `json:"norm_topk_prob"`
	RoutedScalingFactor         float32 `json:"routed_scaling_factor"`
	ExpertGroupCount            uint32  `json:"n_group"`
	ExpertGroupUsedCount        uint32  `json:"topk_group"`
}

var _ ModelConverter = (*nemotronHModel)(nil)

func (n *nemotronHModel) parseMore(_ fs.FS) error {
	if n.NumHiddenLayers == 0 {
		return fmt.Errorf("nemotron_h: num_hidden_layers must be set")
	}
	if n.HiddenSize == 0 {
		return fmt.Errorf("nemotron_h: hidden_size must be set")
	}
	if n.NumAttentionHeads == 0 {
		return fmt.Errorf("nemotron_h: num_attention_heads must be set")
	}
	if n.HeadDim == 0 {
		if n.HiddenSize%n.NumAttentionHeads != 0 {
			return fmt.Errorf("nemotron_h: hidden_size (%d) must be divisible by num_attention_heads (%d)", n.HiddenSize, n.NumAttentionHeads)
		}
		n.HeadDim = n.HiddenSize / n.NumAttentionHeads
	}
	if n.NumKeyValueHeads == 0 {
		n.NumKeyValueHeads = n.NumAttentionHeads
	}
	if n.ConvKernel == 0 {
		return fmt.Errorf("nemotron_h: conv_kernel must be set")
	}
	if n.SSMStateSize == 0 {
		return fmt.Errorf("nemotron_h: ssm_state_size must be set")
	}
	if n.ssmHeadCount() == 0 {
		return fmt.Errorf("nemotron_h: mamba_num_heads must be set")
	}
	if n.MambaHeadDim == 0 {
		return fmt.Errorf("nemotron_h: mamba_head_dim must be set")
	}
	if n.NGroups == 0 {
		n.NGroups = 1
	}

	if _, _, err := n.layerArrays(); err != nil {
		return err
	}

	if n.isMoE() {
		if n.routedExpertCount() == 0 {
			return fmt.Errorf("nemotron_h: routed expert count must be set for MoE models")
		}
		if n.NumExpertsPerTok == 0 {
			return fmt.Errorf("nemotron_h: num_experts_per_tok must be set for MoE models")
		}
		if n.NumExpertsPerTok > n.routedExpertCount() {
			return fmt.Errorf("nemotron_h: num_experts_per_tok (%d) cannot exceed expert_count (%d)", n.NumExpertsPerTok, n.routedExpertCount())
		}
		if n.moeIntermediateSize() == 0 {
			return fmt.Errorf("nemotron_h: moe_intermediate_size must be set for MoE models")
		}
	}

	return nil
}

func (n *nemotronHModel) isMoE() bool {
	return cmp.Or(n.routedExpertCount(), n.NumExpertsPerTok, n.MoEIntermediateSize) > 0
}

func (n *nemotronHModel) routedExpertCount() uint32 {
	return cmp.Or(n.NRoutedExperts, n.NumExperts)
}

func (n *nemotronHModel) sharedExpertCount() uint32 {
	return cmp.Or(n.NSharedExperts, n.NumSharedExperts)
}

func (n *nemotronHModel) ssmHeadCount() uint32 {
	return n.MambaNumHeads
}

func (n *nemotronHModel) ssmInnerSize() uint32 {
	return n.MambaHeadDim * n.ssmHeadCount()
}

func (n *nemotronHModel) epsilon() float32 {
	return cmp.Or(n.NormEpsilon, n.LayerNormEpsilon, float32(1e-5))
}

func (n *nemotronHModel) moeIntermediateSize() uint32 {
	return cmp.Or(n.MoEIntermediateSize, n.IntermediateSize)
}

func (n *nemotronHModel) denseIntermediateSize() uint32 {
	return cmp.Or(n.IntermediateSize, n.MoEIntermediateSize)
}

func (n *nemotronHModel) layerArrays() (headCountKV []uint32, ffnLengths []uint32, err error) {
	pattern := strings.TrimSpace(string(n.HybridOverridePattern))
	if pattern == "" {
		return nil, nil, fmt.Errorf("nemotron_h: hybrid_override_pattern must be set")
	}

	runes := []rune(pattern)
	if len(runes) != int(n.NumHiddenLayers) {
		return nil, nil, fmt.Errorf("nemotron_h: hybrid_override_pattern length (%d) must match num_hidden_layers (%d)", len(runes), n.NumHiddenLayers)
	}

	headCountKV = make([]uint32, n.NumHiddenLayers)
	ffnLengths = make([]uint32, n.NumHiddenLayers)

	attnKVHeads := cmp.Or(n.NumKeyValueHeads, n.NumAttentionHeads)
	moeFFN := n.moeIntermediateSize()
	denseFFN := n.denseIntermediateSize()

	for i, layerType := range runes {
		switch layerType {
		case 'M':
			// Recurrent layer: no KV heads and no FFN.
		case '*', 'A':
			// Attention-only layer.
			headCountKV[i] = attnKVHeads
		case 'E':
			// MoE layer.
			if moeFFN == 0 {
				return nil, nil, fmt.Errorf("nemotron_h: moe layer at index %d but moe_intermediate_size is zero", i)
			}
			ffnLengths[i] = moeFFN
		case '-':
			// Dense FFN layer.
			if denseFFN == 0 {
				return nil, nil, fmt.Errorf("nemotron_h: dense FFN layer at index %d but intermediate_size is zero", i)
			}
			ffnLengths[i] = denseFFN
		default:
			return nil, nil, fmt.Errorf("nemotron_h: unsupported layer type %q in hybrid_override_pattern at index %d", layerType, i)
		}
	}

	return headCountKV, ffnLengths, nil
}

func (n *nemotronHModel) KV(t *Tokenizer) KV {
	kv := n.ModelParameters.KV(t)

	arch := "nemotron_h"
	if n.isMoE() {
		arch = "nemotron_h_moe"
	}
	kv["general.architecture"] = arch
	kv["block_count"] = n.NumHiddenLayers
	kv["context_length"] = n.MaxPositionEmbeddings
	kv["embedding_length"] = n.HiddenSize
	kv["attention.head_count"] = n.NumAttentionHeads
	kv["attention.key_length"] = n.HeadDim
	kv["attention.value_length"] = n.HeadDim
	kv["attention.layer_norm_epsilon"] = n.epsilon()
	kv["attention.layer_norm_rms_epsilon"] = n.epsilon()
	kv["rope.freq_base"] = cmp.Or(n.RopeTheta, float32(10000))
	if n.PartialRotaryFactor > 0 && n.PartialRotaryFactor <= 1 {
		kv["rope.dimension_count"] = uint32(float32(n.HeadDim) * n.PartialRotaryFactor)
	}

	if headCountKV, ffnLengths, err := n.layerArrays(); err == nil {
		kv["attention.head_count_kv"] = headCountKV
		kv["feed_forward_length"] = ffnLengths
	}

	kv["ssm.conv_kernel"] = n.ConvKernel
	kv["ssm.inner_size"] = n.ssmInnerSize()
	kv["ssm.state_size"] = n.SSMStateSize
	kv["ssm.group_count"] = n.NGroups
	kv["ssm.time_step_rank"] = n.ssmHeadCount()

	if n.isMoE() {
		kv["expert_count"] = n.routedExpertCount()
		kv["expert_used_count"] = n.NumExpertsPerTok
		kv["expert_feed_forward_length"] = n.moeIntermediateSize()
		if n.sharedExpertCount() > 0 {
			kv["expert_shared_count"] = n.sharedExpertCount()
		}
		if n.MoESharedExpertIntermediate > 0 {
			kv["expert_shared_feed_forward_length"] = n.MoESharedExpertIntermediate
		}
		kv["expert_weights_norm"] = n.NormTopKProb
		kv["expert_weights_scale"] = n.RoutedScalingFactor
		if n.ExpertGroupCount > 0 {
			kv["expert_group_count"] = n.ExpertGroupCount
		}
		if n.ExpertGroupUsedCount > 0 {
			kv["expert_group_used_count"] = n.ExpertGroupUsedCount
		}
	}

	return kv
}

func normalizeVectorShapeToColumn(shape []uint64) []uint64 {
	switch len(shape) {
	case 1:
		return []uint64{shape[0], 1}
	case 2:
		if shape[0] == 1 && shape[1] > 1 {
			return []uint64{shape[1], 1}
		}
		if shape[1] == 1 && shape[0] > 1 {
			return []uint64{shape[0], 1}
		}
	}

	return slices.Clone(shape)
}

func (n *nemotronHModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	remaining := ts
	if n.isMoE() {
		merges := make([]merge, 0, n.NumHiddenLayers*2)
		for i := range n.NumHiddenLayers {
			merges = append(merges, merge{
				fmt.Sprintf("blk.%d.mixer.experts.*.up_proj.weight", i),
				fmt.Sprintf("blk.%d.ffn_up_exps.weight", i),
			}, merge{
				fmt.Sprintf("blk.%d.mixer.experts.*.down_proj.weight", i),
				fmt.Sprintf("blk.%d.ffn_down_exps.weight", i),
			})
		}

		merged, rest := mergeTensors(ts, merges...)
		out = append(out, merged...)
		remaining = rest
	}

	nGroups := uint64(cmp.Or(n.NGroups, uint32(1)))
	for _, t := range remaining {
		name := t.Name()
		shape := slices.Clone(t.Shape())

		switch {
		case strings.HasSuffix(name, ".ssm_a"):
			shape = normalizeVectorShapeToColumn(shape)
			t.SetRepacker(func(_ string, data []float32, _ []uint64) ([]float32, error) {
				out := make([]float32, len(data))
				for i, v := range data {
					out[i] = -float32(math.Exp(float64(v)))
				}
				return out, nil
			})
		case strings.HasSuffix(name, ".ssm_d"):
			shape = normalizeVectorShapeToColumn(shape)
		case strings.HasSuffix(name, ".ssm_norm.weight"):
			switch len(shape) {
			case 1:
				if nGroups > 0 && shape[0]%nGroups == 0 {
					shape = []uint64{nGroups, shape[0] / nGroups}
				}
			case 2:
				if shape[0] == 1 && nGroups > 0 && shape[1]%nGroups == 0 {
					shape = []uint64{nGroups, shape[1] / nGroups}
				}
			}
		case strings.HasSuffix(name, ".ssm_conv1d.weight"):
			if len(shape) == 3 {
				if shape[0] == 1 {
					shape = []uint64{shape[1], shape[2]}
				} else if shape[1] == 1 {
					shape = []uint64{shape[0], shape[2]}
				}
			}
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    shape,
			WriterTo: t,
		})
	}

	return out
}

func (n *nemotronHModel) Replacements() []string {
	return []string{
		// Embedding and output
		"lm_head", "output",
		"backbone.embeddings", "token_embd",
		"backbone.norm_f", "output_norm",
		"backbone.layers", "blk",

		// Recurrent (Mamba2) tensors
		"mixer.in_proj", "ssm_in",
		"mixer.out_proj", "ssm_out",
		"mixer.dt_bias", "ssm_dt.bias",
		"mixer.A_log", "ssm_a",
		"mixer.D", "ssm_d",
		"mixer.conv1d", "ssm_conv1d",
		"mixer.norm.weight", "ssm_norm.weight",

		// Attention tensors
		"mixer.q_proj", "attn_q",
		"mixer.k_proj", "attn_k",
		"mixer.v_proj", "attn_v",
		"mixer.o_proj", "attn_output",

		// FFN / MoE tensors
		"mixer.gate.e_score_correction_bias", "exp_probs_b.bias",
		"mixer.gate", "ffn_gate_inp",
		"mixer.fc1_latent_proj", "ffn_latent_in",
		"mixer.fc2_latent_proj", "ffn_latent_out",
		"mixer.shared_experts.up_proj", "ffn_up_shexp",
		"mixer.shared_experts.down_proj", "ffn_down_shexp",
		"mixer.up_proj", "ffn_up",
		"mixer.down_proj", "ffn_down",

		// Per-layer pre-norm
		".norm.weight", ".attn_norm.weight",
	}
}
