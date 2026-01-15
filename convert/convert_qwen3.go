package convert

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

type qwen3Model struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	HeadDim               uint32  `json:"head_dim"`
	NumExperts            uint32  `json:"num_experts"`
	NumExpertsPerToken    uint32  `json:"num_experts_per_tok"`
	NormTopkProb          bool    `json:"norm_topk_prob"`
	RopeTheta             float32 `json:"rope_theta"`
	RopeScaling           struct {
		Type                          string     `json:"type"`
		Factor                        ropeFactor `json:"factor"`
		OriginalMaxPositionEmbeddings uint32     `json:"original_max_position_embeddings"`
		MropeSection                  []int32    `json:"mrope_section"`
	} `json:"rope_scaling"`
	RMSNormEPS float32 `json:"rms_norm_eps"`
}

// KV implements ModelConverter.
func (q *qwen3Model) KV(t *Tokenizer) KV {
	arch := "qwen3"
	if q.NumExperts > 0 {
		arch += "moe"
	}

	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = arch
	kv["block_count"] = q.HiddenLayers
	kv["context_length"] = q.MaxPositionEmbeddings
	kv["embedding_length"] = q.HiddenSize
	kv["feed_forward_length"] = q.IntermediateSize
	kv["attention.head_count"] = q.NumAttentionHeads
	kv["attention.head_count_kv"] = q.NumKeyValueHeads
	kv["attention.key_length"] = q.HeadDim
	kv["attention.value_length"] = q.HeadDim

	if q.NumExperts > 0 {
		kv["expert_count"] = q.NumExperts
		kv["expert_used_count"] = q.NumExpertsPerToken
		kv["norm_top_k_prob"] = q.NormTopkProb
	}

	kv["rope.freq_base"] = q.RopeTheta
	kv["attention.layer_norm_rms_epsilon"] = q.RMSNormEPS

	switch q.RopeScaling.Type {
	case "":
		// no scaling
	case "yarn":
		kv["rope.scaling.type"] = q.RopeScaling.Type
		kv["rope.scaling.factor"] = q.RopeScaling.Factor
	case "mrope", "default":
		kv["rope.mrope_section"] = q.RopeScaling.MropeSection
	default:
		panic("unknown rope scaling type")
	}
	return kv
}

// Tensors implements ModelConverter.
func (q *qwen3Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	// TODO: handle split experts

	for _, t := range ts {
		switch {
		case strings.Contains(t.Name(), "ffn_gate_up_exps"):
			afterFunc := func(t tensor.Tensor) (tensor.Tensor, error) { return tensor.Transpose(t, 0, 2, 1) }
			for t := range splitDim(t, 2,
				split{Replacer: strings.NewReplacer("gate_up", "gate"), afterFunc: afterFunc},
				split{Replacer: strings.NewReplacer("gate_up", "up"), afterFunc: afterFunc},
			) {
				t.Shape[1], t.Shape[2] = t.Shape[2], t.Shape[1]
				out = append(out, t)
			}
		case strings.Contains(t.Name(), "ffn_down_exps"):
			shape := slices.Clone(t.Shape())
			shape[1], shape[2] = shape[2], shape[1]
			t.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
				dims := make([]int, len(shape))
				for i := range shape {
					dims[i] = int(shape[i])
				}

				var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
				tt, err := tensor.Transpose(tt, 0, 2, 1)
				if err != nil {
					return nil, err
				}

				// flatten tensor so it can be written as a vector
				if err := tt.Reshape(tt.Shape().TotalSize()); err != nil {
					return nil, err
				}

				return native.VectorF32(tt.(*tensor.Dense))
			})
			out = append(out, &ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    shape,
				WriterTo: t,
			})
		default:
			out = append(out, &ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
	}

	return out
}

// Replacements implements ModelConverter.
func (q *qwen3Model) Replacements() []string {
	return []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.o_proj", "attn_output",
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"mlp.gate.weight", "ffn_gate_inp.weight",
		"mlp.experts.down_proj", "ffn_down_exps.weight",
		"mlp.experts.gate_up_proj", "ffn_gate_up_exps.weight",
		"post_attention_layernorm", "ffn_norm",
		"model.norm", "output_norm",
	}
}

var _ ModelConverter = (*qwen3Model)(nil)
