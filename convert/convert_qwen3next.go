package convert

import (
	"fmt"
	"io/fs"
	"math"
	"slices"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen3NextModel struct {
	ModelParameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	NumHiddenLayers       uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	HeadDim               uint32  `json:"head_dim"`
	RopeTheta             float32 `json:"rope_theta"`
	RMSNormEPS            float32 `json:"rms_norm_eps"`

	// MoE config
	NumExperts             uint32 `json:"num_experts"`
	NumExpertsPerToken     uint32 `json:"num_experts_per_tok"`
	NormTopkProb           bool   `json:"norm_topk_prob"`
	MoEIntermediateSize    uint32 `json:"moe_intermediate_size"`
	SharedExpertIntermSize uint32 `json:"shared_expert_intermediate_size"`

	// Hybrid attention config
	FullAttentionInterval uint32 `json:"full_attention_interval"`

	// Linear attention (Gated Delta Net) config
	LinearConvKernelDim uint32 `json:"linear_conv_kernel_dim"`
	LinearKeyHeadDim    uint32 `json:"linear_key_head_dim"`
	LinearNumKeyHeads   uint32 `json:"linear_num_key_heads"`
	LinearNumValueHeads uint32 `json:"linear_num_value_heads"`
	LinearValueHeadDim  uint32 `json:"linear_value_head_dim"`

	// RoPE config
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
	RopeScaling         struct {
		Type   string     `json:"type"`
		Factor ropeFactor `json:"factor"`
	} `json:"rope_scaling"`
}

var _ ModelConverter = (*qwen3NextModel)(nil)

func (q *qwen3NextModel) parseMore(_ fs.FS) error {
	if q.NumHiddenLayers == 0 {
		return fmt.Errorf("qwen3next: num_hidden_layers must be set")
	}
	if q.NumAttentionHeads == 0 {
		return fmt.Errorf("qwen3next: num_attention_heads must be set")
	}
	if q.NumKeyValueHeads == 0 {
		return fmt.Errorf("qwen3next: num_key_value_heads must be set")
	}
	if q.HeadDim == 0 {
		return fmt.Errorf("qwen3next: head_dim must be set")
	}
	if q.RopeTheta == 0 {
		return fmt.Errorf("qwen3next: rope_theta must be set")
	}
	if q.PartialRotaryFactor <= 0 || q.PartialRotaryFactor > 1 {
		return fmt.Errorf("qwen3next: partial_rotary_factor must be in (0,1], got %v", q.PartialRotaryFactor)
	}
	if q.LinearNumKeyHeads == 0 || q.LinearNumValueHeads == 0 || q.LinearKeyHeadDim == 0 || q.LinearValueHeadDim == 0 {
		return fmt.Errorf("qwen3next: linear attention config must be set (linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim)")
	}
	if q.FullAttentionInterval == 0 {
		return fmt.Errorf("qwen3next: full_attention_interval must be set")
	}
	if q.FullAttentionInterval > q.NumHiddenLayers {
		return fmt.Errorf("qwen3next: full_attention_interval (%d) exceeds num_hidden_layers (%d)", q.FullAttentionInterval, q.NumHiddenLayers)
	}

	hasFull := false
	for i := range q.NumHiddenLayers {
		if (i+1)%q.FullAttentionInterval == 0 {
			hasFull = true
			break
		}
	}
	if !hasFull {
		return fmt.Errorf("qwen3next: head_count_kv would be all zeros (full_attention_interval=%d, num_hidden_layers=%d)", q.FullAttentionInterval, q.NumHiddenLayers)
	}

	return nil
}

func (q *qwen3NextModel) KV(t *Tokenizer) KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen3next"
	kv["tokenizer.ggml.pre"] = "qwen2"
	kv["block_count"] = q.NumHiddenLayers
	kv["context_length"] = q.MaxPositionEmbeddings
	kv["embedding_length"] = q.HiddenSize
	kv["feed_forward_length"] = q.IntermediateSize
	kv["attention.head_count"] = q.NumAttentionHeads
	headDim := q.HeadDim
	if headDim == 0 && q.NumAttentionHeads > 0 {
		headDim = q.HiddenSize / q.NumAttentionHeads
	}
	kv["attention.key_length"] = headDim
	kv["attention.value_length"] = headDim
	kv["attention.layer_norm_rms_epsilon"] = q.RMSNormEPS
	kv["rope.freq_base"] = q.RopeTheta

	// RoPE dimension count (partial rotary)
	// partial_rotary_factor = 0.25 means only 25% of head_dim uses RoPE
	partialRotary := q.PartialRotaryFactor
	if partialRotary > 0 && partialRotary <= 1 {
		kv["rope.dimension_count"] = uint32(float32(headDim) * partialRotary)
	}

	// MoE config
	if q.NumExperts > 0 {
		kv["expert_count"] = q.NumExperts
		kv["expert_used_count"] = q.NumExpertsPerToken
		kv["norm_top_k_prob"] = q.NormTopkProb
		if q.MoEIntermediateSize > 0 {
			kv["expert_feed_forward_length"] = q.MoEIntermediateSize
		}
		if q.SharedExpertIntermSize > 0 {
			kv["expert_shared_feed_forward_length"] = q.SharedExpertIntermSize
		}
	}

	// SSM/Linear attention config
	// d_inner = linear_value_head_dim * linear_num_value_heads
	dInner := q.LinearValueHeadDim * q.LinearNumValueHeads
	kv["ssm.inner_size"] = dInner
	kv["ssm.state_size"] = q.LinearKeyHeadDim        // head_k_dim
	kv["ssm.group_count"] = q.LinearNumKeyHeads      // num_k_heads
	kv["ssm.time_step_rank"] = q.LinearNumValueHeads // num_v_heads
	kv["ssm.conv_kernel"] = q.LinearConvKernelDim
	interval := q.FullAttentionInterval
	kv["full_attention_interval"] = interval

	// Build per-layer KV head count array to identify layer types
	// 0 = recurrent (linear attention), non-zero = full attention
	kvHeadCounts := make([]uint32, q.NumHiddenLayers)
	for i := range q.NumHiddenLayers {
		// Full attention every full_attention_interval layers (starting at interval-1)
		if interval > 0 && (i+1)%interval == 0 {
			kvHeadCounts[i] = q.NumKeyValueHeads
		}
		// else stays 0 (recurrent layer)
	}
	kv["attention.head_count_kv"] = kvHeadCounts

	// RoPE scaling
	if q.RopeScaling.Type != "" {
		kv["rope.scaling.type"] = q.RopeScaling.Type
		kv["rope.scaling.factor"] = q.RopeScaling.Factor
	}

	return kv
}

func (q *qwen3NextModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	// Create merges for expert tensors - stack individual experts into batched tensors
	merges := make([]merge, q.NumHiddenLayers*3)
	for i := range q.NumHiddenLayers {
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

	// Merge expert tensors
	merged, remaining := mergeTensors(ts, merges...)
	out = append(out, merged...)

	// Process remaining tensors
	for _, t := range remaining {
		name := t.Name()
		shape := t.Shape()

		// Split linear_attn.in_proj_qkvz (ssm_in) into attn_qkv + attn_gate when possible
		if strings.HasSuffix(name, ".ssm_in.weight") {
			if qkv, gate, ok := q.splitQKVZTensor(t); ok {
				out = append(out, qkv, gate)
				continue
			}
			panic(fmt.Sprintf("qwen3next: failed to split %s into attn_qkv/attn_gate (shape=%v)", name, shape))
		}

		switch {
		// Add 1 to norm weights (except ssm_norm which is linear_attn.norm)
		// This matches the Python converter behavior for qwen3next
		case strings.HasSuffix(name, "_norm.weight") && !strings.HasSuffix(name, ".ssm_norm.weight"):
			t.SetRepacker(q.addOne)
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    slices.Clone(shape),
				WriterTo: t,
			})

		// Handle linear attention A_log -> ssm_a (negate and exp)
		// Note: name has already been transformed by Replacements at this point
		case strings.HasSuffix(name, ".ssm_a"):
			t.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
				// Compute -exp(A_log)
				result := make([]float32, len(data))
				for i, v := range data {
					// -exp(v)
					result[i] = -float32(math.Exp(float64(v)))
				}
				return result, nil
			})
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    slices.Clone(shape),
				WriterTo: t,
			})

		// Squeeze conv1d weights: [1, D, K] or [D, 1, K] -> [D, K]
		case strings.HasSuffix(name, ".ssm_conv1d.weight"):
			newShape := slices.Clone(shape)
			if len(shape) == 3 {
				if shape[0] == 1 {
					// [1, D, K] -> [D, K]
					newShape = []uint64{shape[1], shape[2]}
				} else if shape[1] == 1 {
					// [D, 1, K] -> [D, K]
					newShape = []uint64{shape[0], shape[2]}
				}
			}
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    newShape,
				WriterTo: t,
			})
		// Squeeze shared expert gate: [D, 1] or [1, D] -> [D]
		case strings.HasSuffix(name, ".ffn_gate_inp_shexp.weight"):
			newShape := slices.Clone(shape)
			if len(shape) == 2 {
				if shape[0] == 1 && shape[1] > 1 {
					newShape = []uint64{shape[1]}
				} else if shape[1] == 1 && shape[0] > 1 {
					newShape = []uint64{shape[0]}
				}
			}
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    newShape,
				WriterTo: t,
			})

		default:
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    slices.Clone(shape),
				WriterTo: t,
			})
		}
	}

	return out
}

type qkvzSplitSpec struct {
	hidden    int
	headKDim  int
	headVDim  int
	numKHeads int
	numVHeads int
	qkvzDim   int
	qkvOut    int
	gateOut   int
}

func (q *qwen3NextModel) qkvzSpec(shape []uint64) (qkvzSplitSpec, bool) {
	if len(shape) != 2 {
		return qkvzSplitSpec{}, false
	}

	numKHeads := int(q.LinearNumKeyHeads)
	numVHeads := int(q.LinearNumValueHeads)
	headKDim := int(q.LinearKeyHeadDim)
	headVDim := int(q.LinearValueHeadDim)
	if numKHeads == 0 || numVHeads == 0 || headKDim == 0 || headVDim == 0 {
		return qkvzSplitSpec{}, false
	}
	if numVHeads%numKHeads != 0 {
		return qkvzSplitSpec{}, false
	}

	hidden := int(shape[1])
	vPerHead := headVDim * (numVHeads / numKHeads)
	qkvzDim := 2*headKDim + 2*vPerHead
	expectedOut := qkvzDim * numKHeads
	if int(shape[0]) != expectedOut {
		return qkvzSplitSpec{}, false
	}

	return qkvzSplitSpec{
		hidden:    hidden,
		headKDim:  headKDim,
		headVDim:  headVDim,
		numKHeads: numKHeads,
		numVHeads: numVHeads,
		qkvzDim:   qkvzDim,
		qkvOut:    2*headKDim*numKHeads + headVDim*numVHeads,
		gateOut:   headVDim * numVHeads,
	}, true
}

func (q *qwen3NextModel) splitQKVZTensor(t Tensor) (*ggml.Tensor, *ggml.Tensor, bool) {
	spec, ok := q.qkvzSpec(t.Shape())
	if !ok {
		return nil, nil, false
	}

	qkvTensor := t.Clone()
	qkvTensor.SetRepacker(q.repackQKVZ(spec, false))

	gateTensor := t.Clone()
	gateTensor.SetRepacker(q.repackQKVZ(spec, true))

	qkvName := strings.Replace(t.Name(), "ssm_in", "attn_qkv", 1)
	gateName := strings.Replace(t.Name(), "ssm_in", "attn_gate", 1)

	return &ggml.Tensor{
			Name:     qkvName,
			Kind:     t.Kind(),
			Shape:    []uint64{uint64(spec.qkvOut), uint64(spec.hidden)},
			WriterTo: qkvTensor,
		}, &ggml.Tensor{
			Name:     gateName,
			Kind:     t.Kind(),
			Shape:    []uint64{uint64(spec.gateOut), uint64(spec.hidden)},
			WriterTo: gateTensor,
		}, true
}

func (q *qwen3NextModel) repackQKVZ(spec qkvzSplitSpec, extractGate bool) Repacker {
	vPerHead := spec.headVDim * (spec.numVHeads / spec.numKHeads)

	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		dims := make([]int, len(shape))
		for i := range shape {
			dims[i] = int(shape[i])
		}

		var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
		var err error

		// Convert to [hidden, out_features] layout for slicing
		tt, err = tensor.Transpose(tt, 1, 0)
		if err != nil {
			return nil, err
		}
		tt = tensor.Materialize(tt)

		if err := tt.Reshape(spec.hidden, spec.numKHeads, spec.qkvzDim); err != nil {
			return nil, err
		}

		offset := 0
		qSlice, err := tt.Slice(nil, nil, tensor.S(offset, offset+spec.headKDim))
		if err != nil {
			return nil, err
		}
		offset += spec.headKDim
		kSlice, err := tt.Slice(nil, nil, tensor.S(offset, offset+spec.headKDim))
		if err != nil {
			return nil, err
		}
		offset += spec.headKDim
		vSlice, err := tt.Slice(nil, nil, tensor.S(offset, offset+vPerHead))
		if err != nil {
			return nil, err
		}
		offset += vPerHead
		zSlice, err := tt.Slice(nil, nil, tensor.S(offset, offset+vPerHead))
		if err != nil {
			return nil, err
		}

		qMat := tensor.Materialize(qSlice).(*tensor.Dense)
		kMat := tensor.Materialize(kSlice).(*tensor.Dense)
		vMat := tensor.Materialize(vSlice).(*tensor.Dense)
		zMat := tensor.Materialize(zSlice).(*tensor.Dense)

		if err := qMat.Reshape(spec.hidden, spec.numKHeads*spec.headKDim); err != nil {
			return nil, err
		}
		if err := kMat.Reshape(spec.hidden, spec.numKHeads*spec.headKDim); err != nil {
			return nil, err
		}
		if err := vMat.Reshape(spec.hidden, spec.numKHeads*vPerHead); err != nil {
			return nil, err
		}
		if err := zMat.Reshape(spec.hidden, spec.numKHeads*vPerHead); err != nil {
			return nil, err
		}

		var out tensor.Tensor
		if extractGate {
			out = zMat
		} else {
			out, err = tensor.Concat(1, qMat, kMat, vMat)
			if err != nil {
				return nil, err
			}
		}

		out = tensor.Materialize(out)
		out, err = tensor.Transpose(out, 1, 0)
		if err != nil {
			return nil, err
		}
		out = tensor.Materialize(out)

		if err := out.Reshape(out.Shape().TotalSize()); err != nil {
			return nil, err
		}

		return native.VectorF32(out.(*tensor.Dense))
	}
}

// addOne adds 1.0 to all elements in the tensor (for norm weights)
func (*qwen3NextModel) addOne(_ string, data []float32, shape []uint64) ([]float32, error) {
	n := tensor.New(tensor.WithShape(int(shape[0])), tensor.WithBacking(data))
	ones := tensor.Ones(tensor.Float32, int(shape[0]))

	n, err := n.Add(ones)
	if err != nil {
		return nil, err
	}

	ts, err := native.SelectF32(n, 0)
	if err != nil {
		return nil, err
	}

	var f32s []float32
	for _, t := range ts {
		f32s = append(f32s, t...)
	}

	return f32s, nil
}

func (q *qwen3NextModel) Replacements() []string {
	return []string{
		// Embeddings and output
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",

		// Layer norms
		"input_layernorm", "attn_norm",
		"post_attention_layernorm", "post_attention_norm",

		// Full attention (self_attn)
		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",

		// Linear attention (Gated Delta Net)
		"linear_attn.in_proj_qkvz", "ssm_in",
		"linear_attn.in_proj_ba", "ssm_ba",
		"linear_attn.conv1d", "ssm_conv1d",
		"linear_attn.dt_bias", "ssm_dt",
		"linear_attn.dt_proj", "ssm_dt",
		"linear_attn.A_log", "ssm_a",
		"linear_attn.norm", "ssm_norm",
		"linear_attn.out_proj", "ssm_out",

		// MoE (experts are stacked via mergeTensors, not replaced here)
		"mlp.gate.weight", "ffn_gate_inp.weight",
		"mlp.shared_expert.down_proj", "ffn_down_shexp",
		"mlp.shared_expert.gate_proj", "ffn_gate_shexp",
		"mlp.shared_expert.up_proj", "ffn_up_shexp",
		"mlp.shared_expert_gate", "ffn_gate_inp_shexp",

		// Dense FFN (if any layers use it)
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
	}
}
