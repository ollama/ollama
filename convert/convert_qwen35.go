package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"
)

type ropeScalingFactor struct {
	Set    bool
	Scalar float32
	Vector ropeFactor
}

func (f *ropeScalingFactor) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}

	var scalar float32
	if err := json.Unmarshal(b, &scalar); err == nil {
		f.Set = true
		f.Scalar = scalar
		f.Vector = nil
		return nil
	}

	var vec ropeFactor
	if err := json.Unmarshal(b, &vec); err == nil {
		f.Set = true
		f.Vector = vec
		if len(vec) == 1 {
			f.Scalar = vec[0]
		}
		return nil
	}

	return fmt.Errorf("qwen35: invalid rope scaling factor: %s", string(b))
}

func (f ropeScalingFactor) Value() (any, bool) {
	if !f.Set {
		return nil, false
	}
	if len(f.Vector) == 1 {
		return f.Vector[0], true
	}
	if len(f.Vector) > 1 {
		return f.Vector, true
	}
	return f.Scalar, true
}

type qwen35RopeScaling struct {
	Type                          string            `json:"type"`
	Factor                        ropeScalingFactor `json:"factor"`
	OriginalMaxPositionEmbeddings uint32            `json:"original_max_position_embeddings"`
}

type qwen35RopeParameters struct {
	RopeTheta            float32            `json:"rope_theta"`
	RopeType             string             `json:"rope_type"`
	PartialRotaryFactor  float32            `json:"partial_rotary_factor"`
	Factor               ropeScalingFactor  `json:"factor"`
	OriginalMaxPositions uint32             `json:"original_max_position_embeddings"`
	MropeSection         []int32            `json:"mrope_section"`
	RopeScaling          *qwen35RopeScaling `json:"rope_scaling"`
}

type qwen35Model struct {
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

	NumExperts             uint32 `json:"num_experts"`
	NumExpertsPerToken     uint32 `json:"num_experts_per_tok"`
	NormTopkProb           bool   `json:"norm_topk_prob"`
	MoEIntermediateSize    uint32 `json:"moe_intermediate_size"`
	SharedExpertIntermSize uint32 `json:"shared_expert_intermediate_size"`

	LayerTypes            []string `json:"layer_types"`
	FullAttentionInterval uint32   `json:"full_attention_interval"`

	LinearConvKernelDim uint32 `json:"linear_conv_kernel_dim"`
	LinearKeyHeadDim    uint32 `json:"linear_key_head_dim"`
	LinearNumKeyHeads   uint32 `json:"linear_num_key_heads"`
	LinearNumValueHeads uint32 `json:"linear_num_value_heads"`
	LinearValueHeadDim  uint32 `json:"linear_value_head_dim"`

	PartialRotaryFactor float32              `json:"partial_rotary_factor"`
	RopeScaling         qwen35RopeScaling    `json:"rope_scaling"`
	RopeParameters      qwen35RopeParameters `json:"rope_parameters"`

	VisionModel struct {
		Depth             uint32  `json:"depth"`
		HiddenSize        uint32  `json:"hidden_size"`
		NumHeads          uint32  `json:"num_heads"`
		InChannels        uint32  `json:"in_channels"`
		PatchSize         uint32  `json:"patch_size"`
		SpatialMergeSize  uint32  `json:"spatial_merge_size"`
		RMSNormEps        float32 `json:"layer_norm_epsilon"`
		RopeTheta         float32 `json:"rope_theta"`
		TemporalPatchSize uint32  `json:"temporal_patch_size"`

		Size struct {
			ShortestEdge uint32 `json:"shortest_edge"`
			LongestEdge  uint32 `json:"longest_edge"`
		} `json:"size"`

		ImageMean []float32 `json:"image_mean"`
		ImageStd  []float32 `json:"image_std"`
	} `json:"vision_config"`
}

var _ ModelConverter = (*qwen35Model)(nil)

func (q *qwen35Model) parseMore(fsys fs.FS) error {
	if q.NumHiddenLayers == 0 {
		return fmt.Errorf("qwen35: num_hidden_layers must be set")
	}
	if q.NumAttentionHeads == 0 {
		return fmt.Errorf("qwen35: num_attention_heads must be set")
	}
	if q.NumKeyValueHeads == 0 {
		return fmt.Errorf("qwen35: num_key_value_heads must be set")
	}
	if q.HeadDim == 0 {
		return fmt.Errorf("qwen35: head_dim must be set")
	}

	if q.LinearNumKeyHeads == 0 || q.LinearNumValueHeads == 0 || q.LinearKeyHeadDim == 0 || q.LinearValueHeadDim == 0 {
		return fmt.Errorf("qwen35: linear attention config must be set (linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim)")
	}

	ropeTheta := q.ropeTheta()
	if ropeTheta == 0 {
		return fmt.Errorf("qwen35: rope_theta must be set")
	}
	partialRotary := q.partialRotaryFactor()
	if partialRotary <= 0 || partialRotary > 1 {
		return fmt.Errorf("qwen35: partial_rotary_factor must be in (0,1], got %v", partialRotary)
	}

	// Validate layer types or interval
	if len(q.LayerTypes) > 0 {
		if uint32(len(q.LayerTypes)) != q.NumHiddenLayers {
			return fmt.Errorf("qwen35: layer_types length (%d) does not match num_hidden_layers (%d)", len(q.LayerTypes), q.NumHiddenLayers)
		}
		hasFull := false
		hasLinear := false
		for i, t := range q.LayerTypes {
			switch t {
			case "full_attention":
				hasFull = true
			case "linear_attention":
				hasLinear = true
			default:
				return fmt.Errorf("qwen35: unknown layer_types[%d]=%q", i, t)
			}
		}
		if !hasFull || !hasLinear {
			return fmt.Errorf("qwen35: layer_types must include both full_attention and linear_attention")
		}
		return q.parseVisionConfig(fsys)
	}

	if q.FullAttentionInterval == 0 {
		return fmt.Errorf("qwen35: full_attention_interval must be set when layer_types is missing")
	}
	if q.FullAttentionInterval > q.NumHiddenLayers {
		return fmt.Errorf("qwen35: full_attention_interval (%d) exceeds num_hidden_layers (%d)", q.FullAttentionInterval, q.NumHiddenLayers)
	}

	hasFull := false
	for i := range q.NumHiddenLayers {
		if (i+1)%q.FullAttentionInterval == 0 {
			hasFull = true
			break
		}
	}
	if !hasFull {
		return fmt.Errorf("qwen35: head_count_kv would be all zeros (full_attention_interval=%d, num_hidden_layers=%d)", q.FullAttentionInterval, q.NumHiddenLayers)
	}

	return q.parseVisionConfig(fsys)
}

func (q *qwen35Model) parseVisionConfig(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "preprocessor_config.json")
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}

	return json.Unmarshal(bts, &q.VisionModel)
}

func (q *qwen35Model) ropeTheta() float32 {
	if q.RopeTheta != 0 {
		return q.RopeTheta
	}
	return q.RopeParameters.RopeTheta
}

func (q *qwen35Model) partialRotaryFactor() float32 {
	if q.PartialRotaryFactor != 0 {
		return q.PartialRotaryFactor
	}
	if q.RopeParameters.PartialRotaryFactor != 0 {
		return q.RopeParameters.PartialRotaryFactor
	}
	return 0.25
}

func (q *qwen35Model) ropeScalingType() string {
	if q.RopeScaling.Type != "" {
		return q.RopeScaling.Type
	}
	if q.RopeParameters.RopeScaling != nil && q.RopeParameters.RopeScaling.Type != "" {
		return q.RopeParameters.RopeScaling.Type
	}
	if q.RopeParameters.RopeType != "" && q.RopeParameters.RopeType != "default" {
		return q.RopeParameters.RopeType
	}
	return ""
}

func (q *qwen35Model) ropeScalingFactor() (any, bool) {
	if v, ok := q.RopeScaling.Factor.Value(); ok {
		return v, true
	}
	if q.RopeParameters.RopeScaling != nil {
		if v, ok := q.RopeParameters.RopeScaling.Factor.Value(); ok {
			return v, true
		}
	}
	if v, ok := q.RopeParameters.Factor.Value(); ok {
		return v, true
	}
	return nil, false
}

func (q *qwen35Model) ropeScalingOriginalContext() uint32 {
	if q.RopeScaling.OriginalMaxPositionEmbeddings != 0 {
		return q.RopeScaling.OriginalMaxPositionEmbeddings
	}
	if q.RopeParameters.RopeScaling != nil && q.RopeParameters.RopeScaling.OriginalMaxPositionEmbeddings != 0 {
		return q.RopeParameters.RopeScaling.OriginalMaxPositionEmbeddings
	}
	if q.RopeParameters.OriginalMaxPositions != 0 {
		return q.RopeParameters.OriginalMaxPositions
	}
	return 0
}

func (q *qwen35Model) KV(t *Tokenizer) KV {
	kv := q.ModelParameters.KV(t)
	if q.NumExperts > 0 {
		kv["general.architecture"] = "qwen35moe"
	} else {
		kv["general.architecture"] = "qwen35"
	}
	kv["tokenizer.ggml.pre"] = "qwen35"
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
	kv["rope.freq_base"] = q.ropeTheta()

	partialRotary := q.partialRotaryFactor()
	if partialRotary > 0 && partialRotary <= 1 {
		kv["rope.dimension_count"] = uint32(float32(headDim) * partialRotary)
	}

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

	dInner := q.LinearValueHeadDim * q.LinearNumValueHeads
	kv["ssm.inner_size"] = dInner
	kv["ssm.state_size"] = q.LinearKeyHeadDim
	kv["ssm.group_count"] = q.LinearNumKeyHeads
	kv["ssm.time_step_rank"] = q.LinearNumValueHeads
	kv["ssm.conv_kernel"] = q.LinearConvKernelDim

	kvHeadCounts := make([]uint32, q.NumHiddenLayers)
	if len(q.LayerTypes) > 0 {
		for i := range q.NumHiddenLayers {
			if i < uint32(len(q.LayerTypes)) && q.LayerTypes[i] == "full_attention" {
				kvHeadCounts[i] = q.NumKeyValueHeads
			}
		}
	} else {
		interval := q.FullAttentionInterval
		kv["full_attention_interval"] = interval
		for i := range q.NumHiddenLayers {
			if interval > 0 && (i+1)%interval == 0 {
				kvHeadCounts[i] = q.NumKeyValueHeads
			}
		}
	}
	kv["attention.head_count_kv"] = kvHeadCounts

	if ropeType := q.ropeScalingType(); ropeType != "" {
		kv["rope.scaling.type"] = ropeType
	}
	if v, ok := q.ropeScalingFactor(); ok {
		kv["rope.scaling.factor"] = v
	}
	if orig := q.ropeScalingOriginalContext(); orig != 0 {
		kv["rope.scaling.original_context_length"] = orig
	}

	if sections := q.RopeParameters.MropeSection; len(sections) > 0 {
		kv["mrope_sections"] = sections
	}

	kv["vision.block_count"] = cmp.Or(q.VisionModel.Depth, 27)
	kv["vision.embedding_length"] = cmp.Or(q.VisionModel.HiddenSize, 1152)
	kv["vision.attention.head_count"] = cmp.Or(q.VisionModel.NumHeads, 16)
	kv["vision.num_channels"] = cmp.Or(q.VisionModel.InChannels, 3)
	kv["vision.patch_size"] = cmp.Or(q.VisionModel.PatchSize, 16)
	kv["vision.spatial_merge_size"] = cmp.Or(q.VisionModel.SpatialMergeSize, 2)
	kv["vision.attention.layer_norm_epsilon"] = cmp.Or(q.VisionModel.RMSNormEps, 1e-6)
	kv["vision.rope.freq_base"] = cmp.Or(q.VisionModel.RopeTheta, 1e4)
	kv["vision.temporal_patch_size"] = cmp.Or(q.VisionModel.TemporalPatchSize, 2)

	kv["vision.shortest_edge"] = q.VisionModel.Size.ShortestEdge
	kv["vision.longest_edge"] = q.VisionModel.Size.LongestEdge

	kv["vision.image_mean"] = q.VisionModel.ImageMean
	kv["vision.image_std"] = q.VisionModel.ImageStd

	return kv
}

func (q *qwen35Model) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	var nonVision []Tensor
	for _, t := range ts {
		switch {
		case strings.Contains(t.Name(), "attn_qkv"):
			out = append(out, slices.Collect(splitDim(t, 0,
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_q")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_k")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_v")},
			))...)
		case strings.Contains(t.Name(), "patch_embed") && strings.HasSuffix(t.Name(), "weight"):
			shape := t.Shape()
			out = append(out, &ggml.Tensor{
				Name:     t.Name(),
				Kind:     t.Kind(),
				Shape:    append([]uint64{shape[0] * shape[1]}, shape[2:]...),
				WriterTo: t,
			})
		default:
			nonVision = append(nonVision, t)
		}
	}
	ts = nonVision

	remaining := ts
	if q.NumExperts > 0 {
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

		var merged []*ggml.Tensor
		merged, remaining = mergeTensors(ts, merges...)
		out = append(out, merged...)
	}

	type pendingBA struct {
		a Tensor
		b Tensor
	}
	pending := make(map[string]*pendingBA)
	hasDirectBA := make(map[string]struct{})

	addBA := func(base string, a, b Tensor) *ggml.Tensor {
		if a.Kind() != b.Kind() {
			panic(fmt.Sprintf("qwen35: ssm_ba kinds do not match for %s: %d vs %d", base, a.Kind(), b.Kind()))
		}
		if len(a.Shape()) != len(b.Shape()) {
			panic(fmt.Sprintf("qwen35: ssm_ba shape rank mismatch for %s: %v vs %v", base, a.Shape(), b.Shape()))
		}
		newShape := slices.Clone(a.Shape())
		newShape[0] += b.Shape()[0]
		return &ggml.Tensor{
			Name:  base + ".ssm_ba.weight",
			Kind:  a.Kind(),
			Shape: newShape,
			WriterTo: concatWriter{
				first:  b,
				second: a,
			},
		}
	}

	for i := range remaining {
		t := remaining[i]
		name := t.Name()
		shape := t.Shape()

		if strings.HasSuffix(name, ".ssm_ba.weight") {
			base := strings.TrimSuffix(name, ".ssm_ba.weight")
			if p, ok := pending[base]; ok && (p.a != nil || p.b != nil) {
				panic(fmt.Sprintf("qwen35: found ssm_ba.weight with pending ssm_ba_a/b for %s", base))
			}
			hasDirectBA[base] = struct{}{}
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    slices.Clone(shape),
				WriterTo: t,
			})
			continue
		}

		if strings.HasSuffix(name, ".ssm_ba_a.weight") || strings.HasSuffix(name, ".ssm_ba_b.weight") {
			var base string
			isA := strings.HasSuffix(name, ".ssm_ba_a.weight")
			if isA {
				base = strings.TrimSuffix(name, ".ssm_ba_a.weight")
			} else {
				base = strings.TrimSuffix(name, ".ssm_ba_b.weight")
			}
			if _, ok := hasDirectBA[base]; ok {
				panic(fmt.Sprintf("qwen35: found ssm_ba_a/b for %s but direct ssm_ba.weight already present", base))
			}

			pair := pending[base]
			if pair == nil {
				pair = &pendingBA{}
				pending[base] = pair
			}
			if isA {
				if pair.a != nil {
					panic(fmt.Sprintf("qwen35: duplicate ssm_ba_a for %s", base))
				}
				pair.a = t
			} else {
				if pair.b != nil {
					panic(fmt.Sprintf("qwen35: duplicate ssm_ba_b for %s", base))
				}
				pair.b = t
			}

			if pair.a != nil && pair.b != nil {
				out = append(out, addBA(base, pair.a, pair.b))
				delete(pending, base)
			}
			continue
		}

		if strings.HasSuffix(name, ".ssm_in.weight") {
			if qkv, gate, ok := q.splitQKVZTensor(t); ok {
				out = append(out, qkv, gate)
				continue
			}
			panic(fmt.Sprintf("qwen35: failed to split %s into attn_qkv/attn_gate (shape=%v)", name, shape))
		}

		switch {
		case strings.HasSuffix(name, "_norm.weight") && !strings.HasSuffix(name, ".ssm_norm.weight"):
			t.SetRepacker(q.addOne)
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    slices.Clone(shape),
				WriterTo: t,
			})
		case strings.HasSuffix(name, ".ssm_a"):
			t.SetRepacker(func(_ string, data []float32, shape []uint64) ([]float32, error) {
				result := make([]float32, len(data))
				for i, v := range data {
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
		case strings.HasSuffix(name, ".ssm_conv1d.weight"):
			newShape := slices.Clone(shape)
			if len(shape) == 3 {
				if shape[0] == 1 {
					newShape = []uint64{shape[1], shape[2]}
				} else if shape[1] == 1 {
					newShape = []uint64{shape[0], shape[2]}
				}
			}
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    newShape,
				WriterTo: t,
			})
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

	if len(pending) > 0 {
		var bases []string
		for base := range pending {
			bases = append(bases, base)
		}
		slices.Sort(bases)
		panic(fmt.Sprintf("qwen35: missing ssm_ba pair(s) for %v", bases))
	}

	return out
}

type concatWriter struct {
	first  Tensor
	second Tensor
}

func (w concatWriter) WriteTo(out io.Writer) (int64, error) {
	n1, err := w.first.WriteTo(out)
	if err != nil {
		return n1, err
	}
	n2, err := w.second.WriteTo(out)
	return n1 + n2, err
}

type qwen35QKVZSplitSpec struct {
	hidden    int
	headKDim  int
	headVDim  int
	numKHeads int
	numVHeads int
	qkvzDim   int
	qkvOut    int
	gateOut   int
}

func (q *qwen35Model) qkvzSpec(shape []uint64) (qwen35QKVZSplitSpec, bool) {
	if len(shape) != 2 {
		return qwen35QKVZSplitSpec{}, false
	}

	numKHeads := int(q.LinearNumKeyHeads)
	numVHeads := int(q.LinearNumValueHeads)
	headKDim := int(q.LinearKeyHeadDim)
	headVDim := int(q.LinearValueHeadDim)
	if numKHeads == 0 || numVHeads == 0 || headKDim == 0 || headVDim == 0 {
		return qwen35QKVZSplitSpec{}, false
	}
	if numVHeads%numKHeads != 0 {
		return qwen35QKVZSplitSpec{}, false
	}

	hidden := int(shape[1])
	vPerHead := headVDim * (numVHeads / numKHeads)
	qkvzDim := 2*headKDim + 2*vPerHead
	expectedOut := qkvzDim * numKHeads
	if int(shape[0]) != expectedOut {
		return qwen35QKVZSplitSpec{}, false
	}

	return qwen35QKVZSplitSpec{
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

func (q *qwen35Model) splitQKVZTensor(t Tensor) (*ggml.Tensor, *ggml.Tensor, bool) {
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

func (q *qwen35Model) repackQKVZ(spec qwen35QKVZSplitSpec, extractGate bool) Repacker {
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

func (*qwen35Model) addOne(_ string, data []float32, shape []uint64) ([]float32, error) {
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

func (q *qwen35Model) Replacements() []string {
	r := []string{
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",

		"input_layernorm", "attn_norm",
		"post_attention_layernorm", "post_attention_norm",

		"self_attn.q_proj", "attn_q",
		"self_attn.q_norm", "attn_q_norm",
		"self_attn.k_proj", "attn_k",
		"self_attn.k_norm", "attn_k_norm",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",

		"linear_attn.in_proj_qkv", "attn_qkv",
		"linear_attn.in_proj_z", "attn_gate",
		"linear_attn.in_proj_a", "ssm_ba_a",
		"linear_attn.in_proj_b", "ssm_ba_b",
		"linear_attn.in_proj_qkvz", "ssm_in",
		"linear_attn.in_proj_ba", "ssm_ba",
		"linear_attn.conv1d", "ssm_conv1d",
		"linear_attn.dt_bias", "ssm_dt",
		"linear_attn.dt_proj", "ssm_dt",
		"linear_attn.A_log", "ssm_a",
		"linear_attn.norm", "ssm_norm",
		"linear_attn.out_proj", "ssm_out",

		"mlp.gate.weight", "ffn_gate_inp.weight",
		"mlp.shared_expert.down_proj", "ffn_down_shexp",
		"mlp.shared_expert.gate_proj", "ffn_gate_shexp",
		"mlp.shared_expert.up_proj", "ffn_up_shexp",
		"mlp.shared_expert_gate", "ffn_gate_inp_shexp",

		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
	}

	r = append(r,
		"model.language_", "",
		"model.visual", "v",
		"patch_embed.proj", "patch_embed",
		"blocks", "blk",
		"attn.qkv", "attn_qkv",
		"attn.proj", "attn_out",
	)

	return r
}

