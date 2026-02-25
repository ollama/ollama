package convert

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"slices"
	"strings"

	"github.com/pdevine/tensor"
	"github.com/pdevine/tensor/native"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen3NextRopeScaling struct {
	Type         string     `json:"type"`
	Factor       ropeFactor `json:"factor"`
	MropeSection []int32    `json:"mrope_section"`
}

type qwen3NextRopeParams struct {
	MRopeInterleaved    bool    `json:"mrope_interleaved"`
	MropeSection        []int32 `json:"mrope_section"`
	RopeType            string  `json:"rope_type"`
	RopeTheta           float32 `json:"rope_theta"`
	PartialRotaryFactor float32 `json:"partial_rotary_factor"`
}

type qwen3NextTextConfig struct {
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
	NormTopkProb           *bool  `json:"norm_topk_prob"`
	MoEIntermediateSize    uint32 `json:"moe_intermediate_size"`
	SharedExpertIntermSize uint32 `json:"shared_expert_intermediate_size"`

	// Hybrid attention config
	FullAttentionInterval uint32   `json:"full_attention_interval"`
	LayerTypes            []string `json:"layer_types"`

	// Linear attention (Gated Delta Net) config
	LinearConvKernelDim uint32 `json:"linear_conv_kernel_dim"`
	LinearKeyHeadDim    uint32 `json:"linear_key_head_dim"`
	LinearNumKeyHeads   uint32 `json:"linear_num_key_heads"`
	LinearNumValueHeads uint32 `json:"linear_num_value_heads"`
	LinearValueHeadDim  uint32 `json:"linear_value_head_dim"`

	// RoPE config
	PartialRotaryFactor float32              `json:"partial_rotary_factor"`
	RopeScaling         qwen3NextRopeScaling `json:"rope_scaling"`
	RopeParameters      qwen3NextRopeParams  `json:"rope_parameters"`
}

type qwen3NextVisionConfig struct {
	Depth                  uint32  `json:"depth"`
	HiddenSize             uint32  `json:"hidden_size"`
	NumHeads               uint32  `json:"num_heads"`
	InChannels             uint32  `json:"in_channels"`
	PatchSize              uint32  `json:"patch_size"`
	SpatialMergeSize       uint32  `json:"spatial_merge_size"`
	RMSNormEps             float32 `json:"layer_norm_epsilon"`
	RopeTheta              float32 `json:"rope_theta"`
	TemporalPatchSize      uint32  `json:"temporal_patch_size"`
	DeepstackVisualIndexes []int32 `json:"deepstack_visual_indexes"`

	Size struct {
		ShortestEdge uint32 `json:"shortest_edge"`
		LongestEdge  uint32 `json:"longest_edge"`
	} `json:"size"`

	ImageMean []float32 `json:"image_mean"`
	ImageStd  []float32 `json:"image_std"`
}

type qwen3NextModel struct {
	ModelParameters
	qwen3NextTextConfig

	TextConfig  *qwen3NextTextConfig  `json:"text_config"`
	VisionModel qwen3NextVisionConfig `json:"vision_config"`

	ImageTokenID       uint32 `json:"image_token_id"`
	VisionStartTokenID uint32 `json:"vision_start_token_id"`
	VisionEndTokenID   uint32 `json:"vision_end_token_id"`
}

var _ ModelConverter = (*qwen3NextModel)(nil)

func (q *qwen3NextModel) parseMore(fsys fs.FS) error {
	if q.TextConfig != nil {
		q.qwen3NextTextConfig = *q.TextConfig
	}

	if q.RopeTheta == 0 {
		q.RopeTheta = q.RopeParameters.RopeTheta
	}
	if q.PartialRotaryFactor == 0 {
		q.PartialRotaryFactor = q.RopeParameters.PartialRotaryFactor
	}

	if q.RopeScaling.Type == "" && q.RopeParameters.RopeType != "" {
		q.RopeScaling.Type = q.RopeParameters.RopeType
	}

	// Pull vision preprocessing fields when present.
	if q.VisionModel.Depth > 0 {
		if bts, err := fs.ReadFile(fsys, "preprocessor_config.json"); err == nil {
			var pre struct {
				Size struct {
					ShortestEdge uint32 `json:"shortest_edge"`
					LongestEdge  uint32 `json:"longest_edge"`
				} `json:"size"`
				PatchSize         uint32    `json:"patch_size"`
				TemporalPatchSize uint32    `json:"temporal_patch_size"`
				MergeSize         uint32    `json:"merge_size"`
				ImageMean         []float32 `json:"image_mean"`
				ImageStd          []float32 `json:"image_std"`
			}
			if json.Unmarshal(bts, &pre) == nil {
				if q.VisionModel.PatchSize == 0 {
					q.VisionModel.PatchSize = pre.PatchSize
				}
				if q.VisionModel.TemporalPatchSize == 0 {
					q.VisionModel.TemporalPatchSize = pre.TemporalPatchSize
				}
				if q.VisionModel.SpatialMergeSize == 0 {
					q.VisionModel.SpatialMergeSize = pre.MergeSize
				}
				if q.VisionModel.Size.ShortestEdge == 0 {
					q.VisionModel.Size.ShortestEdge = pre.Size.ShortestEdge
				}
				if q.VisionModel.Size.LongestEdge == 0 {
					q.VisionModel.Size.LongestEdge = pre.Size.LongestEdge
				}
				if len(q.VisionModel.ImageMean) == 0 {
					q.VisionModel.ImageMean = pre.ImageMean
				}
				if len(q.VisionModel.ImageStd) == 0 {
					q.VisionModel.ImageStd = pre.ImageStd
				}
			}
		}
	}

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
	if _, err := q.kvHeadCounts(); err != nil {
		return err
	}

	return nil
}

func (q *qwen3NextModel) kvHeadCounts() ([]uint32, error) {
	if len(q.LayerTypes) > 0 {
		kv := make([]uint32, q.NumHiddenLayers)
		hasFull := false
		hasRecurrent := false
		for i := range q.NumHiddenLayers {
			layerType := ""
			if i < uint32(len(q.LayerTypes)) {
				layerType = q.LayerTypes[i]
			}
			if layerType == "full_attention" {
				kv[i] = q.NumKeyValueHeads
				hasFull = true
			} else {
				hasRecurrent = true
			}
		}
		if !hasFull || !hasRecurrent {
			return nil, fmt.Errorf("qwen3next: layer_types must include both full_attention and linear_attention")
		}
		return kv, nil
	}

	if q.FullAttentionInterval == 0 {
		return nil, fmt.Errorf("qwen3next: full_attention_interval must be set")
	}
	if q.FullAttentionInterval > q.NumHiddenLayers {
		return nil, fmt.Errorf("qwen3next: full_attention_interval (%d) exceeds num_hidden_layers (%d)", q.FullAttentionInterval, q.NumHiddenLayers)
	}

	kv := make([]uint32, q.NumHiddenLayers)
	hasFull := false
	for i := range q.NumHiddenLayers {
		if (i+1)%q.FullAttentionInterval == 0 {
			kv[i] = q.NumKeyValueHeads
			hasFull = true
		}
	}
	if !hasFull {
		return nil, fmt.Errorf("qwen3next: head_count_kv would be all zeros (full_attention_interval=%d, num_hidden_layers=%d)", q.FullAttentionInterval, q.NumHiddenLayers)
	}
	return kv, nil
}

func (q *qwen3NextModel) ropeSections() []int32 {
	if len(q.RopeParameters.MropeSection) > 0 {
		return q.RopeParameters.MropeSection
	}
	return q.RopeScaling.MropeSection
}

func (q *qwen3NextModel) shouldReorderVHeads() bool {
	modelType := strings.ToLower(q.ModelType)
	if strings.Contains(modelType, "qwen3_next") || strings.Contains(modelType, "qwen3next") {
		return false
	}

	for _, arch := range q.Architectures {
		arch = strings.ToLower(arch)
		if strings.Contains(arch, "qwen3next") || strings.Contains(arch, "qwen3_next") {
			return false
		}
	}

	// Default to qwen3.5 layout for all other qwen3next-family imports.
	return true
}

func (q *qwen3NextModel) KV(t *Tokenizer) KV {
	kv := q.ModelParameters.KV(t)

	arch := "qwen35"
	if q.NumExperts > 0 {
		arch = "qwen35moe"
	}
	kv["general.architecture"] = arch
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
	kv["rope.freq_base"] = q.RopeTheta

	partialRotary := q.PartialRotaryFactor
	if partialRotary > 0 && partialRotary <= 1 {
		kv["rope.dimension_count"] = uint32(float32(headDim) * partialRotary)
	}

	if sections := q.ropeSections(); len(sections) > 0 {
		kv["mrope_sections"] = sections
		kv["rope.mrope_section"] = sections
		kv["rope.dimension_sections"] = sections
	}
	if q.RopeParameters.MRopeInterleaved {
		kv["rope.mrope_interleaved"] = true
	}

	if q.RopeScaling.Type != "" && q.RopeScaling.Type != "default" {
		kv["rope.scaling.type"] = q.RopeScaling.Type
		kv["rope.scaling.factor"] = q.RopeScaling.Factor
	}

	if q.NumExperts > 0 {
		kv["expert_count"] = q.NumExperts
		kv["expert_used_count"] = q.NumExpertsPerToken
		if q.NormTopkProb != nil {
			kv["norm_top_k_prob"] = *q.NormTopkProb
		}
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
	if q.shouldReorderVHeads() {
		kv["ssm.v_head_reordered"] = true
	}
	if q.FullAttentionInterval > 0 {
		kv["full_attention_interval"] = q.FullAttentionInterval
	}

	if headCounts, err := q.kvHeadCounts(); err == nil {
		kv["attention.head_count_kv"] = headCounts
	}

	if q.VisionModel.Depth > 0 {
		kv["vision.block_count"] = q.VisionModel.Depth
		kv["vision.embedding_length"] = q.VisionModel.HiddenSize
		kv["vision.attention.head_count"] = q.VisionModel.NumHeads
		kv["vision.num_channels"] = q.VisionModel.InChannels
		if q.VisionModel.PatchSize > 0 {
			kv["vision.patch_size"] = q.VisionModel.PatchSize
		}
		if q.VisionModel.SpatialMergeSize > 0 {
			kv["vision.spatial_merge_size"] = q.VisionModel.SpatialMergeSize
		}
		if q.VisionModel.RMSNormEps > 0 {
			kv["vision.attention.layer_norm_epsilon"] = q.VisionModel.RMSNormEps
		}
		if q.VisionModel.RopeTheta > 0 {
			kv["vision.rope.freq_base"] = q.VisionModel.RopeTheta
		}
		if q.VisionModel.TemporalPatchSize > 0 {
			kv["vision.temporal_patch_size"] = q.VisionModel.TemporalPatchSize
		}
		kv["vision.deepstack_visual_indexes"] = q.VisionModel.DeepstackVisualIndexes
		if q.VisionModel.Size.ShortestEdge > 0 {
			kv["vision.shortest_edge"] = q.VisionModel.Size.ShortestEdge
		}
		if q.VisionModel.Size.LongestEdge > 0 {
			kv["vision.longest_edge"] = q.VisionModel.Size.LongestEdge
		}
		if len(q.VisionModel.ImageMean) > 0 {
			kv["vision.image_mean"] = q.VisionModel.ImageMean
		}
		if len(q.VisionModel.ImageStd) > 0 {
			kv["vision.image_std"] = q.VisionModel.ImageStd
		}
	}

	if q.ImageTokenID > 0 {
		kv["image_token_id"] = q.ImageTokenID
	}
	if q.VisionStartTokenID > 0 {
		kv["vision_start_token_id"] = q.VisionStartTokenID
	}
	if q.VisionEndTokenID > 0 {
		kv["vision_end_token_id"] = q.VisionEndTokenID
	}

	return kv
}

func (q *qwen3NextModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

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

	merged, remaining := mergeTensors(ts, merges...)
	out = append(out, merged...)

	for _, t := range remaining {
		name := t.Name()
		shape := t.Shape()

		if strings.HasSuffix(name, ".ssm_in.weight") {
			if qkv, gate, ok := q.splitQKVZTensor(t); ok {
				out = append(out, qkv, gate)
				continue
			}
			panic(fmt.Sprintf("qwen3next: failed to split %s into attn_qkv/attn_gate (shape=%v)", name, shape))
		}

		switch {
		case strings.Contains(name, ".mlp.experts.gate_up_proj"):
			out = append(out, slices.Collect(splitDim(t, 1,
				split{Replacer: strings.NewReplacer(".mlp.experts.gate_up_proj", ".ffn_gate_exps.weight")},
				split{Replacer: strings.NewReplacer(".mlp.experts.gate_up_proj", ".ffn_up_exps.weight")},
			))...)

		case strings.Contains(name, ".mlp.experts.down_proj"):
			out = append(out, &ggml.Tensor{
				Name:     strings.NewReplacer(".mlp.experts.down_proj", ".ffn_down_exps.weight").Replace(name),
				Kind:     t.Kind(),
				Shape:    slices.Clone(shape),
				WriterTo: t,
			})

		case strings.HasPrefix(name, "v.blk.") && strings.Contains(name, ".attn_qkv"):
			out = append(out, slices.Collect(splitDim(t, 0,
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_q")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_k")},
				split{Replacer: strings.NewReplacer("attn_qkv", "attn_v")},
			))...)

		case strings.Contains(name, "patch_embed") && strings.HasSuffix(name, "weight"):
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    append([]uint64{shape[0] * shape[1]}, shape[2:]...),
				WriterTo: t,
			})

		case strings.HasSuffix(name, "_norm.weight") && !strings.HasSuffix(name, ".ssm_norm.weight"):
			t.SetRepacker(q.addOne)
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})

		case strings.HasSuffix(name, ".ssm_a"):
			t.SetRepacker(q.repackSSMA())
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})

		case strings.HasSuffix(name, ".attn_qkv.weight"):
			if q.shouldReorderVHeads() {
				t.SetRepacker(q.repackAttnQKV())
			}
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})

		case strings.HasSuffix(name, ".attn_gate.weight"):
			if q.shouldReorderVHeads() {
				// HF tensor layout is [out_features, in_features]; reorder rows.
				t.SetRepacker(q.repackReorderDim(0, int(q.LinearValueHeadDim)))
			}
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})

		case strings.HasSuffix(name, ".ssm_beta.weight"), strings.HasSuffix(name, ".ssm_alpha.weight"):
			if q.shouldReorderVHeads() {
				// HF tensor layout is [out_features, in_features]; reorder rows.
				t.SetRepacker(q.repackReorderDim(0, 1))
			}
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})

		case strings.HasSuffix(name, ".ssm_dt"):
			if q.shouldReorderVHeads() {
				t.SetRepacker(q.repackReorderDim(0, 1))
			}
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})

		case strings.HasSuffix(name, ".ssm_out.weight"):
			if q.shouldReorderVHeads() {
				// HF out_proj layout is [out_features, in_features]; reorder columns.
				t.SetRepacker(q.repackReorderDim(1, int(q.LinearValueHeadDim)))
			}
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})

		case strings.HasSuffix(name, ".ssm_conv1d.weight"):
			newShape := slices.Clone(shape)
			if len(shape) == 3 {
				if shape[0] == 1 {
					newShape = []uint64{shape[1], shape[2]}
				} else if shape[1] == 1 {
					newShape = []uint64{shape[0], shape[2]}
				}
			}
			if q.shouldReorderVHeads() {
				t.SetRepacker(q.repackConv1D())
			}
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: newShape, WriterTo: t})

		default:
			out = append(out, &ggml.Tensor{Name: name, Kind: t.Kind(), Shape: slices.Clone(shape), WriterTo: t})
		}
	}

	return out
}

func (q *qwen3NextModel) repackReorderDim(dim, headDim int) Repacker {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		if !q.shouldReorderVHeads() {
			return data, nil
		}
		numK := int(q.LinearNumKeyHeads)
		numVPerK := int(q.LinearNumValueHeads / q.LinearNumKeyHeads)
		return reorderHeadLayout(data, shape, dim, numK, numVPerK, headDim)
	}
}

func (q *qwen3NextModel) repackAttnQKV() Repacker {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		if !q.shouldReorderVHeads() || len(shape) != 2 {
			return data, nil
		}

		rows := int(shape[0])
		cols := int(shape[1])
		numK := int(q.LinearNumKeyHeads)
		numV := int(q.LinearNumValueHeads)
		headK := int(q.LinearKeyHeadDim)
		headV := int(q.LinearValueHeadDim)
		qDim := headK * numK
		kDim := headK * numK
		vDim := headV * numV
		qkvDim := qDim + kDim + vDim

		switch {
		case rows == qkvDim:
			// HF layout: [out_features, in_features]. Keep Q/K rows unchanged and
			// reorder only V rows from grouped -> tiled head layout.
			out := make([]float32, len(data))
			qkRows := qDim + kDim
			qkSize := qkRows * cols
			copy(out[:qkSize], data[:qkSize])

			vStart := qkSize
			vEnd := vStart + vDim*cols
			reorderedV, err := reorderHeadLayout(data[vStart:vEnd], []uint64{uint64(vDim), uint64(cols)}, 0, numK, numV/numK, headV)
			if err != nil {
				return nil, err
			}
			copy(out[vStart:vEnd], reorderedV)
			copy(out[vEnd:], data[vEnd:])
			return out, nil

		case cols == qkvDim:
			// Fallback for already-transposed [in_features, out_features] tensors.
			out := make([]float32, len(data))
			copy(out, data)
			for r := range rows {
				base := r * cols
				vStart := base + qDim + kDim
				vEnd := vStart + vDim
				reorderedV, err := reorderHeadLayout(out[vStart:vEnd], []uint64{uint64(vDim)}, 0, numK, numV/numK, headV)
				if err != nil {
					return nil, err
				}
				copy(out[vStart:vEnd], reorderedV)
			}
			return out, nil

		default:
			return data, nil
		}
	}
}

func (q *qwen3NextModel) repackConv1D() Repacker {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		if !q.shouldReorderVHeads() {
			return data, nil
		}

		normShape := slices.Clone(shape)
		if len(shape) == 3 {
			if shape[0] == 1 {
				normShape = []uint64{shape[1], shape[2]}
			} else if shape[1] == 1 {
				normShape = []uint64{shape[0], shape[2]}
			}
		}
		if len(normShape) != 2 {
			return data, nil
		}

		rows := int(normShape[0])
		cols := int(normShape[1])
		numK := int(q.LinearNumKeyHeads)
		numV := int(q.LinearNumValueHeads)
		headK := int(q.LinearKeyHeadDim)
		headV := int(q.LinearValueHeadDim)
		qkChannels := 2 * headK * numK
		totalChannels := qkChannels + headV*numV
		if qkChannels <= 0 {
			return data, nil
		}

		switch {
		case rows == totalChannels:
			// HF layout after squeeze: [channels, kernel]
			out := make([]float32, len(data))
			prefix := qkChannels * cols
			copy(out[:prefix], data[:prefix])
			reorderedV, err := reorderHeadLayout(data[prefix:], []uint64{uint64(totalChannels - qkChannels), uint64(cols)}, 0, numK, numV/numK, headV)
			if err != nil {
				return nil, err
			}
			copy(out[prefix:], reorderedV)
			return out, nil
		case cols == totalChannels:
			// Fallback for transposed [kernel, channels]
			out := make([]float32, len(data))
			copy(out, data)
			vChannels := totalChannels - qkChannels
			for r := range rows {
				base := r * cols
				vStart := base + qkChannels
				vEnd := vStart + vChannels
				reorderedV, err := reorderHeadLayout(out[vStart:vEnd], []uint64{uint64(vChannels)}, 0, numK, numV/numK, headV)
				if err != nil {
					return nil, err
				}
				copy(out[vStart:vEnd], reorderedV)
			}
			return out, nil
		default:
			return data, nil
		}
	}
}

func (q *qwen3NextModel) repackSSMA() Repacker {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		result := make([]float32, len(data))
		for i, v := range data {
			result[i] = -float32(math.Exp(float64(v)))
		}
		if !q.shouldReorderVHeads() {
			return result, nil
		}
		numK := int(q.LinearNumKeyHeads)
		numVPerK := int(q.LinearNumValueHeads / q.LinearNumKeyHeads)
		return reorderHeadLayout(result, shape, 0, numK, numVPerK, 1)
	}
}

func reorderHeadLayout(data []float32, shape []uint64, dim int, numKHeads, numVPerK, headDim int) ([]float32, error) {
	if len(shape) == 0 || numKHeads <= 0 || numVPerK <= 0 || headDim <= 0 {
		return data, nil
	}

	dims := make([]int, len(shape))
	for i := range shape {
		dims[i] = int(shape[i])
	}
	if dim < 0 {
		dim += len(dims)
	}
	if dim < 0 || dim >= len(dims) {
		return data, nil
	}

	expected := numKHeads * numVPerK * headDim
	if dims[dim] != expected {
		return data, nil
	}

	newShape := make([]int, 0, len(dims)+2)
	newShape = append(newShape, dims[:dim]...)
	newShape = append(newShape, numKHeads, numVPerK, headDim)
	newShape = append(newShape, dims[dim+1:]...)

	var tt tensor.Tensor = tensor.New(tensor.WithShape(dims...), tensor.WithBacking(data))
	if err := tt.Reshape(newShape...); err != nil {
		return nil, err
	}

	perm := make([]int, len(newShape))
	for i := range perm {
		perm[i] = i
	}
	perm[dim], perm[dim+1] = perm[dim+1], perm[dim]

	tt, err := tensor.Transpose(tt, perm...)
	if err != nil {
		return nil, err
	}
	tt = tensor.Materialize(tt)

	total := 1
	for _, d := range dims {
		total *= d
	}
	if err := tt.Reshape(total); err != nil {
		return nil, err
	}
	return native.VectorF32(tt.(*tensor.Dense))
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
		"model.language_model.embed_tokens", "token_embd",
		"model.language_model.norm", "output_norm",
		"model.language_model.layers", "blk",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",

		// Vision
		"model.visual", "v",
		"patch_embed.proj", "patch_embed",
		"blocks", "blk",
		"attn.qkv", "attn_qkv",
		"attn.proj", "attn_out",
		"deepstack_merger_list", "deepstack_merger",

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

		// Linear attention (legacy qwen3next)
		"linear_attn.in_proj_qkvz", "ssm_in",
		"linear_attn.in_proj_ba", "ssm_ba",

		// Linear attention (qwen35)
		"linear_attn.in_proj_qkv", "attn_qkv",
		"linear_attn.in_proj_z", "attn_gate",
		"linear_attn.in_proj_a", "ssm_alpha",
		"linear_attn.in_proj_b", "ssm_beta",

		"linear_attn.conv1d", "ssm_conv1d",
		"linear_attn.dt_bias", "ssm_dt",
		"linear_attn.dt_proj", "ssm_dt",
		"linear_attn.A_log", "ssm_a",
		"linear_attn.norm", "ssm_norm",
		"linear_attn.out_proj", "ssm_out",

		// MoE
		"mlp.gate.weight", "ffn_gate_inp.weight",
		"mlp.shared_expert.down_proj", "ffn_down_shexp",
		"mlp.shared_expert.gate_proj", "ffn_gate_shexp",
		"mlp.shared_expert.up_proj", "ffn_up_shexp",
		"mlp.shared_expert_gate", "ffn_gate_inp_shexp",

		// Dense FFN
		"mlp.down_proj", "ffn_down",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
	}
}
