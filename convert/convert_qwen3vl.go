package convert

import (
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen3VLModel struct {
	qwen3Model `json:"text_config"`

	VisionModel struct {
		Depth                  uint32  `json:"depth"`
		HiddenSize             uint32  `json:"hidden_size"`
		NumHeads               uint32  `json:"num_heads"`
		InChannels             uint32  `json:"in_channels"`
		PatchSize              uint32  `json:"patch_size"`
		SpatialMergeSize       uint32  `json:"spatial_merge_size"`
		WindowSize             uint32  `json:"window_size"`
		RMSNormEps             float32 `json:"layer_norm_epsilon"`
		RopeTheta              float32 `json:"rope_theta"`
		TemporalPatchSize      uint32  `json:"temporal_patch_size"`
		DeepstackVisualIndexes []int32 `json:"deepstack_visual_indexes"`
		IntermediateSize       uint32  `json:"intermediate_size"`
		OutHiddenSize          uint32  `json:"out_hidden_size"`
		NumPositionEmbeddings  uint32  `json:"num_position_embeddings"`

		Size struct {
			ShortestEdge uint32 `json:"shortest_edge"`
			LongestEdge  uint32 `json:"longest_edge"`
		} `json:"size"`

		ImageMean []float32 `json:"image_mean"`
		ImageStd  []float32 `json:"image_std"`
	} `json:"vision_config"`
}

var _ MultimodalConverter = (*qwen3VLModel)(nil)

func (m *qwen3VLModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "preprocessor_config.json")
	if err != nil {
		return err
	}

	return json.Unmarshal(bts, &m.VisionModel)
}

func (m *qwen3VLModel) KV(t *Tokenizer) KV {
	kv := m.qwen3Model.KV(t)

	arch := "qwen3vl"
	if m.NumExperts > 0 {
		arch += "moe"
	}
	// override architecture
	kv["general.architecture"] = arch

	if sections := m.RopeScaling.MropeSection; len(sections) > 0 {
		dimensionSections := append([]int32(nil), sections...)
		if len(dimensionSections) == 3 {
			dimensionSections = append(dimensionSections, 0)
		}
		kv["rope.dimension_sections"] = dimensionSections
	}
	kv["n_deepstack_layers"] = uint32(len(m.VisionModel.DeepstackVisualIndexes))

	kv["vision.block_count"] = cmp.Or(m.VisionModel.Depth, 32)
	kv["vision.embedding_length"] = m.VisionModel.HiddenSize
	if m.VisionModel.IntermediateSize > 0 {
		kv["vision.feed_forward_length"] = m.VisionModel.IntermediateSize
	}
	kv["vision.attention.head_count"] = cmp.Or(m.VisionModel.NumHeads, 16)
	kv["vision.num_channels"] = m.VisionModel.InChannels
	kv["vision.patch_size"] = cmp.Or(m.VisionModel.PatchSize, 14)
	kv["vision.spatial_merge_size"] = cmp.Or(m.VisionModel.SpatialMergeSize, 2)
	kv["vision.attention.layer_norm_epsilon"] = cmp.Or(m.VisionModel.RMSNormEps, 1e-6)
	kv["vision.rope.freq_base"] = cmp.Or(m.VisionModel.RopeTheta, 1e4)
	kv["vision.temporal_patch_size"] = cmp.Or(m.VisionModel.TemporalPatchSize, 2)
	kv["vision.deepstack_visual_indexes"] = m.VisionModel.DeepstackVisualIndexes

	kv["vision.shortest_edge"] = m.VisionModel.Size.ShortestEdge
	kv["vision.longest_edge"] = m.VisionModel.Size.LongestEdge

	kv["vision.image_mean"] = m.VisionModel.ImageMean
	kv["vision.image_std"] = m.VisionModel.ImageStd

	return kv
}

func (m *qwen3VLModel) TextKV(t *Tokenizer) KV {
	kv := m.KV(t)
	for _, key := range []string{
		"vision.block_count",
		"vision.embedding_length",
		"vision.feed_forward_length",
		"vision.attention.head_count",
		"vision.num_channels",
		"vision.patch_size",
		"vision.spatial_merge_size",
		"vision.attention.layer_norm_epsilon",
		"vision.rope.freq_base",
		"vision.temporal_patch_size",
		"vision.deepstack_visual_indexes",
		"vision.shortest_edge",
		"vision.longest_edge",
		"vision.image_mean",
		"vision.image_std",
		"rope.mrope_section",
	} {
		delete(kv, key)
	}

	return kv
}

func (m *qwen3VLModel) ProjectorKV(*Tokenizer) KV {
	depth := cmp.Or(m.VisionModel.Depth, uint32(32))
	deepstack := make([]bool, depth)
	for _, idx := range m.VisionModel.DeepstackVisualIndexes {
		if idx >= 0 && uint32(idx) < depth {
			deepstack[idx] = true
		}
	}

	projectionDim := m.VisionModel.OutHiddenSize
	if projectionDim == 0 {
		projectionDim = m.HiddenSize
	}
	layerNormEps := m.VisionModel.RMSNormEps
	if layerNormEps == 0 {
		layerNormEps = 1e-6
	}

	kv := KV{
		"general.architecture":                     "clip",
		"general.type":                             "mmproj",
		"general.file_type":                        uint32(1),
		"general.quantization_version":             uint32(2),
		"clip.has_vision_encoder":                  true,
		"clip.projector_type":                      "qwen3vl_merger",
		"clip.use_gelu":                            true,
		"clip.vision.block_count":                  depth,
		"clip.vision.embedding_length":             m.VisionModel.HiddenSize,
		"clip.vision.feed_forward_length":          cmp.Or(m.VisionModel.IntermediateSize, m.VisionModel.HiddenSize*4),
		"clip.vision.attention.head_count":         cmp.Or(m.VisionModel.NumHeads, uint32(16)),
		"clip.vision.attention.layer_norm_epsilon": layerNormEps,
		"clip.vision.num_channels":                 m.VisionModel.InChannels,
		"clip.vision.patch_size":                   cmp.Or(m.VisionModel.PatchSize, uint32(14)),
		"clip.vision.spatial_merge_size":           cmp.Or(m.VisionModel.SpatialMergeSize, uint32(2)),
		"clip.vision.image_size":                   m.projectorImageSize(),
		"clip.vision.projection_dim":               projectionDim,
		"clip.vision.temporal_patch_size":          cmp.Or(m.VisionModel.TemporalPatchSize, uint32(2)),
		"clip.vision.rope.freq_base":               cmp.Or(m.VisionModel.RopeTheta, float32(1e4)),
		"clip.vision.is_deepstack_layers":          deepstack,
	}
	if m.VisionModel.Size.ShortestEdge > 0 {
		kv["clip.vision.image_min_pixels"] = m.VisionModel.Size.ShortestEdge
	}
	if m.VisionModel.Size.LongestEdge > 0 {
		kv["clip.vision.image_max_pixels"] = m.VisionModel.Size.LongestEdge
	}
	if len(m.VisionModel.ImageMean) == 3 {
		kv["clip.vision.image_mean"] = m.VisionModel.ImageMean
	}
	if len(m.VisionModel.ImageStd) == 3 {
		kv["clip.vision.image_std"] = m.VisionModel.ImageStd
	}

	return kv
}

func (m *qwen3VLModel) projectorImageSize() uint32 {
	if m.VisionModel.NumPositionEmbeddings > 0 && m.VisionModel.PatchSize > 0 {
		root := uint32(math.Sqrt(float64(m.VisionModel.NumPositionEmbeddings)))
		if root*root == m.VisionModel.NumPositionEmbeddings {
			return root * m.VisionModel.PatchSize
		}
	}
	return uint32(768)
}

func qwen3VLVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func (m *qwen3VLModel) TextTensors(ts []Tensor, _ *Tokenizer) []*ggml.Tensor {
	var textOnly []Tensor
	for _, t := range ts {
		if qwen3VLVisionTensor(t.Name()) {
			continue
		}
		textOnly = append(textOnly, t)
	}

	return m.qwen3Model.Tensors(textOnly)
}

func (m *qwen3VLModel) qwen3VLProjectorRename(name string) string {
	if strings.HasPrefix(name, "v.merger.") {
		name = strings.Replace(name, "v.merger.linear_fc1", "mm.0", 1)
		name = strings.Replace(name, "v.merger.linear_fc2", "mm.2", 1)
		name = strings.Replace(name, "v.merger.norm", "v.post_ln", 1)
		return name
	}

	if strings.HasPrefix(name, "v.deepstack.") {
		re := regexp.MustCompile(`^v\.deepstack\.(\d+)\.(.+)$`)
		if matches := re.FindStringSubmatch(name); matches != nil {
			seqIdx, err := strconv.Atoi(matches[1])
			if err == nil && seqIdx < len(m.VisionModel.DeepstackVisualIndexes) {
				blockIdx := m.VisionModel.DeepstackVisualIndexes[seqIdx]
				suffix := matches[2]
				suffix = strings.Replace(suffix, "linear_fc1", "fc1", 1)
				suffix = strings.Replace(suffix, "linear_fc2", "fc2", 1)
				return fmt.Sprintf("v.deepstack.%d.%s", blockIdx, suffix)
			}
		}
	}

	return name
}

func (m *qwen3VLModel) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		if !qwen3VLVisionTensor(t.Name()) {
			continue
		}

		name := m.qwen3VLProjectorRename(t.Name())
		if name == "v.patch_embd.weight" {
			out = append(out, m.qwen3VLPatchEmbedTensors(t)...)
			continue
		}

		kind := t.Kind()
		var writer io.WriterTo = t
		if name == "v.position_embd.weight" {
			kind = tensorKindFP32
			writer = tensorFloat32Writer{tensor: t}
		} else if sourceDType(t) == "BF16" && kind == tensorKindFP16 {
			kind = tensorKindBF16
			writer = tensorBF16Writer{tensor: t}
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     kind,
			Shape:    slices.Clone(t.Shape()),
			WriterTo: writer,
		})
	}

	return out
}

func (m *qwen3VLModel) qwen3VLPatchEmbedTensors(t Tensor) []*ggml.Tensor {
	shape := t.Shape()
	if len(shape) != 5 || shape[2] != 2 {
		return nil
	}

	outShape := []uint64{shape[0], shape[1], shape[3], shape[4]}
	return []*ggml.Tensor{
		{
			Name:     "v.patch_embd.weight",
			Kind:     tensorKindFP32,
			Shape:    slices.Clone(outShape),
			WriterTo: tensorFloat32Writer{tensor: t, repacker: qwenTemporalPatchEmbedSlice(0)},
		},
		{
			Name:     "v.patch_embd.weight.1",
			Kind:     tensorKindFP32,
			Shape:    slices.Clone(outShape),
			WriterTo: tensorFloat32Writer{tensor: t, repacker: qwenTemporalPatchEmbedSlice(1)},
		},
	}
}

func qwenTemporalPatchEmbedSlice(slice int) Repacker {
	return func(_ string, data []float32, shape []uint64) ([]float32, error) {
		if len(shape) != 5 || shape[2] != 2 {
			return nil, fmt.Errorf("qwen temporal patch embedding shape %v", shape)
		}

		outChannels := int(shape[0])
		inChannels := int(shape[1])
		frames := int(shape[2])
		height := int(shape[3])
		width := int(shape[4])
		if slice < 0 || slice >= frames {
			return nil, fmt.Errorf("qwen temporal patch embedding slice %d out of range", slice)
		}

		expected := outChannels * inChannels * frames * height * width
		if len(data) != expected {
			return nil, fmt.Errorf("qwen temporal patch embedding data size %d, expected %d", len(data), expected)
		}

		out := make([]float32, outChannels*inChannels*height*width)
		for oc := range outChannels {
			for ic := range inChannels {
				for y := range height {
					for x := range width {
						src := ((((oc*inChannels+ic)*frames+slice)*height + y) * width) + x
						dst := (((oc*inChannels+ic)*height + y) * width) + x
						out[dst] = data[src]
					}
				}
			}
		}

		return out, nil
	}
}

func (m *qwen3VLModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var rest []Tensor
	var out []*ggml.Tensor
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
			rest = append(rest, t)
		}
	}

	return append(m.qwen3Model.Tensors(rest), out...)
}

func (m *qwen3VLModel) Replacements() []string {
	return append(
		m.qwen3Model.Replacements(),
		"model.language_", "",
		"model.visual", "v",
		"patch_embed.proj", "patch_embd",
		"pos_embed", "position_embd",
		"blocks", "blk",
		"attn.qkv", "attn_qkv",
		"attn.proj", "attn_out",
		"norm1", "ln1",
		"norm2", "ln2",
		"mlp.linear_fc1", "ffn_up",
		"mlp.linear_fc2", "ffn_down",
		"deepstack_merger_list", "deepstack",
	)
}
