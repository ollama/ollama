package convert

import (
	"cmp"
	"encoding/json"
	"fmt"
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
	kv["general.architecture"] = arch

	// rope.dimension_sections — required by llama-server for M-RoPE
	// Pad mrope_section to 4 elements (upstream convention)
	if len(m.RopeScaling.MropeSection) > 0 {
		sections := make([]int32, 4)
		copy(sections, m.RopeScaling.MropeSection)
		kv["rope.dimension_sections"] = sections
	}

	// Number of deepstack layers (used by llama-server to compute n_embd_inp)
	kv["n_deepstack_layers"] = uint32(len(m.VisionModel.DeepstackVisualIndexes))

	return kv
}

// ProjectorKV returns KV metadata for the qwen3vl vision projector.
func (m *qwen3VLModel) ProjectorKV(t *Tokenizer) KV {
	kv := KV{
		"general.architecture":    "clip",
		"clip.projector_type":     "qwen3vl_merger",
		"clip.has_vision_encoder": true,

		"clip.vision.block_count":                  cmp.Or(m.VisionModel.Depth, 32),
		"clip.vision.embedding_length":             m.VisionModel.HiddenSize,
		"clip.vision.feed_forward_length":          cmp.Or(m.VisionModel.IntermediateSize, m.VisionModel.HiddenSize*4),
		"clip.vision.attention.head_count":         cmp.Or(m.VisionModel.NumHeads, 16),
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(m.VisionModel.RMSNormEps, 1e-6),
		"clip.vision.num_channels":                 m.VisionModel.InChannels,
		"clip.vision.patch_size":                   cmp.Or(m.VisionModel.PatchSize, 14),
		"clip.vision.spatial_merge_size":           cmp.Or(m.VisionModel.SpatialMergeSize, 2),
		"clip.vision.image_size":                   uint32(math.Sqrt(float64(cmp.Or(m.VisionModel.NumPositionEmbeddings, 2304)))) * cmp.Or(m.VisionModel.PatchSize, 16),
		"clip.vision.projection_dim":               cmp.Or(m.VisionModel.OutHiddenSize, m.HiddenSize),
		"clip.use_gelu":                            true,
		"clip.vision.temporal_patch_size":          cmp.Or(m.VisionModel.TemporalPatchSize, 2),
		"clip.vision.rope.freq_base":               cmp.Or(m.VisionModel.RopeTheta, 1e4),
		"clip.vision.deepstack_visual_indexes":     m.VisionModel.DeepstackVisualIndexes,
	}

	if m.VisionModel.Size.ShortestEdge > 0 {
		kv["clip.vision.min_pixels"] = m.VisionModel.Size.ShortestEdge
	}
	if m.VisionModel.Size.LongestEdge > 0 {
		kv["clip.vision.max_pixels"] = m.VisionModel.Size.LongestEdge
	}
	if len(m.VisionModel.ImageMean) == 3 {
		kv["clip.vision.image_mean"] = m.VisionModel.ImageMean
	}
	if len(m.VisionModel.ImageStd) == 3 {
		kv["clip.vision.image_std"] = m.VisionModel.ImageStd
	}

	return kv
}

func isQwen3VLVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

// TextTensors returns only text model tensors (no vision/merger).
func (m *qwen3VLModel) TextTensors(ts []Tensor, t *Tokenizer) []*ggml.Tensor {
	var textOnly []Tensor
	for _, tensor := range ts {
		if !isQwen3VLVisionTensor(tensor.Name()) {
			textOnly = append(textOnly, tensor)
		}
	}
	return m.qwen3Model.Tensors(textOnly)
}

// qwen3VLProjectorRename renames merger and deepstack tensors to match
// what llama-server's clip/mtmd system expects. For deepstack, the sequential
// index (0, 1, 2) from HF weights is remapped to the actual vision block index
// from DeepstackVisualIndexes — this is what clip.cpp expects (it loads
// v.deepstack.{block_idx}.* for each vision block il).
func (m *qwen3VLModel) qwen3VLProjectorRename(name string) string {
	// Merger: v.merger.linear_fc1 → mm.0, v.merger.linear_fc2 → mm.2
	if strings.HasPrefix(name, "v.merger.") {
		name = strings.Replace(name, "v.merger.linear_fc1", "mm.0", 1)
		name = strings.Replace(name, "v.merger.linear_fc2", "mm.2", 1)
		name = strings.Replace(name, "v.merger.norm", "v.post_ln", 1)
		return name
	}
	// Deepstack: v.deepstack.{seq}.linear_fc1 → v.deepstack.{block_idx}.fc1
	// The sequential index from HF tensors must be remapped to the actual
	// vision block index from deepstack_visual_indexes.
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

// ProjectorTensors returns only vision/merger tensors.
func (m *qwen3VLModel) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		if !isQwen3VLVisionTensor(t.Name()) {
			continue
		}

		name := m.qwen3VLProjectorRename(t.Name())

		// Split patch_embd along temporal dimension (5D → two 4D tensors)
		// First: v.patch_embd.weight, Second: v.patch_embd.weight.1
		if strings.Contains(name, "patch_embd") && strings.HasSuffix(name, "weight") {
			shape := t.Shape()
			if len(shape) == 5 && shape[2] == 2 {
				idx := 0
				for t := range splitDim(t, 2,
					split{Replacer: strings.NewReplacer("patch_embd", "patch_embd")},
					split{Replacer: strings.NewReplacer("patch_embd", "patch_embd")},
				) {
					t.Shape = slices.DeleteFunc(t.Shape, func(i uint64) bool { return i == 1 })
					if idx == 1 {
						t.Name = t.Name + ".1"
					}
					out = append(out, t)
					idx++
				}
				continue
			}
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
		// Vision MLP: strip mlp. prefix and rename linear_fc
		"mlp.linear_fc1", "ffn_up",
		"mlp.linear_fc2", "ffn_down",
		"deepstack_merger_list", "deepstack",
	)
}
