package convert

import (
	"cmp"
	"encoding/json"
	"io/fs"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen25VLModel struct {
	qwen2Model

	Preprocessor struct {
		ImageMean []float32 `json:"image_mean"`
		ImageStd  []float32 `json:"image_std"`
		MinPixels uint32    `json:"min_pixels"`
		MaxPixels uint32    `json:"max_pixels"`
	} `json:"-"`

	VisionModel struct {
		Depth               uint32  `json:"depth"`
		HiddenSize          uint32  `json:"hidden_size"`
		NumHeads            uint32  `json:"num_heads"`
		InChannels          uint32  `json:"in_chans"`
		PatchSize           uint32  `json:"patch_size"`
		SpatialMergeSize    uint32  `json:"spatial_merge_size"`
		SpatialPatchSize    uint32  `json:"spatial_patch_size"`
		WindowSize          uint32  `json:"window_size"`
		RMSNormEps          float32 `json:"layer_norm_epsilon"`
		RopeTheta           float32 `json:"rope_theta"`
		FullAttentionBlocks []int32 `json:"fullatt_block_indexes"`
		TemporalPatchSize   uint32  `json:"temporal_patch_size"`
		IntermediateSize    uint32  `json:"intermediate_size"`
		ImageSize           uint32  `json:"image_size"`
	} `json:"vision_config"`
}

var _ MultimodalConverter = (*qwen25VLModel)(nil)

func (q *qwen25VLModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "preprocessor_config.json")
	if err != nil {
		return err
	}
	return json.Unmarshal(bts, &q.Preprocessor)
}

func (q *qwen25VLModel) KV(t *Tokenizer) KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen2vl"

	for k, v := range q.qwen2Model.KV(t) {
		if strings.HasPrefix(k, "qwen2.") {
			kv[strings.Replace(k, "qwen2.", "qwen2vl.", 1)] = v
		}
	}

	// rope.dimension_sections — required by llama-server for M-RoPE
	if len(q.RopeScaling.MropeSection) > 0 {
		sections := make([]int32, 4)
		copy(sections, q.RopeScaling.MropeSection)
		kv["rope.dimension_sections"] = sections
	}

	return kv
}

// ProjectorKV returns KV metadata for the qwen2.5vl vision projector.
func (q *qwen25VLModel) ProjectorKV(t *Tokenizer) KV {
	kv := KV{
		"general.architecture":    "clip",
		"clip.projector_type":     "qwen2.5vl_merger",
		"clip.has_vision_encoder": true,

		"clip.vision.block_count":                  cmp.Or(q.VisionModel.Depth, 32),
		"clip.vision.embedding_length":             q.VisionModel.HiddenSize,
		"clip.vision.feed_forward_length":          cmp.Or(q.VisionModel.IntermediateSize, q.VisionModel.HiddenSize*4),
		"clip.vision.attention.head_count":         cmp.Or(q.VisionModel.NumHeads, 16),
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(q.VisionModel.RMSNormEps, 1e-6),
		"clip.vision.num_channels":                 q.VisionModel.InChannels,
		"clip.vision.patch_size":                   cmp.Or(q.VisionModel.PatchSize, 14),
		"clip.vision.spatial_merge_size":           cmp.Or(q.VisionModel.SpatialMergeSize, 2),
		"clip.vision.image_size":                   cmp.Or(q.VisionModel.ImageSize, 560),
		"clip.vision.projection_dim":               q.HiddenSize, // text model hidden_size
		"clip.vision.temporal_patch_size":          cmp.Or(q.VisionModel.TemporalPatchSize, 2),
		"clip.vision.n_wa_pattern":                 cmp.Or(q.VisionModel.WindowSize, 112) / cmp.Or(q.VisionModel.PatchSize, 14),
		"clip.use_silu":                            true,
		"clip.vision.fullatt_block_indexes":        q.VisionModel.FullAttentionBlocks,
		"clip.vision.rope.freq_base":               cmp.Or(q.VisionModel.RopeTheta, 1e4),
	}

	if q.VisionModel.FullAttentionBlocks == nil {
		kv["clip.vision.fullatt_block_indexes"] = []int32{7, 15, 23, 31}
	}

	if len(q.Preprocessor.ImageMean) == 3 {
		kv["clip.vision.image_mean"] = q.Preprocessor.ImageMean
	}
	if len(q.Preprocessor.ImageStd) == 3 {
		kv["clip.vision.image_std"] = q.Preprocessor.ImageStd
	}
	if q.Preprocessor.MinPixels > 0 {
		kv["clip.vision.min_pixels"] = q.Preprocessor.MinPixels
	}
	if q.Preprocessor.MaxPixels > 0 {
		kv["clip.vision.max_pixels"] = q.Preprocessor.MaxPixels
	}

	return kv
}

func isQwen25VLVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

// TextTensors returns only text model tensors (no vision/merger).
func (q *qwen25VLModel) TextTensors(ts []Tensor, t *Tokenizer) []*ggml.Tensor {
	var textOnly []Tensor
	for _, tensor := range ts {
		if !isQwen25VLVisionTensor(tensor.Name()) {
			textOnly = append(textOnly, tensor)
		}
	}
	return q.qwen2Model.Tensors(textOnly)
}

// ProjectorTensors returns only vision/merger tensors.
func (q *qwen25VLModel) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		if !isQwen25VLVisionTensor(t.Name()) {
			continue
		}

		name := t.Name()

		// Split patch_embed.proj along temporal dimension into two 4D tensors
		// First: v.patch_embd.weight, Second: v.patch_embd.weight.1
		if strings.Contains(name, "patch_embed.proj") {
			idx := 0
			for t := range splitDim(t, 2,
				split{Replacer: strings.NewReplacer("patch_embed.proj", "patch_embd")},
				split{Replacer: strings.NewReplacer("patch_embed.proj", "patch_embd")},
			) {
				t.Shape = slices.DeleteFunc(t.Shape, func(i uint64) bool { return i == 1 })
				if idx == 1 {
					// Second temporal slice: append .1 before extension
					// v.patch_embd.weight → v.patch_embd.weight.1
					t.Name = t.Name + ".1"
				}
				out = append(out, t)
				idx++
			}
			continue
		}

		// Split fused qkv into separate q, k, v
		if strings.Contains(name, "attn.qkv") {
			out = append(out, slices.Collect(splitDim(t, 0,
				split{Replacer: strings.NewReplacer("attn.qkv", "attn_q")},
				split{Replacer: strings.NewReplacer("attn.qkv", "attn_k")},
				split{Replacer: strings.NewReplacer("attn.qkv", "attn_v")},
			))...)
			continue
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

func (q *qwen25VLModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		if strings.Contains(t.Name(), "patch_embed.proj") {
			for t := range splitDim(t, 2,
				split{Replacer: strings.NewReplacer("patch_embed.proj", "patch_embd_0")},
				split{Replacer: strings.NewReplacer("patch_embed.proj", "patch_embd_1")},
			) {
				t.Shape = slices.DeleteFunc(t.Shape, func(i uint64) bool { return i == 1 })
				out = append(out, t)
			}
		} else if strings.Contains(t.Name(), "attn.qkv") {
			out = append(out, slices.Collect(splitDim(t, 0,
				split{Replacer: strings.NewReplacer("attn.qkv", "attn_q")},
				split{Replacer: strings.NewReplacer("attn.qkv", "attn_k")},
				split{Replacer: strings.NewReplacer("attn.qkv", "attn_v")},
			))...)
		} else {
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

func (p *qwen25VLModel) Replacements() []string {
	return append(
		p.qwen2Model.Replacements(),
		// Merger (multimodal projector) — must come before "visual" → "v" to match full path
		"visual.merger.mlp", "mm",
		"visual.merger.ln_q", "v.post_ln",
		// Vision encoder
		"visual", "v",
		"blocks", "blk",
		"attn.proj", "attn_out",
		"norm1", "ln1",
		"norm2", "ln2",
	)
}
