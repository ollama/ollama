package convert

import (
	"cmp"
	"encoding/json"
	"errors"
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
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}

	return json.Unmarshal(bts, &q.Preprocessor)
}

func (q *qwen25VLModel) KV(t *Tokenizer) KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen25vl"

	for k, v := range q.qwen2Model.KV(t) {
		if strings.HasPrefix(k, "qwen2.") {
			kv[strings.Replace(k, "qwen2.", "qwen25vl.", 1)] = v
		}
	}

	kv["qwen25vl.vision.block_count"] = cmp.Or(q.VisionModel.Depth, 32)
	kv["qwen25vl.vision.embedding_length"] = q.VisionModel.HiddenSize
	kv["qwen25vl.vision.attention.head_count"] = cmp.Or(q.VisionModel.NumHeads, 16)
	kv["qwen25vl.vision.num_channels"] = q.VisionModel.InChannels
	kv["qwen25vl.vision.patch_size"] = cmp.Or(q.VisionModel.PatchSize, 14)
	kv["qwen25vl.vision.spatial_merge_size"] = cmp.Or(q.VisionModel.SpatialMergeSize, 2)
	kv["qwen25vl.vision.spatial_patch_size"] = q.VisionModel.SpatialPatchSize
	kv["qwen25vl.vision.window_size"] = cmp.Or(q.VisionModel.WindowSize, 112)
	kv["qwen25vl.vision.attention.layer_norm_epsilon"] = cmp.Or(q.VisionModel.RMSNormEps, 1e-6)
	kv["qwen25vl.vision.rope.freq_base"] = cmp.Or(q.VisionModel.RopeTheta, 1e4)
	kv["qwen25vl.vision.fullatt_block_indexes"] = q.fullAttentionBlocks()
	kv["qwen25vl.vision.temporal_patch_size"] = cmp.Or(q.VisionModel.TemporalPatchSize, 2)

	return kv
}

func (q *qwen25VLModel) TextKV(t *Tokenizer) KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen2vl"

	for k, v := range q.qwen2Model.KV(t) {
		if !strings.HasPrefix(k, "qwen2.") || k == "qwen2.rope.mrope_section" {
			continue
		}
		kv[strings.Replace(k, "qwen2.", "qwen2vl.", 1)] = v
	}

	if sections := q.RopeScaling.MropeSection; len(sections) > 0 {
		dimensionSections := append([]int32(nil), sections...)
		for len(dimensionSections) < 4 {
			dimensionSections = append(dimensionSections, 0)
		}
		kv["qwen2vl.rope.dimension_sections"] = dimensionSections[:4]
	}

	return kv
}

func (q *qwen25VLModel) ProjectorKV(*Tokenizer) KV {
	kv := KV{
		"general.architecture":                     "clip",
		"general.type":                             "mmproj",
		"general.file_type":                        uint32(1),
		"general.quantization_version":             uint32(2),
		"clip.has_vision_encoder":                  true,
		"clip.projector_type":                      "qwen2.5vl_merger",
		"clip.use_silu":                            true,
		"clip.vision.block_count":                  cmp.Or(q.VisionModel.Depth, 32),
		"clip.vision.embedding_length":             q.VisionModel.HiddenSize,
		"clip.vision.feed_forward_length":          cmp.Or(q.VisionModel.IntermediateSize, q.VisionModel.HiddenSize*4),
		"clip.vision.attention.head_count":         cmp.Or(q.VisionModel.NumHeads, 16),
		"clip.vision.attention.layer_norm_epsilon": cmp.Or(q.VisionModel.RMSNormEps, 1e-6),
		"clip.vision.num_channels":                 q.VisionModel.InChannels,
		"clip.vision.patch_size":                   cmp.Or(q.VisionModel.PatchSize, 14),
		"clip.vision.spatial_merge_size":           cmp.Or(q.VisionModel.SpatialMergeSize, 2),
		"clip.vision.window_size":                  cmp.Or(q.VisionModel.WindowSize, 112),
		"clip.vision.image_size":                   cmp.Or(q.VisionModel.ImageSize, 560),
		"clip.vision.projection_dim":               q.HiddenSize,
		"clip.vision.temporal_patch_size":          cmp.Or(q.VisionModel.TemporalPatchSize, 2),
		"clip.vision.rope.freq_base":               cmp.Or(q.VisionModel.RopeTheta, 1e4),
	}

	if blocks := q.fullAttentionBlocks(); len(blocks) > 0 {
		kv["clip.vision.n_wa_pattern"] = uint32(blocks[0] + 1)
	}
	if len(q.Preprocessor.ImageMean) == 3 {
		kv["clip.vision.image_mean"] = q.Preprocessor.ImageMean
	} else {
		kv["clip.vision.image_mean"] = []float32{0.48145466, 0.4578275, 0.40821073}
	}
	if len(q.Preprocessor.ImageStd) == 3 {
		kv["clip.vision.image_std"] = q.Preprocessor.ImageStd
	} else {
		kv["clip.vision.image_std"] = []float32{0.26862954, 0.26130258, 0.27577711}
	}
	if q.Preprocessor.MinPixels > 0 {
		kv["clip.vision.image_min_pixels"] = q.Preprocessor.MinPixels
	}
	if q.Preprocessor.MaxPixels > 0 {
		kv["clip.vision.image_max_pixels"] = q.Preprocessor.MaxPixels
	}

	return kv
}

func (q *qwen25VLModel) fullAttentionBlocks() []int32 {
	if len(q.VisionModel.FullAttentionBlocks) > 0 {
		return q.VisionModel.FullAttentionBlocks
	}
	return []int32{7, 15, 23, 31}
}

func qwen25VLVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func (q *qwen25VLModel) TextTensors(ts []Tensor, _ *Tokenizer) []*ggml.Tensor {
	var textOnly []Tensor
	for _, t := range ts {
		if !qwen25VLVisionTensor(t.Name()) {
			textOnly = append(textOnly, t)
		}
	}

	return q.qwen2Model.Tensors(textOnly)
}

func (q *qwen25VLModel) ProjectorTensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		if !qwen25VLVisionTensor(t.Name()) {
			continue
		}

		name := q.qwen25VLProjectorTensorName(t.Name())
		if strings.Contains(name, "patch_embed.proj") {
			out = append(out, q.qwen25VLPatchEmbedTensors(t)...)
			continue
		}
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
			Shape:    slices.Clone(t.Shape()),
			WriterTo: t,
		})
	}

	return out
}

func (q *qwen25VLModel) qwen25VLProjectorTensorName(name string) string {
	name = strings.Replace(name, "v.merger.ln_q", "v.post_ln", 1)
	name = strings.Replace(name, "v.merger.mlp.0", "mm.0", 1)
	name = strings.Replace(name, "v.merger.mlp.2", "mm.2", 1)
	return name
}

func (q *qwen25VLModel) qwen25VLPatchEmbedTensors(t Tensor) []*ggml.Tensor {
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
		"visual", "v",
		"blocks", "blk",
		"attn.proj", "attn_out",
		"norm1", "ln1",
		"norm2", "ln2",
	)
}
