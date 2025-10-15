package convert

import (
	"cmp"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen25VLModel struct {
	qwen2Model

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
	} `json:"vision_config"`
}

var _ ModelConverter = (*qwen25VLModel)(nil)

func (q *qwen25VLModel) KV(t *Tokenizer) ggml.KV {
	kv := q.qwen2Model.KV(t)
	kv["general.architecture"] = "qwen25vl"

	if q.VisionModel.FullAttentionBlocks == nil {
		kv["vision.fullatt_block_indexes"] = []int32{7, 15, 23, 31}
	}

	kv["vision.block_count"] = cmp.Or(q.VisionModel.Depth, 32)
	kv["vision.embedding_length"] = q.VisionModel.HiddenSize
	kv["vision.attention.head_count"] = cmp.Or(q.VisionModel.NumHeads, 16)
	kv["vision.num_channels"] = q.VisionModel.InChannels
	kv["vision.patch_size"] = cmp.Or(q.VisionModel.PatchSize, 14)
	kv["vision.spatial_merge_size"] = cmp.Or(q.VisionModel.SpatialMergeSize, 2)
	kv["vision.spatial_patch_size"] = q.VisionModel.SpatialPatchSize
	kv["vision.window_size"] = cmp.Or(q.VisionModel.WindowSize, 112)
	kv["vision.attention.layer_norm_epsilon"] = cmp.Or(q.VisionModel.RMSNormEps, 1e-6)
	kv["vision.rope.freq_base"] = cmp.Or(q.VisionModel.RopeTheta, 1e4)
	kv["vision.fullatt_block_indexes"] = q.VisionModel.FullAttentionBlocks
	kv["vision.temporal_patch_size"] = cmp.Or(q.VisionModel.TemporalPatchSize, 2)

	return kv
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
