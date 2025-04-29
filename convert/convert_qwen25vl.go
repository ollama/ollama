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
		Depth            uint32  `json:"depth"`
		HiddenSize       uint32  `json:"hidden_size"`
		IntermediateSize uint32  `json:"intermediate_size"`
		InChannels       uint32  `json:"in_chans"`
		NumHeads         uint32  `json:"num_heads"`
		PatchSize        uint32  `json:"patch_size"`
		SpatialMergeSize uint32  `json:"spatial_merge_size"`
		SpatialPatchSize uint32  `json:"spatial_patch_size"`
		WindowSize       uint32  `json:"window_size"`
		RopeTheta        float32 `json:"rope_theta"`
	} `json:"vision_config"`
}

var _ ModelConverter = (*qwen25VLModel)(nil)

func (q *qwen25VLModel) KV(t *Tokenizer) ggml.KV {
	kv := q.ModelParameters.KV(t)
	kv["general.architecture"] = "qwen25vl"

	for k, v := range q.qwen2Model.KV(t) {
		if strings.HasPrefix(k, "qwen2.") {
			kv[strings.Replace(k, "qwen2.", "qwen25vl.", 1)] = v
		}
	}

	kv["qwen25vl.vision.block_count"] = q.VisionModel.Depth
	kv["qwen25vl.vision.embedding_length"] = q.VisionModel.HiddenSize
	kv["qwen25vl.vision.feed_forward_length"] = q.VisionModel.IntermediateSize
	kv["qwen25vl.vision.attention.head_count"] = q.VisionModel.NumHeads
	kv["qwen25vl.vision.num_channels"] = q.VisionModel.InChannels
	kv["qwen25vl.vision.patch_size"] = q.VisionModel.PatchSize
	kv["qwen25vl.vision.spatial_merge_size"] = q.VisionModel.SpatialMergeSize
	kv["qwen25vl.vision.spatial_patch_size"] = q.VisionModel.SpatialPatchSize
	kv["qwen25vl.vision.rope.freq_base"] = cmp.Or(q.VisionModel.RopeTheta, 1e5)

	return kv
}

func (q *qwen25VLModel) Tensors(ts []Tensor) []ggml.Tensor {
	var out []ggml.Tensor

	for _, t := range ts {
		if strings.Contains(t.Name(), "patch_embed.proj") {
			for t := range splitDim(t, 2,
				strings.NewReplacer("patch_embed.proj", "patch_embd_0"),
				strings.NewReplacer("patch_embed.proj", "patch_embd_1"),
			) {
				t.Shape = slices.DeleteFunc(t.Shape, func(i uint64) bool { return i == 1 })
				out = append(out, t)
			}
		} else if strings.Contains(t.Name(), "attn.qkv") {
			out = append(out, slices.Collect(splitDim(t, 0,
				strings.NewReplacer("attn.qkv", "attn_q"),
				strings.NewReplacer("attn.qkv", "attn_k"),
				strings.NewReplacer("attn.qkv", "attn_v"),
			))...)
		} else {
			out = append(out, ggml.Tensor{
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
