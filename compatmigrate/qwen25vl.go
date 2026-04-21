package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen25VLMigrator struct{}

func (qwen25VLMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "qwen25vl"
}

func (qwen25VLMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if strings.HasPrefix(tensor.name, "v.") || strings.HasPrefix(tensor.name, "mm.") {
			name := qwen25VLProjectorTensorName(tensor.name)
			switch {
			case strings.Contains(name, "patch_embd_0.weight"):
				projectorTensors = append(projectorTensors, copyTensor(strings.Replace(name, "patch_embd_0.weight", "patch_embd.weight", 1), tensor))
			case strings.Contains(name, "patch_embd_1.weight"):
				projectorTensors = append(projectorTensors, copyTensor(strings.Replace(name, "patch_embd_1.weight", "patch_embd.weight.1", 1), tensor))
			case strings.Contains(name, "attn_qkv"):
				split, err := splitSourceTensorDim(tensor, 0,
					strings.Replace(name, "attn_qkv", "attn_q", 1),
					strings.Replace(name, "attn_qkv", "attn_k", 1),
					strings.Replace(name, "attn_qkv", "attn_v", 1),
				)
				if err != nil {
					return nil, err
				}
				projectorTensors = append(projectorTensors, split...)
			default:
				projectorTensors = append(projectorTensors, copyTensor(name, tensor))
			}
			continue
		}
		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case key == "general.architecture":
			modelKV[key] = "qwen2vl"
		case strings.HasPrefix(key, "qwen25vl."):
			modelKV[strings.Replace(key, "qwen25vl.", "qwen2vl.", 1)] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}
	if sections := qwen25VLMropeSections(src); len(sections) > 0 {
		modelKV["rope.dimension_sections"] = sections
	}

	hidden := uint32(src.GGUF.KeyValue("embedding_length").Uint())
	visionHidden := uint32(src.GGUF.KeyValue("vision.embedding_length").Uint())
	nWAPattern := uint32(8)
	if fullatt := src.GGUF.KeyValue("vision.fullatt_block_indexes"); fullatt.Valid() {
		indexes := fullatt.Ints()
		if len(indexes) > 0 && indexes[0] >= 0 {
			nWAPattern = uint32(indexes[0] + 1)
		}
	}
	patchSize := uint32(src.GGUF.KeyValue("vision.patch_size").Uint())
	projectorKV := ggml.KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "qwen2.5vl_merger",
		"clip.has_vision_encoder":                  true,
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             visionHidden,
		"clip.vision.feed_forward_length":          visionHidden * 4,
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.attention.layer_norm_epsilon": float32(src.GGUF.KeyValue("vision.attention.layer_norm_epsilon").Float()),
		"clip.vision.num_channels":                 uint32(src.GGUF.KeyValue("vision.num_channels").Uint()),
		"clip.vision.patch_size":                   patchSize,
		"clip.vision.spatial_merge_size":           uint32(src.GGUF.KeyValue("vision.spatial_merge_size").Uint()),
		"clip.vision.image_size":                   uint32(560),
		"clip.vision.projection_dim":               hidden,
		"clip.vision.temporal_patch_size":          uint32(src.GGUF.KeyValue("vision.temporal_patch_size").Uint()),
		"clip.vision.n_wa_pattern":                 nWAPattern,
		"clip.use_silu":                            true,
		"clip.vision.rope.freq_base":               float32(src.GGUF.KeyValue("vision.rope.freq_base").Float()),
	}
	if fullatt := src.GGUF.KeyValue("vision.fullatt_block_indexes"); fullatt.Valid() {
		projectorKV["clip.vision.fullatt_block_indexes"] = normalizeGGUFValue(rawGGUFValue(fullatt.Value))
	} else {
		projectorKV["clip.vision.fullatt_block_indexes"] = []int32{7, 15, 23, 31}
	}
	if minPixels := src.GGUF.KeyValue("vision.min_pixels"); minPixels.Valid() {
		projectorKV["clip.vision.min_pixels"] = uint32(minPixels.Uint())
	}
	if maxPixels := src.GGUF.KeyValue("vision.max_pixels"); maxPixels.Valid() {
		projectorKV["clip.vision.max_pixels"] = uint32(maxPixels.Uint())
	}
	if mean := src.GGUF.KeyValue("vision.image_mean"); mean.Valid() {
		projectorKV["clip.vision.image_mean"] = normalizeGGUFValue(rawGGUFValue(mean.Value))
	}
	if std := src.GGUF.KeyValue("vision.image_std"); std.Valid() {
		projectorKV["clip.vision.image_std"] = normalizeGGUFValue(rawGGUFValue(std.Value))
	}
	if _, ok := projectorKV["clip.vision.image_mean"]; !ok {
		// Legacy installed GGUFs do not retain the preprocessor normalization.
		// Use the standard CLIP normalization as a best-effort local migration
		// fallback; an explicit pull can still refresh to the fully recreated
		// artifact if users want exact parity.
		projectorKV["clip.vision.image_mean"] = []float32{0.48145466, 0.4578275, 0.40821073}
		projectorKV["clip.vision.image_std"] = []float32{0.26862954, 0.26130258, 0.27577711}
	}

	return &Result{
		ModelKV:          modelKV,
		ModelTensors:     modelTensors,
		ProjectorKV:      projectorKV,
		ProjectorTensors: projectorTensors,
	}, nil
}

func qwen25VLProjectorTensorName(name string) string {
	name = strings.Replace(name, "v.merger.mlp.0", "mm.0", 1)
	name = strings.Replace(name, "v.merger.mlp.2", "mm.2", 1)
	name = strings.Replace(name, "v.merger.ln_q", "v.post_ln", 1)
	return name
}

func qwen25VLMropeSections(src *SourceModel) []int32 {
	for _, key := range []string{"rope.dimension_sections", "rope.mrope_section", "qwen25vl.rope.mrope_section", "qwen2vl.rope.mrope_section"} {
		if keyValue := src.GGUF.KeyValue(key); keyValue.Valid() {
			ints := keyValue.Ints()
			if len(ints) > 0 {
				out := make([]int32, 4)
				for i, v := range ints[:min(len(ints), len(out))] {
					out[i] = int32(v)
				}
				return out
			}
		}
	}
	return nil
}
