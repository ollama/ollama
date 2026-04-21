package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type deepseekOCRMigrator struct{}

func (deepseekOCRMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "deepseekocr"
}

func (deepseekOCRMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if deepseekOCRVisionTensor(tensor.name) {
			projectorTensors = append(projectorTensors, copyTensor(deepseekOCRProjectorTensorName(tensor.name), tensor))
			continue
		}
		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	modelKV := deepseekOCRModelKV(src)
	if _, ok := modelKV["deepseek2-ocr.expert_feed_forward_length"]; !ok {
		if inferred := deepseekOCRExpertFeedForwardLength(tensors); inferred != 0 {
			modelKV["deepseek2-ocr.expert_feed_forward_length"] = inferred
		}
	}
	if _, ok := modelKV["deepseek2-ocr.expert_shared_count"]; !ok {
		if inferred := deepseekOCRSharedExpertCount(tensors); inferred != 0 {
			modelKV["deepseek2-ocr.expert_shared_count"] = inferred
		}
	}

	return &Result{
		ModelKV:          modelKV,
		ModelTensors:     modelTensors,
		ProjectorKV:      deepseekOCRProjectorKV(src),
		ProjectorTensors: projectorTensors,
	}, nil
}

func deepseekOCRVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.") || strings.HasPrefix(name, "s.")
}

func deepseekOCRModelKV(src *SourceModel) ggml.KV {
	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case key == "general.architecture":
			modelKV[key] = "deepseek2-ocr"
		case strings.HasPrefix(key, "deepseekocr.vision."), strings.HasPrefix(key, "deepseek2-ocr.vision."):
			continue
		case strings.HasPrefix(key, "deepseekocr."):
			modelKV[strings.Replace(key, "deepseekocr.", "deepseek2-ocr.", 1)] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "deepseek2-ocr."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case deepseekOCRTextKVKey(key):
			modelKV["deepseek2-ocr."+key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}
	if _, ok := modelKV["deepseek2-ocr.attention.layer_norm_rms_epsilon"]; !ok {
		modelKV["deepseek2-ocr.attention.layer_norm_rms_epsilon"] = float32(1e-6)
	}
	return modelKV
}

func deepseekOCRExpertFeedForwardLength(tensors []*sourceTensor) uint32 {
	for _, tensor := range tensors {
		if !strings.Contains(tensor.name, "ffn_down_exps.weight") || len(tensor.shape) == 0 {
			continue
		}
		return uint32(tensor.shape[0])
	}
	for _, tensor := range tensors {
		if !(strings.Contains(tensor.name, "ffn_gate_exps.weight") || strings.Contains(tensor.name, "ffn_up_exps.weight")) || len(tensor.shape) < 2 {
			continue
		}
		return uint32(tensor.shape[1])
	}
	return 0
}

func deepseekOCRSharedExpertCount(tensors []*sourceTensor) uint32 {
	expertFF := deepseekOCRExpertFeedForwardLength(tensors)
	if expertFF == 0 {
		return 0
	}
	for _, tensor := range tensors {
		if !(strings.Contains(tensor.name, "ffn_gate_shexp.weight") || strings.Contains(tensor.name, "ffn_up_shexp.weight") || strings.Contains(tensor.name, "ffn_down_shexp.weight")) || len(tensor.shape) < 2 {
			continue
		}
		sharedWidth := tensor.shape[0]
		if tensor.shape[1] > sharedWidth {
			sharedWidth = tensor.shape[1]
		}
		if sharedWidth%uint64(expertFF) == 0 {
			return uint32(sharedWidth / uint64(expertFF))
		}
	}
	return 0
}

func deepseekOCRTextKVKey(key string) bool {
	switch key {
	case "block_count",
		"context_length",
		"embedding_length",
		"feed_forward_length",
		"attention.head_count",
		"attention.head_count_kv",
		"attention.layer_norm_rms_epsilon",
		"expert_count",
		"expert_feed_forward_length",
		"expert_used_count",
		"leading_dense_block_count",
		"expert_shared_count",
		"expert_group_count",
		"expert_group_used_count",
		"rope.dimension_count",
		"vocab_size":
		return true
	default:
		return false
	}
}

func deepseekOCRProjectorKV(src *SourceModel) ggml.KV {
	return ggml.KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "deepseekocr",
		"clip.has_vision_encoder":                  true,
		"clip.use_gelu":                            true,
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             uint32(src.GGUF.KeyValue("vision.embedding_length").Uint()),
		"clip.vision.feed_forward_length":          uint32(64),
		"clip.vision.attention.head_count":         deepseekOCRUint(src, "vision.attention.head_count", "vision.head_count"),
		"clip.vision.attention.layer_norm_epsilon": float32(1e-6),
		"clip.vision.image_size":                   uint32(src.GGUF.KeyValue("vision.image_size").Uint()),
		"clip.vision.image_mean":                   []float32{0.5, 0.5, 0.5},
		"clip.vision.image_std":                    []float32{0.5, 0.5, 0.5},
		"clip.vision.patch_size":                   uint32(src.GGUF.KeyValue("vision.patch_size").Uint()),
		"clip.vision.projection_dim":               deepseekOCRUint(src, "vision.projection_dim", "embedding_length"),
		"clip.vision.projector.scale_factor":       uint32(1),
		"clip.vision.window_size":                  uint32(14),
		"clip.vision.sam.block_count":              deepseekOCRUint(src, "vision.sam.block_count", "sam.block_count"),
		"clip.vision.sam.embedding_length":         deepseekOCRUint(src, "vision.sam.embedding_length", "sam.embedding_length"),
		"clip.vision.sam.head_count":               deepseekOCRUint(src, "vision.sam.head_count", "sam.head_count"),
	}
}

func deepseekOCRUint(src *SourceModel, keys ...string) uint32 {
	for _, key := range keys {
		if value := src.GGUF.KeyValue(key); value.Valid() {
			return uint32(value.Uint())
		}
	}
	return 0
}

func deepseekOCRProjectorTensorName(name string) string {
	replacer := strings.NewReplacer(
		"self_attn.out_proj", "attn_out",
		"self_attn.qkv_proj", "attn_qkv",
		"layer_norm1", "ln1",
		"layer_norm2", "ln2",
		"mlp.fc1", "ffn_up",
		"mlp.fc2", "ffn_down",
		"pre_layrnorm", "pre_ln",
		"s.blk.", "v.sam.blk.",
		"s.patch_embd.", "v.sam.patch_embd.",
		"s.position_embd", "v.sam.pos_embd.weight",
		"s.neck.", "v.sam.neck.",
		"s.net_", "v.sam.net_",
		"attn.proj.", "attn.out.",
		"attn.rel_pos_h", "attn.pos_h.weight",
		"attn.rel_pos_w", "attn.pos_w.weight",
		".norm1.", ".pre_ln.",
		".norm2.", ".post_ln.",
		"mm.layers.", "mm.model.fc.",
		"mm.image_newline", "v.image_newline",
		"mm.view_separator", "v.view_separator",
	)
	return replacer.Replace(name)
}
