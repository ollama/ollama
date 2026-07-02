package compatmigrate

import (
	"math"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type llama4Migrator struct{}

func (llama4Migrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "llama4" {
		return false
	}
	return sourceTensorHasPrefix(src, "v.") || sourceTensorHasPrefix(src, "mm.")
}

func (llama4Migrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if llama4ProjectorTensor(tensor.name) {
			projectorTensors = append(projectorTensors, copyTensor(llama4ProjectorTensorName(tensor.name), tensor))
		} else {
			modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
		}
	}
	if len(projectorTensors) == 0 {
		return nil, errUnsupportedFamily
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case strings.HasPrefix(key, "llama4.vision."):
			continue
		case key == "general.architecture", strings.HasPrefix(key, "llama4."), strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}

	projectorKV := ggml.KV{
		"general.architecture":                     "clip",
		"clip.has_vision_encoder":                  true,
		"clip.projector_type":                      "llama4",
		"clip.vision.projector_type":               "llama4",
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             uint32(src.GGUF.KeyValue("vision.embedding_length").Uint()),
		"clip.vision.feed_forward_length":          uint32(src.GGUF.KeyValue("vision.feed_forward_length").Uint()),
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.attention.layer_norm_epsilon": float32(src.GGUF.KeyValue("vision.layer_norm_epsilon").Float()),
		"clip.vision.image_size":                   uint32(src.GGUF.KeyValue("vision.image_size").Uint()),
		"clip.vision.patch_size":                   uint32(src.GGUF.KeyValue("vision.patch_size").Uint()),
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("embedding_length").Uint()),
		"clip.vision.projector.scale_factor":       llama4ProjectorScale(src),
		"clip.vision.image_mean":                   []float32{0.5, 0.5, 0.5},
		"clip.vision.image_std":                    []float32{0.5, 0.5, 0.5},
		"clip.use_gelu":                            true,
	}

	return &Result{
		ModelKV:          modelKV,
		ModelTensors:     modelTensors,
		ProjectorKV:      projectorKV,
		ProjectorTensors: projectorTensors,
	}, nil
}

func llama4ProjectorTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func llama4ProjectorTensorName(name string) string {
	switch {
	case name == "v.class_embedding":
		return "v.class_embd"
	case name == "v.positional_embedding_vlm":
		return "v.position_embd.weight"
	case strings.HasPrefix(name, "v.patch_embedding."):
		return strings.Replace(name, "v.patch_embedding.", "v.patch_embd.", 1)
	case strings.HasPrefix(name, "v.layernorm_pre."):
		return strings.Replace(name, "v.layernorm_pre.", "v.pre_ln.", 1)
	case strings.HasPrefix(name, "v.layernorm_post."):
		return strings.Replace(name, "v.layernorm_post.", "v.post_ln.", 1)
	case strings.HasPrefix(name, "v.vision_adapter.mlp.fc1."):
		return strings.Replace(name, "v.vision_adapter.mlp.fc1.", "mm.model.mlp.1.", 1)
	case strings.HasPrefix(name, "v.vision_adapter.mlp.fc2."):
		return strings.Replace(name, "v.vision_adapter.mlp.fc2.", "mm.model.mlp.2.", 1)
	case strings.HasPrefix(name, "mm.linear_1."):
		return strings.Replace(name, "mm.linear_1.", "mm.model.fc.", 1)
	}

	name = strings.Replace(name, ".attn_output.", ".attn_out.", 1)
	name = strings.Replace(name, ".attn_norm.", ".ln1.", 1)
	name = strings.Replace(name, ".ffn_norm.", ".ln2.", 1)
	name = strings.Replace(name, ".mlp.fc1.", ".ffn_up.", 1)
	name = strings.Replace(name, ".mlp.fc2.", ".ffn_down.", 1)
	return name
}

func llama4ProjectorScale(src *SourceModel) uint32 {
	scale := uint32(2)
	ratio := float32(src.GGUF.KeyValue("vision.pixel_shuffle_ratio").Float())
	if ratio > 0 {
		scale = uint32(math.Round(float64(1 / ratio)))
	}
	if scale == 0 {
		return 2
	}
	return scale
}
