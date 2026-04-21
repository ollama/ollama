package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

const gemma3ChatTemplate = `{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}`

type gemma3Migrator struct{}

func (gemma3Migrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "gemma3" {
		return false
	}
	return src.GGUF.KeyValue("mm.tokens_per_image").Valid() ||
		sourceTensorHasPrefix(src, "v.") ||
		sourceTensorHasPrefix(src, "mm.") ||
		src.GGUF.KeyValue("rope.global.freq_base").Valid() ||
		src.GGUF.KeyValue("rope.local.freq_base").Valid() ||
		src.GGUF.KeyValue("tokenizer.ggml.add_padding_token").Valid() ||
		src.GGUF.KeyValue("tokenizer.ggml.add_unknown_token").Valid() ||
		!src.GGUF.KeyValue("attention.layer_norm_rms_epsilon").Valid()
}

func (gemma3Migrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	vocabSize := gemma3VocabSize(tensors)
	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if isGemma3ProjectorTensor(tensor.name) {
			projectorTensors = append(projectorTensors, copyTensor(gemma3ProjectorTensorName(tensor.name), tensor))
			continue
		}

		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	modelKV := gemma3ModelKV(src, vocabSize)

	result := &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
		// Gemma 3 uses the standard template/image-tag path; there is no
		// built-in gemma3 renderer or parser to select.
		ClearRenderer: true,
		ClearParser:   true,
	}

	if len(projectorTensors) > 0 {
		result.ProjectorKV = gemma3ProjectorKV(src)
		result.ProjectorTensors = projectorTensors
	}

	return result, nil
}

func gemma3ModelKV(src *SourceModel, vocabSize int) ggml.KV {
	modelKV := ggml.KV{}
	var globalRope any
	var localRope any
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		value := normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		switch {
		case key == "gemma3.rope.global.freq_base":
			globalRope = value
			modelKV["gemma3.rope.freq_base"] = value
		case key == "gemma3.rope.local.freq_base":
			localRope = value
			modelKV["gemma3.rope.freq_base_swa"] = value
		case strings.HasPrefix(key, "gemma3.vision."), strings.HasPrefix(key, "gemma3.mm."):
			continue
		case key == "tokenizer.ggml.tokens", key == "tokenizer.ggml.scores", key == "tokenizer.ggml.token_type":
			modelKV[key] = truncateGemma3TokenizerValue(value, vocabSize)
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."), strings.HasPrefix(key, "gemma3."):
			modelKV[key] = value
		}
	}

	if _, ok := modelKV["general.architecture"]; !ok {
		modelKV["general.architecture"] = "gemma3"
	}
	if _, ok := modelKV["gemma3.attention.layer_norm_rms_epsilon"]; !ok {
		modelKV["gemma3.attention.layer_norm_rms_epsilon"] = float32(1e-6)
	}
	if _, ok := modelKV["gemma3.rope.freq_base"]; !ok {
		if globalRope != nil {
			modelKV["gemma3.rope.freq_base"] = globalRope
		} else {
			modelKV["gemma3.rope.freq_base"] = float32(1000000)
		}
	}
	if _, ok := modelKV["gemma3.rope.freq_base_swa"]; !ok {
		if localRope != nil {
			modelKV["gemma3.rope.freq_base_swa"] = localRope
		} else {
			modelKV["gemma3.rope.freq_base_swa"] = float32(10000)
		}
	}
	if _, ok := modelKV["tokenizer.chat_template"]; !ok {
		modelKV["tokenizer.chat_template"] = gemma3ChatTemplate
	}
	if src.GGUF.KeyValue("context_length").Uint() >= 131072 {
		if _, ok := modelKV["gemma3.rope.scaling.type"]; !ok {
			modelKV["gemma3.rope.scaling.type"] = "linear"
		}
		if _, ok := modelKV["gemma3.rope.scaling.factor"]; !ok {
			modelKV["gemma3.rope.scaling.factor"] = float32(8)
		}
	}
	delete(modelKV, "gemma3.rope.local.freq_base")
	delete(modelKV, "tokenizer.ggml.add_padding_token")
	delete(modelKV, "tokenizer.ggml.add_unknown_token")

	return modelKV
}

func gemma3ProjectorKV(src *SourceModel) ggml.KV {
	imageSize := uint32(src.GGUF.KeyValue("vision.image_size").Uint())
	if imageSize == 0 {
		imageSize = 896
	}

	patchSize := uint32(src.GGUF.KeyValue("vision.patch_size").Uint())
	if patchSize == 0 {
		patchSize = 14
	}

	eps := float32(src.GGUF.KeyValue("vision.attention.layer_norm_epsilon").Float())
	if eps == 0 {
		eps = 1e-6
	}

	return ggml.KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "gemma3",
		"clip.has_vision_encoder":                  true,
		"clip.has_text_encoder":                    false,
		"clip.has_llava_projector":                 false,
		"clip.use_gelu":                            true,
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             uint32(src.GGUF.KeyValue("vision.embedding_length").Uint()),
		"clip.vision.feed_forward_length":          uint32(src.GGUF.KeyValue("vision.feed_forward_length").Uint()),
		"clip.vision.image_size":                   imageSize,
		"clip.vision.patch_size":                   patchSize,
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.attention.layer_norm_epsilon": eps,
		"clip.vision.image_mean":                   []float32{0.5, 0.5, 0.5},
		"clip.vision.image_std":                    []float32{0.5, 0.5, 0.5},
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("embedding_length").Uint()),
	}
}

func gemma3VocabSize(tensors []*sourceTensor) int {
	for _, tensor := range tensors {
		if tensor.name == "token_embd.weight" && len(tensor.shape) > 0 {
			return int(tensor.shape[len(tensor.shape)-1])
		}
	}
	return 0
}

func truncateGemma3TokenizerValue(v any, vocabSize int) any {
	if vocabSize <= 0 {
		return v
	}

	switch values := v.(type) {
	case []string:
		if len(values) > vocabSize {
			return append([]string(nil), values[:vocabSize]...)
		}
	case []float32:
		if len(values) > vocabSize {
			return append([]float32(nil), values[:vocabSize]...)
		}
	case []int32:
		if len(values) > vocabSize {
			return append([]int32(nil), values[:vocabSize]...)
		}
	}
	return v
}

func isGemma3ProjectorTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func gemma3ProjectorTensorName(name string) string {
	name = strings.Replace(name, "mm.mm_input_projection", "mm.input_projection", 1)
	name = strings.Replace(name, "mm.mm_soft_emb_norm", "mm.soft_emb_norm", 1)
	name = strings.Replace(name, "v.patch_embedding", "v.patch_embd", 1)
	name = strings.Replace(name, "v.position_embedding", "v.position_embd", 1)
	name = strings.Replace(name, "v.post_layernorm", "v.post_ln", 1)
	name = strings.Replace(name, "layer_norm1", "ln1", 1)
	name = strings.Replace(name, "layer_norm2", "ln2", 1)
	name = strings.Replace(name, "attn_output", "attn_out", 1)
	name = strings.Replace(name, "mlp.fc1", "ffn_down", 1)
	name = strings.Replace(name, "mlp.fc2", "ffn_up", 1)
	return name
}
