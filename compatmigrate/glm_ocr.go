package compatmigrate

import (
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
)

type glmOCRMigrator struct{}

func (glmOCRMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "glmocr"
}

func (glmOCRMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	numLayers := glmOCRTextLayerCount(src)
	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	pendingGate := map[string]*sourceTensor{}
	pendingUp := map[string]*sourceTensor{}
	for _, tensor := range tensors {
		if glmOCRVisionTensor(tensor.name) {
			name := glmOCRProjectorTensorName(tensor.name)
			if strings.HasSuffix(name, "patch_embd.weight") && len(tensor.shape) == 5 && tensor.shape[2] == 2 {
				split, err := splitSourceTensorDim(tensor, 2, "v.patch_embd.weight", "v.patch_embd.weight.1")
				if err != nil {
					return nil, err
				}
				for _, t := range split {
					t.Shape = []uint64{tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4]}
					projectorTensors = append(projectorTensors, t)
				}
				continue
			}
			projectorTensors = append(projectorTensors, copyTensor(name, tensor))
			continue
		}

		name := glmOCRTextTensorName(tensor.name, numLayers)
		name = strings.Replace(name, ".attn_out.", ".attn_output.", 1)
		name = strings.Replace(name, ".post_attn_norm.", ".post_attention_norm.", 1)
		name = strings.Replace(name, ".post_ffn_norm.", ".post_ffw_norm.", 1)
		if strings.Contains(name, ".ffn_gate.") {
			fusedName := strings.Replace(name, ".ffn_gate.", ".ffn_up.", 1)
			if up := pendingUp[fusedName]; up != nil {
				merged, err := concatSourceTensorsSlowDim(1, fusedName, tensor, up)
				if err != nil {
					return nil, err
				}
				modelTensors = append(modelTensors, merged)
				delete(pendingUp, fusedName)
				continue
			}
			pendingGate[fusedName] = tensor
			continue
		}
		if strings.Contains(name, ".ffn_up.") {
			if gate := pendingGate[name]; gate != nil {
				merged, err := concatSourceTensorsSlowDim(1, name, gate, tensor)
				if err != nil {
					return nil, err
				}
				modelTensors = append(modelTensors, merged)
				delete(pendingGate, name)
				continue
			}
			pendingUp[name] = tensor
			continue
		}
		modelTensors = append(modelTensors, copyTensor(name, tensor))
	}
	for name, tensor := range pendingUp {
		modelTensors = append(modelTensors, copyTensor(name, tensor))
	}
	for _, tensor := range pendingGate {
		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	return &Result{
		ModelKV:          glmOCRModelKV(src),
		ModelTensors:     modelTensors,
		ProjectorKV:      glmOCRProjectorKV(src),
		ProjectorTensors: projectorTensors,
	}, nil
}

func glmOCRVisionTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")
}

func glmOCRModelKV(src *SourceModel) ggml.KV {
	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case key == "general.architecture":
			modelKV[key] = "glm4"
		case strings.HasPrefix(key, "glm4.vision."), strings.HasPrefix(key, "glmocr.vision."):
			continue
		case key == "glm4.rope.mrope_section":
			modelKV["glm4.rope.dimension_sections"] = padInt32Sections(normalizeGGUFValue(rawGGUFValue(keyValue.Value)))
		case key == "glmocr.rope.mrope_section":
			modelKV["glm4.rope.dimension_sections"] = padInt32Sections(normalizeGGUFValue(rawGGUFValue(keyValue.Value)))
		case strings.HasPrefix(key, "glm4."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "glmocr."):
			modelKV[strings.Replace(key, "glmocr.", "glm4.", 1)] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case key == "rope.mrope_section":
			modelKV["glm4.rope.dimension_sections"] = padInt32Sections(normalizeGGUFValue(rawGGUFValue(keyValue.Value)))
		case glmOCRTextKVKey(key):
			modelKV["glm4."+key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}
	if src.GGUF.KeyValue("tokenizer.ggml.pre").Valid() {
		modelKV["tokenizer.ggml.pre"] = "chatglm-bpe"
	}
	if _, ok := modelKV["glm4.rope.dimension_count"]; !ok {
		if keyLength := src.GGUF.KeyValue("attention.key_length"); keyLength.Valid() {
			modelKV["glm4.rope.dimension_count"] = uint32(keyLength.Uint())
		}
	}
	if _, ok := modelKV["glm4.rope.dimension_sections"]; !ok {
		if sections := src.GGUF.KeyValue("rope.dimension_sections"); sections.Valid() {
			modelKV["glm4.rope.dimension_sections"] = padInt32Sections(normalizeGGUFValue(rawGGUFValue(sections.Value)))
		}
	}
	return modelKV
}

func glmOCRTextKVKey(key string) bool {
	switch key {
	case "block_count",
		"nextn_predict_layers",
		"embedding_length",
		"attention.head_count",
		"attention.head_count_kv",
		"attention.key_length",
		"attention.value_length",
		"feed_forward_length",
		"attention.layer_norm_rms_epsilon",
		"context_length",
		"rope.freq_base",
		"rope.dimension_count":
		return true
	default:
		return false
	}
}

func glmOCRProjectorKV(src *SourceModel) ggml.KV {
	projectorKV := ggml.KV{
		"general.architecture":                     "clip",
		"clip.projector_type":                      "glm4v",
		"clip.use_silu":                            true,
		"clip.has_vision_encoder":                  true,
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             uint32(src.GGUF.KeyValue("vision.embedding_length").Uint()),
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.image_size":                   uint32(src.GGUF.KeyValue("vision.image_size").Uint()),
		"clip.vision.patch_size":                   uint32(src.GGUF.KeyValue("vision.patch_size").Uint()),
		"clip.vision.spatial_merge_size":           uint32(src.GGUF.KeyValue("vision.spatial_merge_size").Uint()),
		"clip.vision.temporal_patch_size":          uint32(src.GGUF.KeyValue("vision.temporal_patch_size").Uint()),
		"clip.vision.out_hidden_size":              uint32(src.GGUF.KeyValue("vision.out_hidden_size").Uint()),
		"clip.vision.intermediate_size":            uint32(src.GGUF.KeyValue("vision.intermediate_size").Uint()),
		"clip.vision.feed_forward_length":          uint32(src.GGUF.KeyValue("vision.intermediate_size").Uint()),
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("vision.out_hidden_size").Uint()),
		"clip.vision.attention.layer_norm_epsilon": glmOCRFloat(src, "vision.attention.layer_norm_epsilon", "vision.attention.layer_norm_rms_epsilon"),
	}
	if mean := src.GGUF.KeyValue("vision.image_mean"); mean.Valid() {
		projectorKV["clip.vision.image_mean"] = normalizeGGUFValue(rawGGUFValue(mean.Value))
	}
	if std := src.GGUF.KeyValue("vision.image_std"); std.Valid() {
		projectorKV["clip.vision.image_std"] = normalizeGGUFValue(rawGGUFValue(std.Value))
	}
	if minPixels := glmOCRKeyValue(src, "vision.min_pixels", "vision.shortest_edge"); minPixels.Valid() {
		projectorKV["clip.vision.min_pixels"] = uint32(minPixels.Uint())
	}
	if maxPixels := glmOCRKeyValue(src, "vision.max_pixels", "vision.longest_edge"); maxPixels.Valid() {
		projectorKV["clip.vision.max_pixels"] = uint32(maxPixels.Uint())
	}
	for _, key := range []string{"image_token_id", "image_start_token_id", "image_end_token_id"} {
		if value := src.GGUF.KeyValue(key); value.Valid() {
			projectorKV["clip.vision."+key] = uint32(value.Uint())
		}
	}
	return projectorKV
}

func glmOCRKeyValue(src *SourceModel, keys ...string) gguf.KeyValue {
	for _, key := range keys {
		if value := src.GGUF.KeyValue(key); value.Valid() {
			return value
		}
	}
	return gguf.KeyValue{}
}

func glmOCRFloat(src *SourceModel, keys ...string) float32 {
	for _, key := range keys {
		if value := src.GGUF.KeyValue(key); value.Valid() {
			return float32(value.Float())
		}
	}
	return 0
}

func glmOCRTextLayerCount(src *SourceModel) int {
	blockCount := src.GGUF.KeyValue("block_count").Uint()
	nextN := src.GGUF.KeyValue("nextn_predict_layers").Uint()
	numLayers := blockCount
	if nextN > 0 && nextN < blockCount {
		numLayers = blockCount - nextN
	}
	return int(numLayers)
}

func glmOCRTextTensorName(name string, numLayers int) string {
	blkNum, ok := tensorBlock(name)
	if !ok || blkNum < numLayers {
		return name
	}

	prefix := "blk." + strconv.Itoa(blkNum) + "."
	suffix := strings.TrimPrefix(name, prefix)
	switch {
	case strings.HasPrefix(suffix, "eh_proj"):
		return prefix + "nextn." + suffix
	case strings.HasPrefix(suffix, "embed_tokens"):
		return prefix + "nextn.embed_tokens" + strings.TrimPrefix(suffix, "embed_tokens")
	case strings.HasPrefix(suffix, "enorm"):
		return prefix + "nextn." + suffix
	case strings.HasPrefix(suffix, "hnorm"):
		return prefix + "nextn." + suffix
	case strings.HasPrefix(suffix, "shared_head.head"):
		return prefix + "nextn.shared_head_head" + strings.TrimPrefix(suffix, "shared_head.head")
	case strings.HasPrefix(suffix, "shared_head.norm"):
		return prefix + "nextn.shared_head_norm" + strings.TrimPrefix(suffix, "shared_head.norm")
	default:
		return name
	}
}

func tensorBlock(name string) (int, bool) {
	if !strings.HasPrefix(name, "blk.") {
		return 0, false
	}
	rest := strings.TrimPrefix(name, "blk.")
	idx := strings.IndexByte(rest, '.')
	if idx < 0 {
		return 0, false
	}
	n, err := strconv.Atoi(rest[:idx])
	return n, err == nil
}

func glmOCRProjectorTensorName(name string) string {
	switch name {
	case "v.patch_embd_0.weight":
		return "v.patch_embd.weight"
	case "v.patch_embd_1.weight":
		return "v.patch_embd.weight.1"
	}

	replacer := strings.NewReplacer(
		"v.patch_embed.", "v.patch_embd.",
		"v.norm_embd", "v.post_ln",
		"attn.qkv", "attn_qkv",
		"attn.proj", "attn_out",
		"attn.q_norm", "attn_q_norm",
		"attn.k_norm", "attn_k_norm",
		"norm1", "ln1",
		"norm2", "ln2",
		"mlp.gate_proj", "ffn_gate",
		"mlp.up_proj", "ffn_up",
		"mlp.down_proj", "ffn_down",
		"mm.patch_merger.proj", "mm.model.fc",
	)
	return replacer.Replace(name)
}

func padInt32Sections(v any) []int32 {
	var sections []int32
	switch values := v.(type) {
	case []int32:
		sections = append([]int32{}, values...)
	case []uint32:
		for _, value := range values {
			sections = append(sections, int32(value))
		}
	}
	for len(sections) < 4 {
		sections = append(sections, 0)
	}
	return sections
}
