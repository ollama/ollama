package compatmigrate

import (
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type gemma4Migrator struct{}

func (gemma4Migrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "gemma4" {
		return false
	}
	return sourceTensorHasPrefix(src, "a.") ||
		sourceTensorHasPrefix(src, "v.") ||
		sourceTensorHasPrefix(src, "mm.") ||
		sourceTensorHasPrefix(src, "model.vision_tower.") ||
		src.GGUF.KeyValue("tokenizer.ggml.model").String() == "llama"
}

func (gemma4Migrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}
	legacyAudioNames := hasLegacyGemma4AudioNames(tensors)

	var modelTensors []*ggml.Tensor
	var projectorTensors []*ggml.Tensor
	pendingGate := map[string]*sourceTensor{}
	pendingUp := map[string]*sourceTensor{}
	for _, tensor := range tensors {
		switch {
		case isGemma4ProjectorTensor(tensor.name):
			projectorTensors = append(projectorTensors, copyTensor(gemma4ProjectorTensorName(tensor.name, legacyAudioNames), tensor))
		case gemma4CanFuseExpert(tensors, tensor.name):
			gateUpName := gemma4GateUpExpertName(tensor.name)
			if strings.Contains(tensor.name, ".ffn_gate_exps.") {
				if up := pendingUp[gateUpName]; up != nil {
					merged, err := concatSourceTensorsDim(1, gateUpName, tensor, up)
					if err != nil {
						return nil, err
					}
					modelTensors = append(modelTensors, merged)
					delete(pendingUp, gateUpName)
					continue
				}
				pendingGate[gateUpName] = tensor
				continue
			}
			if gate := pendingGate[gateUpName]; gate != nil {
				merged, err := concatSourceTensorsDim(1, gateUpName, gate, tensor)
				if err != nil {
					return nil, err
				}
				modelTensors = append(modelTensors, merged)
				delete(pendingGate, gateUpName)
				continue
			}
			pendingUp[gateUpName] = tensor
		default:
			modelTensors = append(modelTensors, copyTensor(gemma4ModelTensorName(tensor.name), tensor))
		}
	}
	for _, tensor := range pendingGate {
		modelTensors = append(modelTensors, copyTensor(gemma4ModelTensorName(tensor.name), tensor))
	}
	for _, tensor := range pendingUp {
		modelTensors = append(modelTensors, copyTensor(gemma4ModelTensorName(tensor.name), tensor))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		if key == "general.architecture" || strings.HasPrefix(key, "general.") || strings.HasPrefix(key, "tokenizer.") {
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
			continue
		}

		if strings.HasPrefix(key, "gemma4.audio.") || strings.HasPrefix(key, "gemma4.vision.") {
			continue
		}
		if strings.HasPrefix(key, "gemma4.") {
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}
	if tokenizerModel, _ := modelKV["tokenizer.ggml.model"].(string); tokenizerModel == "llama" {
		modelKV["tokenizer.ggml.model"] = "gemma4"
	}

	projectorKV := ggml.KV{
		"general.architecture":                     "clip",
		"clip.has_vision_encoder":                  true,
		"clip.projector_type":                      "gemma4v",
		"clip.vision.projector_type":               "gemma4v",
		"clip.vision.block_count":                  uint32(src.GGUF.KeyValue("vision.block_count").Uint()),
		"clip.vision.embedding_length":             uint32(src.GGUF.KeyValue("vision.embedding_length").Uint()),
		"clip.vision.feed_forward_length":          uint32(src.GGUF.KeyValue("vision.feed_forward_length").Uint()),
		"clip.vision.attention.head_count":         uint32(src.GGUF.KeyValue("vision.attention.head_count").Uint()),
		"clip.vision.attention.layer_norm_epsilon": float32(src.GGUF.KeyValue("vision.attention.layer_norm_epsilon").Float()),
		"clip.vision.patch_size":                   uint32(src.GGUF.KeyValue("vision.patch_size").Uint()),
		"clip.vision.image_size":                   uint32(224),
		"clip.vision.projection_dim":               uint32(src.GGUF.KeyValue("embedding_length").Uint()),
		"clip.vision.image_mean":                   []float32{0, 0, 0},
		"clip.vision.image_std":                    []float32{1, 1, 1},
	}
	if scale := src.GGUF.KeyValue("vision.projector.scale_factor"); scale.Valid() {
		projectorKV["clip.vision.projector.scale_factor"] = uint32(scale.Uint())
	}
	if blockCount := src.GGUF.KeyValue("audio.block_count"); blockCount.Valid() && blockCount.Uint() > 0 {
		audioFeedForwardLength := uint32(src.GGUF.KeyValue("audio.feed_forward_length").Uint())
		audioEmbeddingLength := uint32(src.GGUF.KeyValue("audio.embedding_length").Uint())
		if audioFeedForwardLength == 0 {
			audioFeedForwardLength = audioEmbeddingLength * 4
		}
		// llama.cpp treats clip.projector_type as authoritative when present.
		// Mixed Gemma4 projectors must use modality-specific projector keys so
		// audio selects gemma4a instead of falling through to gemma4v.
		delete(projectorKV, "clip.projector_type")
		projectorKV["clip.has_audio_encoder"] = true
		projectorKV["clip.audio.projector_type"] = "gemma4a"
		projectorKV["clip.audio.num_mel_bins"] = uint32(128)
		projectorKV["clip.audio.projection_dim"] = uint32(src.GGUF.KeyValue("embedding_length").Uint())
		projectorKV["clip.audio.embedding_length"] = audioEmbeddingLength
		projectorKV["clip.audio.feed_forward_length"] = audioFeedForwardLength
		projectorKV["clip.audio.block_count"] = uint32(blockCount.Uint())
		projectorKV["clip.audio.attention.head_count"] = uint32(src.GGUF.KeyValue("audio.attention.head_count").Uint())
		projectorKV["clip.audio.attention.layer_norm_epsilon"] = float32(1e-5)
	}

	return &Result{
		ModelKV:          modelKV,
		ModelTensors:     modelTensors,
		ProjectorKV:      projectorKV,
		ProjectorTensors: projectorTensors,
		Renderer:         "gemma4",
		Parser:           "gemma4",
	}, nil
}

func isGemma4AudioTensor(name string) bool {
	return strings.HasPrefix(name, "a.") || strings.HasPrefix(name, "mm.a.")
}

func isGemma4ProjectorTensor(name string) bool {
	return strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.") || isGemma4AudioTensor(name)
}

func gemma4ModelTensorName(name string) string {
	return strings.Replace(name, ".ffn_gate_inp.per_expert_scale", ".ffn_down_exps.scale", 1)
}

func gemma4CanFuseExpert(tensors []*sourceTensor, name string) bool {
	if !strings.Contains(name, ".ffn_gate_exps.weight") && !strings.Contains(name, ".ffn_up_exps.weight") {
		return false
	}
	block, ok := tensorBlock(name)
	if !ok || gemma4ExpertSidecarExists(tensors, block) {
		return false
	}
	return !sourceTensorNameExists(tensors, gemma4GateUpExpertName(name))
}

func gemma4ExpertSidecarExists(tensors []*sourceTensor, block int) bool {
	prefix := "blk." + strconv.Itoa(block) + "."
	for _, name := range []string{
		prefix + "ffn_gate_exps.scale",
		prefix + "ffn_gate_exps.input_scale",
		prefix + "ffn_up_exps.scale",
		prefix + "ffn_up_exps.input_scale",
	} {
		if sourceTensorNameExists(tensors, name) {
			return true
		}
	}
	return false
}

func sourceTensorNameExists(tensors []*sourceTensor, name string) bool {
	for _, tensor := range tensors {
		if tensor.name == name {
			return true
		}
	}
	return false
}

func gemma4GateUpExpertName(name string) string {
	name = strings.Replace(name, ".ffn_gate_exps.", ".ffn_gate_up_exps.", 1)
	return strings.Replace(name, ".ffn_up_exps.", ".ffn_gate_up_exps.", 1)
}

func hasLegacyGemma4AudioNames(tensors []*sourceTensor) bool {
	for _, tensor := range tensors {
		name := tensor.name
		if strings.Contains(name, ".linear_pos.") ||
			strings.Contains(name, ".layer_pre_norm.") ||
			strings.HasPrefix(name, "mm.a.fc.") {
			return true
		}
	}
	return false
}

func gemma4ProjectorTensorName(name string, legacyAudioNames bool) string {
	if !legacyAudioNames || !isGemma4AudioTensor(name) {
		return name
	}

	name = strings.Replace(name, ".linear_pos.", ".attn_k_rel.", 1)
	name = strings.Replace(name, ".ln1.", ".attn_pre_norm.", 1)
	name = strings.Replace(name, ".ln2.", ".attn_post_norm.", 1)
	name = strings.Replace(name, ".layer_pre_norm.", ".ln2.", 1)
	if strings.HasPrefix(name, "mm.a.fc.") {
		name = "a.pre_encode.out." + strings.TrimPrefix(name, "mm.a.fc.")
	} else if strings.HasPrefix(name, "a.pre_encode.out.") {
		name = "a.input_projection." + strings.TrimPrefix(name, "a.pre_encode.out.")
	}

	return name
}
