package compatmigrate

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

const gemma3nVocabSize = 262144

type gemma3nMigrator struct{}

func (gemma3nMigrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "gemma3n" {
		return false
	}
	token, tokenOK := sourceTensorShape(src, "token_embd.weight")
	perLayer, perLayerOK := sourceTensorShape(src, "per_layer_token_embd.weight")
	return tokenOK && perLayerOK && len(token) >= 2 && len(perLayer) >= 2 && token[1] != perLayer[1]
}

func (gemma3nMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	perLayerVocab := uint64(gemma3nVocabSize)
	perLayerInput := src.GGUF.KeyValue("embedding_length_per_layer_input").Uint()
	embedLength := src.GGUF.KeyValue("embedding_length").Uint()
	for _, tensor := range tensors {
		if tensor.name == "per_layer_token_embd.weight" && len(tensor.shape) >= 2 {
			perLayerVocab = inferEmbeddingVocabDim(tensor.shape, perLayerInput, embedLength, perLayerVocab)
			break
		}
	}

	var modelTensors []*ggml.Tensor
	for _, tensor := range tensors {
		if gemma3nDroppedTensor(tensor.name) {
			continue
		}

		if tensor.name == "token_embd.weight" && perLayerVocab > 0 && len(tensor.shape) >= 2 {
			shape := slices.Clone(tensor.shape)
			switch inferEmbeddingVocabAxis(tensor.shape, perLayerVocab, embedLength, perLayerInput) {
			case 0:
				if shape[0] > perLayerVocab {
					shape[0] = perLayerVocab
					modelTensors = append(modelTensors, copyTensorPrefix(tensor.name, tensor, shape))
					continue
				}
			case 1:
				if shape[1] > perLayerVocab {
					shape[1] = perLayerVocab
					modelTensors = append(modelTensors, copyTensorPrefix(tensor.name, tensor, shape))
					continue
				}
			}
		}

		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		value := normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		switch keyValue.Key {
		case "tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type":
			value = truncateGemma3nTokenizerValue(value, perLayerVocab)
		}
		modelKV[keyValue.Key] = value
	}
	modelKV["tokenizer.ggml.model"] = "llama"
	normalizeGemma3nTokenTypes(modelKV)
	ensureGemma3nEOS(modelKV)

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}, nil
}

func gemma3nDroppedTensor(name string) bool {
	return strings.Contains(name, "audio_tower") ||
		strings.Contains(name, "embed_audio") ||
		strings.Contains(name, "vision_tower") ||
		strings.Contains(name, "embed_vision")
}

func truncateGemma3nTokenizerValue(v any, targetVocab uint64) any {
	if targetVocab == 0 {
		return v
	}

	switch values := v.(type) {
	case []string:
		if uint64(len(values)) > targetVocab {
			return slices.Clone(values[:targetVocab])
		}
	case []float32:
		if uint64(len(values)) > targetVocab {
			return slices.Clone(values[:targetVocab])
		}
	case []int32:
		if uint64(len(values)) > targetVocab {
			return slices.Clone(values[:targetVocab])
		}
	}
	return v
}

func normalizeGemma3nTokenTypes(kv ggml.KV) {
	tokens, tokensOK := kv["tokenizer.ggml.tokens"].([]string)
	types, typesOK := kv["tokenizer.ggml.token_type"].([]int32)
	if !tokensOK || !typesOK {
		return
	}

	n := min(len(tokens), len(types))
	normalized := slices.Clone(types)
	for i := range n {
		switch token := tokens[i]; {
		case len(token) == 6 && strings.HasPrefix(token, "<0x") && strings.HasSuffix(token, ">"):
			normalized[i] = 6
		case strings.HasPrefix(token, "<unused"):
			normalized[i] = 1
		}
	}
	kv["tokenizer.ggml.token_type"] = normalized
}

func inferEmbeddingVocabDim(shape []uint64, innerDim uint64, outerDim uint64, fallback uint64) uint64 {
	switch inferEmbeddingVocabAxis(shape, fallback, outerDim, innerDim) {
	case 0:
		return shape[0]
	case 1:
		return shape[1]
	default:
		return fallback
	}
}

func inferEmbeddingVocabAxis(shape []uint64, vocabHint uint64, knownDims ...uint64) int {
	if len(shape) < 2 {
		return -1
	}

	if shape[0] == vocabHint && shape[1] != vocabHint {
		return 0
	}
	if shape[1] == vocabHint && shape[0] != vocabHint {
		return 1
	}

	for _, known := range knownDims {
		if known == 0 {
			continue
		}
		if shape[0] == known && shape[1] != known {
			return 1
		}
		if shape[1] == known && shape[0] != known {
			return 0
		}
	}

	if shape[0] >= shape[1] {
		return 0
	}
	return 1
}

func ensureGemma3nEOS(kv ggml.KV) {
	const turnToken = int32(106)

	switch ids := kv["tokenizer.ggml.eos_token_ids"].(type) {
	case []int32:
		if !slices.Contains(ids, turnToken) {
			kv["tokenizer.ggml.eos_token_ids"] = append(slices.Clone(ids), turnToken)
		}
	case []uint32:
		if !slices.Contains(ids, uint32(turnToken)) {
			kv["tokenizer.ggml.eos_token_ids"] = append(slices.Clone(ids), uint32(turnToken))
		}
	default:
		kv["tokenizer.ggml.eos_token_ids"] = []int32{1, turnToken}
	}
}
