package compatmigrate

import (
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type lfm25ThinkingMigrator struct{}

func (lfm25ThinkingMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "lfm2" &&
		sourceTensorExists(src, "output_norm.weight") &&
		!sourceTensorExists(src, "token_embd_norm.weight")
}

func (lfm25ThinkingMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	var hasEmbeddingNorm bool
	embedLength := src.GGUF.KeyValue("embedding_length").Uint()
	feedForwardLength := inferFeedForwardLength(tensors, embedLength)
	modelTensors := make([]*ggml.Tensor, 0, len(tensors)+1)
	for _, tensor := range tensors {
		switch tensor.name {
		case "token_embd_norm.weight":
			hasEmbeddingNorm = true
		case "output_norm.weight":
			hasEmbeddingNorm = true
			modelTensors = append(modelTensors, copyTensor("token_embd_norm.weight", tensor))
			continue
		}
		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	if !hasEmbeddingNorm {
		nEmbd := src.GGUF.KeyValue("embedding_length").Uint()
		if nEmbd == 0 {
			return nil, errUnsupportedFamily
		}

		// The public LFM2.5-thinking GGUF is missing the embedding norm that
		// llama-server requires. The original weights are not recoverable from
		// the installed artifact, so local migration uses identity scale. A clean
		// pull/re-create can still ship the real tensor if we publish one later.
		ones := slices.Repeat([]float32{1}, int(nEmbd))
		modelTensors = append(modelTensors, f32Tensor("token_embd_norm.weight", []uint64{nEmbd}, ones))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case key == "tokenizer.ggml.pre":
			modelKV[key] = "lfm2"
		case key == "general.architecture", strings.HasPrefix(key, "lfm2."), strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}
	if feedForwardLength > 0 {
		modelKV["lfm2.feed_forward_length"] = uint32(feedForwardLength)
	}

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
		Renderer:     "lfm2-thinking",
		Parser:       "lfm2-thinking",
	}, nil
}

func inferFeedForwardLength(tensors []*sourceTensor, embedLength uint64) uint64 {
	for _, tensor := range tensors {
		if !strings.HasSuffix(tensor.name, ".ffn_gate.weight") && !strings.HasSuffix(tensor.name, ".ffn_up.weight") {
			continue
		}
		if len(tensor.shape) < 2 {
			continue
		}
		if embedLength > 0 {
			switch {
			case tensor.shape[0] == embedLength && tensor.shape[1] != 0:
				return tensor.shape[1]
			case tensor.shape[1] == embedLength && tensor.shape[0] != 0:
				return tensor.shape[0]
			}
		}

		if tensor.shape[0] >= tensor.shape[1] {
			return tensor.shape[0]
		}
		return tensor.shape[1]
	}

	return 0
}
