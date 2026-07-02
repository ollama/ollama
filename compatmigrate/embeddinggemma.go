package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type embeddingGemmaMigrator struct{}

func (embeddingGemmaMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "gemma3" &&
		sourceTensorHasPrefix(src, "dense.0.")
}

func (embeddingGemmaMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	var modelTensors []*ggml.Tensor
	denseShapes := map[string][]uint64{}
	for _, tensor := range tensors {
		name := tensor.name
		switch {
		case strings.HasPrefix(name, "dense.0."):
			name = strings.Replace(name, "dense.0.", "dense_2.", 1)
			denseShapes["dense_2"] = tensor.shape
		case strings.HasPrefix(name, "dense.1."):
			name = strings.Replace(name, "dense.1.", "dense_3.", 1)
			denseShapes["dense_3"] = tensor.shape
		case name == "norm.weight":
			name = "output_norm.weight"
		}
		modelTensors = append(modelTensors, copyTensor(name, tensor))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		switch {
		case key == "general.architecture":
			modelKV[key] = "gemma-embedding"
		case key == "gemma3.attention.causal", key == "gemma3.attention.sliding_window_pattern":
			continue
		case strings.HasPrefix(key, "gemma3."):
			modelKV[strings.Replace(key, "gemma3.", "gemma-embedding.", 1)] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}

	modelKV["gemma-embedding.rope.freq_base_swa"] = float32(10000)
	for _, dense := range []string{"dense_2", "dense_3"} {
		if shape := denseShapes[dense]; len(shape) >= 2 {
			modelKV["gemma-embedding."+dense+"_feat_in"] = uint32(shape[0])
			modelKV["gemma-embedding."+dense+"_feat_out"] = uint32(shape[1])
		}
	}

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}, nil
}
