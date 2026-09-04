package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type lagunaMigrator struct{}

// Mirrors detect_ollama_laguna in llama/compat/llama-ollama-compat.cpp; keep the two in sync.
func (lagunaMigrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "laguna" {
		return false
	}
	return rawGGUFKeyExists(src.GGUF, "laguna.rope.swa.dimension_count") ||
		rawGGUFKeyExists(src.GGUF, "laguna.rope.swa.freq_base") ||
		sourceTensorExists(src, "blk.0.attn_g.weight")
}

func (lagunaMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	modelTensors := make([]*ggml.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		modelTensors = append(modelTensors, copyTensor(lagunaTensorName(tensor.name), tensor))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		value := normalizeGGUFValue(keyValue.Any())
		switch {
		case key == "laguna.rope.swa.dimension_count":
			modelKV["laguna.rope.dimension_count_swa"] = value
		case key == "laguna.rope.swa.freq_base":
			modelKV["laguna.rope.freq_base_swa"] = value
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."), strings.HasPrefix(key, "laguna."):
			modelKV[key] = value
		}
	}
	if _, ok := modelKV["general.architecture"]; !ok {
		modelKV["general.architecture"] = "laguna"
	}

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}, nil
}

func lagunaTensorName(name string) string {
	if base, ok := strings.CutSuffix(name, ".attn_g.weight"); ok {
		return base + ".attn_gate.weight"
	}
	return name
}
