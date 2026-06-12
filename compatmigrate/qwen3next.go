package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type qwen3NextMigrator struct{}

func (qwen3NextMigrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "qwen3next" {
		return false
	}
	for _, tensor := range src.GGUF.TensorInfos() {
		if strings.HasSuffix(tensor.Name, ".ssm_dt") {
			return true
		}
	}
	return false
}

func (qwen3NextMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	modelTensors := make([]*ggml.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		modelTensors = append(modelTensors, copyTensor(qwen3NextTensorName(tensor.name), tensor))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		value := normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		switch {
		case key == "qwen3next.attention.head_count_kv":
			modelKV[key] = qwen3NextMaxHeadCountKV(value)
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."), strings.HasPrefix(key, "qwen3next."):
			modelKV[key] = value
		}
	}
	if _, ok := modelKV["general.architecture"]; !ok {
		modelKV["general.architecture"] = "qwen3next"
	}

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}, nil
}

func qwen3NextTensorName(name string) string {
	if strings.HasSuffix(name, ".ssm_dt") {
		return name + ".bias"
	}
	return name
}

func qwen3NextMaxHeadCountKV(value any) any {
	switch values := value.(type) {
	case []uint32:
		var maxValue uint32
		for _, value := range values {
			maxValue = max(maxValue, value)
		}
		if maxValue == 0 {
			return value
		}
		return maxValue
	case []int32:
		var maxValue uint32
		for _, value := range values {
			if value > 0 {
				maxValue = max(maxValue, uint32(value))
			}
		}
		if maxValue == 0 {
			return value
		}
		return maxValue
	default:
		return value
	}
}
