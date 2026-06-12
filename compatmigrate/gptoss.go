package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type gptossMigrator struct{}

func (gptossMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "gptoss"
}

func (gptossMigrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	modelTensors := make([]*ggml.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		name := strings.Replace(tensor.name, ".attn_out.", ".attn_output.", 1)
		name = strings.Replace(name, ".ffn_norm.", ".post_attention_norm.", 1)
		if strings.HasSuffix(name, ".attn_sinks") {
			name += ".weight"
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
			modelKV[key] = "gpt-oss"
		case strings.HasPrefix(key, "gptoss."):
			modelKV[strings.Replace(key, "gptoss.", "gpt-oss.", 1)] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}

	if _, ok := modelKV["gpt-oss.expert_feed_forward_length"]; !ok {
		if feedForwardLength, ok := modelKV["gpt-oss.feed_forward_length"]; ok {
			modelKV["gpt-oss.expert_feed_forward_length"] = feedForwardLength
		}
	}
	if _, ok := modelKV["gpt-oss.rope.scaling.type"]; !ok {
		modelKV["gpt-oss.rope.scaling.type"] = "yarn"
	}

	modelKV["tokenizer.ggml.pre"] = "gpt-4o"

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}, nil
}
