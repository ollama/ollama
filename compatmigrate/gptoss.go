package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type gptossMigrator struct{}

// Mirrors detect_ollama_gptoss in llama/compat/llama-ollama-compat.cpp; keep the two in sync.
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
			modelKV[strings.Replace(key, "gptoss.", "gpt-oss.", 1)] = normalizeGGUFValue(keyValue.Any())
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."):
			modelKV[key] = normalizeGGUFValue(keyValue.Any())
		}
	}

	if _, ok := modelKV["gpt-oss.expert_feed_forward_length"]; !ok {
		if feedForwardLength := gptOSSExpertFeedForwardLength(tensors); feedForwardLength > 0 {
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

func gptOSSExpertFeedForwardLength(tensors []*sourceTensor) uint32 {
	for _, tensor := range tensors {
		if tensor.name == "blk.0.ffn_gate_exps.weight" && len(tensor.shape) >= 2 {
			return uint32(tensor.shape[1])
		}
	}
	return 0
}
