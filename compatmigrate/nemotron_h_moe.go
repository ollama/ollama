package compatmigrate

import (
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type nemotronHMoeMigrator struct{}

func (nemotronHMoeMigrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "nemotron_h_moe" {
		return false
	}
	return sourceTensorHasPrefix(src, "blk.1.ffn_latent_in") ||
		sourceTensorHasPrefix(src, "blk.0.ffn_latent_in") ||
		sourceTensorHasPrefix(src, "mtp.")
}

func (nemotronHMoeMigrator) Migrate(src *SourceModel) (*Result, error) {
	if src.GGUF.KeyValue("general.architecture").String() != "nemotron_h_moe" {
		return nil, errUnsupportedFamily
	}

	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	modelTensors := make([]*ggml.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		if strings.HasPrefix(tensor.name, "mtp.") {
			continue
		}
		modelTensors = append(modelTensors, copyTensor(nemotronHMoeTensorName(tensor.name), tensor))
	}

	modelKV := nemotronHMoeModelKV(src)
	if !rawGGUFKeyExists(src.GGUF, "nemotron_h_moe.moe_latent_size") {
		if latentSize, ok := nemotronHMoELatentSize(src); ok {
			modelKV["nemotron_h_moe.moe_latent_size"] = latentSize
		}
	}

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}, nil
}

func nemotronHMoeModelKV(src *SourceModel) ggml.KV {
	out := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		value := normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		switch {
		case strings.HasPrefix(key, "general."), strings.HasPrefix(key, "tokenizer."), strings.HasPrefix(key, "nemotron_h_moe."):
			out[key] = value
		}
	}
	if _, ok := out["general.architecture"]; !ok {
		out["general.architecture"] = "nemotron_h_moe"
	}
	return out
}

func nemotronHMoeTensorName(name string) string {
	name = strings.ReplaceAll(name, ".ffn_latent_in", ".ffn_latent_down")
	return strings.ReplaceAll(name, ".ffn_latent_out", ".ffn_latent_up")
}

func nemotronHMoELatentSize(src *SourceModel) (uint32, bool) {
	for i := range 1024 {
		shape, ok := sourceTensorShape(src, "blk."+strconv.Itoa(i)+".ffn_latent_in.weight")
		if !ok || len(shape) < 2 {
			continue
		}
		return uint32(shape[1]), true
	}
	return 0, false
}
