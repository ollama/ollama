package compatmigrate

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type snowflakeArcticEmbed2Migrator struct{}

func (snowflakeArcticEmbed2Migrator) NeedsMigration(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "bert" ||
		src.GGUF.KeyValue("tokenizer.ggml.model").String() != "t5" {
		return false
	}

	charsMap := src.GGUF.KeyValue("tokenizer.ggml.precompiled_charsmap")
	if !charsMap.Valid() {
		return false
	}
	_, ok := rawGGUFValue(charsMap.Value).([]string)
	return ok
}

func (snowflakeArcticEmbed2Migrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		value := rawGGUFValue(keyValue.Value)
		if keyValue.Key == "tokenizer.ggml.precompiled_charsmap" {
			chars, ok := value.([]string)
			if !ok {
				return nil, fmt.Errorf("unexpected precompiled chars map type %T", value)
			}

			for i, s := range chars {
				if len(s) != 1 {
					return nil, fmt.Errorf("unexpected precompiled chars map entry length %d at %d", len(s), i)
				}
			}

			buf, err := base64.StdEncoding.DecodeString(strings.Join(chars, ""))
			if err != nil {
				return nil, fmt.Errorf("decode precompiled chars map: %w", err)
			}
			modelKV[keyValue.Key] = buf
			continue
		}

		modelKV[keyValue.Key] = normalizeGGUFValue(value)
	}

	modelTensors := make([]*ggml.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	return &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}, nil
}
