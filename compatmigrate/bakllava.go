package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type bakllavaMigrator struct{}

func (bakllavaMigrator) NeedsMigration(src *SourceModel) bool {
	return src.ProjectorGGUF != nil &&
		!rawGGUFKeyExists(src.ProjectorGGUF, "clip.projector_type") &&
		!rawGGUFKeyExists(src.ProjectorGGUF, "clip.vision.projector_type")
}

func (bakllavaMigrator) Migrate(src *SourceModel) (*Result, error) {
	modelTensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}
	projectorTensors, err := readAllProjectorTensors(src)
	if err != nil {
		return nil, err
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}
		key := keyValue.Key
		if strings.HasPrefix(key, "general.") || strings.HasPrefix(key, "tokenizer.") || strings.HasPrefix(key, "llama.") {
			modelKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}
	if llama3NeedsMetadataFix(src) {
		applyLlama3MetadataFix(modelKV)
	}

	projectorKV := ggml.KV{}
	for _, keyValue := range src.ProjectorGGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}
		key := keyValue.Key
		if strings.HasPrefix(key, "general.") || strings.HasPrefix(key, "clip.") {
			projectorKV[key] = normalizeGGUFValue(rawGGUFValue(keyValue.Value))
		}
	}

	// The public bakllava projector is already split and tensor-compatible, but
	// it predates llama-server's required projector type metadata. Local
	// migration patches only that missing KV so users with the old q4_0 library
	// artifact can load it without a pull. An explicit pull still gets the
	// cleaner recreated manifest-list artifact once registry support is live.
	projectorKV["clip.projector_type"] = "mlp"

	modelOut := make([]*ggml.Tensor, 0, len(modelTensors))
	for _, tensor := range modelTensors {
		modelOut = append(modelOut, copyTensor(tensor.name, tensor))
	}
	projectorOut := make([]*ggml.Tensor, 0, len(projectorTensors))
	for _, tensor := range projectorTensors {
		projectorOut = append(projectorOut, copyTensor(tensor.name, tensor))
	}

	return &Result{
		ModelKV:          modelKV,
		ModelTensors:     modelOut,
		ProjectorKV:      projectorKV,
		ProjectorTensors: projectorOut,
	}, nil
}
