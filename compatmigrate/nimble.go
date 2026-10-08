package compatmigrate

import (
	_ "embed"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/manifest"
	"golang.org/x/mod/semver"
)

type nimbleMigrator struct{}

// The published original Nimble GGUFs predate llama.cpp's decision metadata.
// Identify the checkpoint, not the user's tag; other Qwen fine-tunes use the
// same backbone but do not necessarily share Nimble's readout or prompt.
func (nimbleMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "qwen35" &&
		src.GGUF.KeyValue("general.basename").String() == "Bespoke-Nimble" &&
		src.GGUF.KeyValue("general.finetune").String() == "merged-current" &&
		src.GGUF.KeyValue("decision.type").String() == ""
}

func (nimbleMigrator) Migrate(src *SourceModel) (*Result, error) {
	chatTemplate := src.GGUF.KeyValue("tokenizer.chat_template").String()
	if chatTemplate == "" || src.Config.Renderer != "" || src.Config.Parser != "" || (qwen35Migrator{}).NeedsMigration(src) {
		return nil, fmt.Errorf("Nimble requires its native chat template and standard Qwen3.5 tensors: %w", errUnsupportedFamily)
	}
	var system string
	for _, layer := range src.Manifest.Layers {
		switch layer.MediaType {
		case manifest.MediaTypeImageTemplate:
			return nil, fmt.Errorf("Nimble has a custom Go template: %w", errUnsupportedFamily)
		case manifest.MediaTypeImageSystem:
			path, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return nil, err
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			system = string(data)
		}
	}

	kv := outKV{}
	for _, entry := range src.GGUF.KeyValues() {
		if entry.Valid() {
			kv[entry.Key] = normalizeGGUFValue(entry.Any())
		}
	}
	kv["qwen35.decision.type"] = "nimble"
	// Preserve the original chat wrapper and SYSTEM, rather than adopting the
	// v3 checkpoint's template from upstream's converter.
	quote := strings.NewReplacer("\\", "\\\\", "'", "\\'", "\n", "\\n", "\r", "\\r", "\t", "\\t")
	kv["tokenizer.chat_template.systemone"] = "{% set ollama_system = '" + quote.Replace(system) + "' %}" + nimbleSchemaTemplate + chatTemplate

	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}
	requires, err := decisionMigrationVersion(src.Config.Requires)
	if err != nil {
		return nil, err
	}
	result := &Result{ModelKV: kv, PreserveProjector: src.ProjectorGGUF != nil, Requires: requires}
	for _, tensor := range tensors {
		// Metadata-only conversion: preserve every tensor's dtype and bytes.
		result.ModelTensors = append(result.ModelTensors, &outTensor{
			Name: tensor.name, Kind: uint32(tensor.info.Type), Shape: slices.Clone(tensor.shape), WriterTo: tensor.Clone(),
		})
	}
	return result, nil
}

func decisionMigrationVersion(source string) (string, error) {
	requires := "0.41.0"
	sourceVersion := "v" + strings.TrimPrefix(source, "v")
	if source != "" && !semver.IsValid(sourceVersion) {
		return "", fmt.Errorf("invalid minimum Ollama version %q", source)
	}
	if semver.Compare(sourceVersion, "v"+requires) > 0 {
		requires = source
	}
	return requires, nil
}

//go:embed templates/nimble.jinja
var nimbleSchemaTemplate string
