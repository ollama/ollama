package compatmigrate

import (
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
)

type llama3Migrator struct{}

// Mirrors detect_ollama_llama3_metadata_gap in llama/compat/llama-ollama-compat.cpp; keep the two in sync.
func (llama3Migrator) NeedsMigration(src *SourceModel) bool {
	return llama3NeedsMetadataFix(src)
}

func (llama3Migrator) Migrate(src *SourceModel) (*Result, error) {
	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}

	modelTensors := make([]*ggml.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		modelTensors = append(modelTensors, copyTensor(tensor.name, tensor))
	}

	modelKV := ggml.KV{}
	for _, keyValue := range src.GGUF.KeyValues() {
		if !keyValue.Valid() {
			continue
		}

		key := keyValue.Key
		if strings.HasPrefix(key, "general.") || strings.HasPrefix(key, "tokenizer.") || strings.HasPrefix(key, "llama.") {
			modelKV[key] = normalizeGGUFValue(keyValue.Any())
		}
	}
	if _, ok := modelKV["general.architecture"]; !ok {
		modelKV["general.architecture"] = "llama"
	}
	if llama3NeedsMetadataFix(src) {
		applyLlama3MetadataFix(modelKV)
	}

	result := &Result{
		ModelKV:      modelKV,
		ModelTensors: modelTensors,
	}
	if src.ProjectorGGUF != nil {
		result.PreserveProjector = true
	}

	return result, nil
}

// applyLlama3MetadataFix replaces the legacy default 128001 (<|end_of_text|>) with 128009 (<|eot_id|>).
func applyLlama3MetadataFix(modelKV ggml.KV) {
	if llama3StringMissingOrDefault(modelKV["tokenizer.ggml.pre"]) {
		modelKV["tokenizer.ggml.pre"] = "llama-bpe"
	}
	if llama3TokenizerIDMissingOrOld(modelKV["tokenizer.ggml.eos_token_id"]) {
		modelKV["tokenizer.ggml.eos_token_id"] = uint32(128009)
	}
	if llama3TokenizerIDMissingOrOld(modelKV["tokenizer.ggml.eot_token_id"]) {
		modelKV["tokenizer.ggml.eot_token_id"] = uint32(128009)
	}
	modelKV["tokenizer.ggml.eos_token_ids"] = []int32{128009}
}

// RewriteLlama3MetadataLayer rewrites a GGUF layer with the legacy Llama 3 tokenizer metadata gap.
func RewriteLlama3MetadataLayer(layer manifest.Layer) (manifest.Layer, bool, error) {
	path, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		return manifest.Layer{}, false, err
	}

	g, err := gguf.Open(path)
	if err != nil {
		return manifest.Layer{}, false, err
	}
	defer g.Close()

	src := &SourceModel{
		GGUFPath:       path,
		GGUF:           g,
		GGUFData:       g.ReaderAt(),
		GGUFDataOffset: g.TensorDataOffset(),
	}
	if !llama3NeedsMetadataFix(src) {
		return layer, false, nil
	}

	result, err := (llama3Migrator{}).Migrate(src)
	if err != nil {
		return manifest.Layer{}, false, err
	}

	rewritten, err := writeGGUFLayer(result.ModelKV, result.ModelTensors, layer.MediaType)
	if err != nil {
		return manifest.Layer{}, false, err
	}
	rewritten.Name = layer.Name
	return rewritten, true, nil
}

func llama3NeedsMetadataFix(src *SourceModel) bool {
	if src.GGUF.KeyValue("general.architecture").String() != "llama" {
		return false
	}
	if !llama3TokenAt(src, 128009, "<|eot_id|>") {
		return false
	}

	hasMarkers := llama3TokenAt(src, 128006, "<|start_header_id|>") ||
		strings.Contains(src.GGUF.KeyValue("tokenizer.chat_template").String(), "<|start_header_id|>") ||
		strings.Contains(src.GGUF.KeyValue("tokenizer.chat_template").String(), "<|eot_id|>")

	return hasMarkers &&
		(llama3StringMissingOrDefault(src.GGUF.KeyValue("tokenizer.ggml.pre").String()) ||
			llama3TokenizerIDOld(src.GGUF.KeyValue("tokenizer.ggml.eos_token_id").Any()))
}

func llama3TokenAt(src *SourceModel, index int, want string) bool {
	tokens := src.GGUF.KeyValue("tokenizer.ggml.tokens").Strings()
	return index >= 0 && index < len(tokens) && tokens[index] == want
}

func llama3StringMissingOrDefault(value any) bool {
	s, ok := value.(string)
	return !ok || s == "" || s == "default"
}

func llama3TokenizerIDMissingOrOld(value any) bool {
	switch value := value.(type) {
	case uint32:
		return value == 0 || value == 128001
	case uint64:
		return value == 0 || value == 128001
	case int32:
		return value <= 0 || value == 128001
	case int64:
		return value <= 0 || value == 128001
	case int:
		return value <= 0 || value == 128001
	default:
		return true
	}
}

func llama3TokenizerIDOld(value any) bool {
	switch value := value.(type) {
	case uint32:
		return value == 128001
	case uint64:
		return value == 128001
	case int32:
		return value == 128001
	case int64:
		return value == 128001
	case int:
		return value == 128001
	default:
		return false
	}
}
