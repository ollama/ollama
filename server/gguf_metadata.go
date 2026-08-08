package server

import (
	"strings"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/types/model"
)

const (
	ggufKeyGeneralArchitecture   = "general.architecture"
	ggufKeyGeneralFileType       = "general.file_type"
	ggufKeyTokenizerChatTemplate = "tokenizer.chat_template"

	ggufKeyAudioBlockCount  = "audio.block_count"
	ggufKeyContextLength    = "context_length"
	ggufKeyEmbeddingLength  = "embedding_length"
	ggufKeyPoolingType      = "pooling_type"
	ggufKeyVisionBlockCount = "vision.block_count"
)

type ggufArchitectureMetadata struct {
	ContextLength   int
	EmbeddingLength int
	HasAudio        bool
	HasEmbedding    bool
	HasVision       bool
}

func ggufArchitectureMetadataFromFile(f *gguf.File) ggufArchitectureMetadata {
	return ggufArchitectureMetadata{
		HasAudio:     f.KeyValue(ggufKeyAudioBlockCount).Valid(),
		HasEmbedding: f.KeyValue(ggufKeyPoolingType).Valid(),
		HasVision:    f.KeyValue(ggufKeyVisionBlockCount).Valid(),
	}
}

func appendGGUFMetadataCapabilities(capabilities []model.Capability, metadata ggufArchitectureMetadata) []model.Capability {
	if metadata.HasEmbedding {
		capabilities = appendCapability(capabilities, model.CapabilityEmbedding)
	} else {
		capabilities = appendCapability(capabilities, model.CapabilityCompletion)
	}
	if metadata.HasVision {
		capabilities = appendCapability(capabilities, model.CapabilityVision)
	}
	if metadata.HasAudio {
		capabilities = appendCapability(capabilities, model.CapabilityAudio)
	}
	return capabilities
}

func cutGGUFArchitectureKey(key string) (architecture, suffix string, ok bool) {
	return strings.Cut(key, ".")
}

func updateGGUFArchitectureMetadata(metadata *ggufArchitectureMetadata, suffix string, value gguf.Value) bool {
	switch suffix {
	case ggufKeyPoolingType:
		metadata.HasEmbedding = true
	case ggufKeyVisionBlockCount:
		metadata.HasVision = true
	case ggufKeyAudioBlockCount:
		metadata.HasAudio = true
	case ggufKeyContextLength:
		metadata.ContextLength = int(value.Uint())
	case ggufKeyEmbeddingLength:
		metadata.EmbeddingLength = int(value.Uint())
	default:
		return false
	}
	return true
}

func isGGUFArchitectureMetadataKey(key string) bool {
	_, suffix, ok := cutGGUFArchitectureKey(key)
	return ok && isGGUFArchitectureMetadataSuffix(suffix)
}

func isGGUFArchitectureMetadataSuffix(suffix string) bool {
	switch suffix {
	case ggufKeyPoolingType, ggufKeyVisionBlockCount, ggufKeyAudioBlockCount, ggufKeyContextLength, ggufKeyEmbeddingLength:
		return true
	default:
		return false
	}
}
