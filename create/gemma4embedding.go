package create

import (
	"encoding/json"
	"fmt"

	"github.com/ollama/ollama/types/model"
)

// Gemma4Embedding (EmbeddingGemma v2) quantization policy. The trunk is a
// 24-layer bidirectional encoder; the embedding_projection output head feeds
// mean-pooling + L2-norm downstream in the runner, so keep it unquantized
// (a low-rank 512→768 projection is cheap and precision-sensitive).
type gemma4EmbeddingImportTransform struct {
	embeddingDim int
	contextLen   int
	hasVision    bool
	hasAudio     bool
}

func newGemma4EmbeddingImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	// The arch's config carries both fields under text_config; parse them
	// here so the shared sourceModelConfig doesn't grow fields only this
	// arch reads.
	var cfg struct {
		TextConfig struct {
			EmbeddingDim          int `json:"embedding_dim"`
			MaxPositionEmbeddings int `json:"max_position_embeddings"`
		} `json:"text_config"`
		VisionConfig json.RawMessage `json:"vision_config"`
		AudioConfig  json.RawMessage `json:"audio_config"`
	}
	if err := json.Unmarshal(rawConfig, &cfg); err != nil {
		return nil, fmt.Errorf("gemma4embedding: parse config.json: %w", err)
	}
	return gemma4EmbeddingImportTransform{
		embeddingDim: cfg.TextConfig.EmbeddingDim,
		contextLen:   cfg.TextConfig.MaxPositionEmbeddings,
		// A null tower config means the checkpoint ships without it; a
		// present (even empty) object means the tower is there.
		hasVision: len(cfg.VisionConfig) > 0 && string(cfg.VisionConfig) != "null",
		hasAudio:  len(cfg.AudioConfig) > 0 && string(cfg.AudioConfig) != "null",
	}, nil
}

// augmentConfig fills the manifest fields inferSafetensorsConfig can't
// derive. Embedding is the arch's one unconditional capability (no lm_head,
// pooled output); vision/audio are advertised per checkpoint, following the
// towers it ships.
func (t gemma4EmbeddingImportTransform) augmentConfig(cfg *model.ConfigV2) error {
	if t.embeddingDim <= 0 {
		return fmt.Errorf("missing text_config.embedding_dim")
	}
	if t.contextLen <= 0 {
		return fmt.Errorf("missing text_config.max_position_embeddings")
	}
	cfg.EmbedLen = t.embeddingDim
	// The checkpoint's max_position_embeddings (262144) is a Gemma 4
	// backbone artifact: the RoPE encoding is mechanically sound to that
	// span, but the embedding model is trained and validated at 8K. The
	// manifest advertises the validated window; the runner enforces the
	// same constant (gemma4embedding.maxContextLength).
	cfg.ContextLen = min(t.contextLen, 8192)
	cfg.Capabilities = []string{"embedding"}
	if t.hasVision {
		cfg.Capabilities = append(cfg.Capabilities, "vision")
	}
	if t.hasAudio {
		cfg.Capabilities = append(cfg.Capabilities, "audio")
	}
	return nil
}

func (t gemma4EmbeddingImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	switch {
	case name == "embedding_projection.weight" ||
		name == "model.embedding_projection.weight" ||
		name == "language_model.embedding_projection.weight":
		// Output head feeding pooled/normalized embedding: keep at source
		// precision regardless of the requested type.
		return ""
	default:
		return GetTensorQuantization(name, shape, quantize)
	}
}
