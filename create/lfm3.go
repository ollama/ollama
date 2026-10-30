package create

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/types/model"
)

// lfm3ImportTransform quantizes only the language backbone of the multimodal
// LFM3 checkpoints; vision, audio encoder, projectors, and the speech decoder
// stay at source precision.
type lfm3ImportTransform struct {
	backbone string
	// Audio nests its language model config under "lfm", where shared
	// config inference doesn't look; the manifest records it instead.
	contextLen, embedLen int
	// speech models' voices are prompt descriptions, not voice tensors, so
	// the manifest names them.
	speech bool
}

func newLFM3ImportTransform(json.RawMessage) (quantizePolicy, error) {
	return lfm3ImportTransform{backbone: "model.language_model."}, nil
}

func newLFM3OmniImportTransform(json.RawMessage) (quantizePolicy, error) {
	return lfm3ImportTransform{backbone: "model.language_model.", speech: true}, nil
}

func newLFM3AudioImportTransform(raw json.RawMessage) (quantizePolicy, error) {
	var cfg struct {
		LFM struct {
			HiddenSize            int `json:"hidden_size"`
			MaxPositionEmbeddings int `json:"max_position_embeddings"`
		} `json:"lfm"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if cfg.LFM.HiddenSize <= 0 || cfg.LFM.MaxPositionEmbeddings <= 0 {
		return nil, fmt.Errorf("missing lfm.hidden_size or lfm.max_position_embeddings")
	}
	return lfm3ImportTransform{
		backbone:   "lfm.model.",
		contextLen: cfg.LFM.MaxPositionEmbeddings,
		embedLen:   cfg.LFM.HiddenSize,
		speech:     true,
	}, nil
}

func (t lfm3ImportTransform) augmentConfig(cfg *model.ConfigV2) error {
	if t.contextLen > 0 {
		cfg.ContextLen = t.contextLen
		cfg.EmbedLen = t.embedLen
	}
	if t.speech {
		cfg.Voices = renderers.LFM3SpeechVoices()
	}
	return nil
}

func (t lfm3ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if !strings.HasPrefix(name, t.backbone) {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}
