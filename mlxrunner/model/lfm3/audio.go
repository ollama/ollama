package lfm3

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/model/lfm3/lfm3audio"
)

const (
	bosTokenID        = 124894
	imStartTokenID    = 124899
	imEndTokenID      = 124900
	padTokenID        = 124893
	audioStartTokenID = 125012
	textEndTokenID    = 125014
	mixedStartTokenID = 125015
	audioTokenID      = 125031
	audioOutTokenID   = 125032
)

func init() {
	model.Register("Lfm3OmniForConditionalGeneration", newOmniModel)
	model.Register("Lfm3AudioForConditionalGeneration", newAudioModel)
}

// audioInput turns a speech clip into encoder embeddings spliced over its
// <|audio|> placeholders. Omni wraps the placeholders in vision span markers.
type audioInput struct {
	cfg     lfm3audio.Config
	fe      *lfm3audio.FeatureExtractor
	Enc     *lfm3audio.Encoder
	markers bool
}

func newAudioInput(configJSON, processorJSON []byte, markers bool) (*audioInput, error) {
	cfg, err := lfm3audio.ParseConfig(configJSON, processorJSON)
	if err != nil {
		return nil, err
	}
	return &audioInput{cfg: cfg, fe: lfm3audio.NewFeatureExtractor(cfg), markers: markers}, nil
}

func (a *audioInput) prepare(data []byte) (model.PreparedItem, []int32, error) {
	f, err := a.fe.FromWAV(data)
	if err != nil {
		return model.PreparedItem{}, nil, err
	}
	var lead int32
	tokens := make([]int32, 0, f.Tokens+2)
	if a.markers {
		tokens = append(tokens, visionStartTokenID)
		lead = 1
	}
	for range f.Tokens {
		tokens = append(tokens, audioTokenID)
	}
	if a.markers {
		tokens = append(tokens, visionEndTokenID)
	}
	return model.PreparedItem{
		MediaData: f.Data,
		Dims:      []int{len(f.Data) / f.Frames, f.Frames},
		Opaque:    preparedAudio{Tokens: int32(f.Tokens), Lead: lead},
		Causal:    true,
	}, tokens, nil
}

// speechTokens names the ids these models rely on; the tokenizer must agree.
var speechTokens = map[int32]string{
	bosTokenID:         "<|startoftext|>",
	imStartTokenID:     "<|im_start|>",
	imEndTokenID:       "<|im_end|>",
	visionStartTokenID: "<|vision_start|>",
	visionEndTokenID:   "<|vision_end|>",
	audioStartTokenID:  "<|audio_start|>",
	textEndTokenID:     "<|text_end|>",
	mixedStartTokenID:  "<|mixed_start|>",
	audioTokenID:       "<|audio|>",
	audioOutTokenID:    "<|audio_out|>",
}

// <|text_end|> closes a text-mode turn as well as <|im_end|>.
var speechGenerationConfig = []byte(fmt.Sprintf(`{"bos_token_id":%d,"eos_token_id":[%d,%d]}`, bosTokenID, imEndTokenID, textEndTokenID))

// finishSpeechModel checks the token ids and keeps chat decoding in text mode:
// the mode markers would switch the model to speech output, and a new turn
// never belongs inside the assistant's reply.
func finishSpeechModel(m *Model) error {
	for id, want := range speechTokens {
		if got := m.tok.Decode([]int32{id}); got != want {
			return fmt.Errorf("LFM3 token %d decodes to %q, want %q", id, got, want)
		}
	}
	bias := make([]float32, m.VocabSize)
	for _, id := range []int32{imStartTokenID, audioStartTokenID, mixedStartTokenID, audioOutTokenID} {
		bias[id] = float32(math.Inf(-1))
	}
	m.Suppress = mlx.FromValues(bias, 1, 1, int(m.VocabSize))
	return nil
}

func rejectQuantized(root *model.Root, prefixes ...string) error {
	for name := range root.AllTensorQuant() {
		for _, prefix := range prefixes {
			if strings.HasPrefix(name, prefix) {
				return fmt.Errorf("quantized LFM3 encoder weight %q is not supported", name)
			}
		}
	}
	return nil
}

func newOmniModel(root *model.Root) (model.Model, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load Omni config: %w", err)
	}
	textConfig, vision, err := parseVisionModel(root, data)
	if err != nil {
		return nil, err
	}
	processor, err := root.Manifest.ReadConfig("processor_config.json")
	if err != nil {
		return nil, fmt.Errorf("load Omni processor: %w", err)
	}
	audio, err := newAudioInput(data, processor, true)
	if err != nil {
		return nil, err
	}
	if err := rejectQuantized(root, "model.audio_encoder.", "model.audio_projector."); err != nil {
		return nil, err
	}
	m, err := newModelFromConfig(root, textConfig, speechGenerationConfig)
	if err != nil {
		return nil, err
	}
	m.prefix = "model.language_model."
	m.Vision = vision
	m.Audio = audio
	if err := finishSpeechModel(m); err != nil {
		return nil, err
	}
	return newSpeechModel(root, &VLModel{Model: m}, data, true)
}

func newAudioModel(root *model.Root) (model.Model, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load Audio config: %w", err)
	}
	var cfg struct {
		LFM json.RawMessage `json:"lfm"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse Audio config: %w", err)
	}
	if len(cfg.LFM) == 0 {
		return nil, fmt.Errorf("Audio config has no lfm block")
	}
	// The preprocessor block in config.json is authoritative when the
	// separate preprocessor file is absent.
	var processor []byte
	if _, ok := root.Manifest.ConfigLayer("preprocessor_config.json"); ok {
		if processor, err = root.Manifest.ReadConfig("preprocessor_config.json"); err != nil {
			return nil, fmt.Errorf("load Audio preprocessor: %w", err)
		}
	}
	audio, err := newAudioInput(data, processor, false)
	if err != nil {
		return nil, err
	}
	if err := rejectQuantized(root, "encoder.", "mlp.model."); err != nil {
		return nil, err
	}
	m, err := newModelFromConfig(root, cfg.LFM, speechGenerationConfig)
	if err != nil {
		return nil, err
	}
	m.prefix = "lfm.model."
	m.Audio = audio
	if err := finishSpeechModel(m); err != nil {
		return nil, err
	}
	return newSpeechModel(root, &VLModel{Model: m}, data, false)
}
