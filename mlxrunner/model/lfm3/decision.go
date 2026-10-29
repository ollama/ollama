package lfm3

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"

	"github.com/ollama/ollama/mlxrunner/model"
)

// NewDecisionBackbone loads the LFM3-VL or LFM3-Omni backbone of a decision
// checkpoint: the decoder with its image encoder and, for Omni, its audio
// encoder, without speech output or chat-time token suppression.
func NewDecisionBackbone(root *model.Root) (*VLModel, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load decision backbone config: %w", err)
	}
	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse decision backbone config: %w", err)
	}
	textConfig, vision, err := parseVisionModel(root, data)
	if err != nil {
		return nil, err
	}
	var audio *audioInput
	switch cfg.ModelType {
	case "lfm3_vl":
	case "lfm3_omni":
		processor, err := root.Manifest.ReadConfig("processor_config.json")
		if err != nil {
			return nil, fmt.Errorf("load Omni processor: %w", err)
		}
		if audio, err = newAudioInput(data, processor, true); err != nil {
			return nil, err
		}
		if err := rejectQuantized(root, "model.audio_encoder.", "model.audio_projector."); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported decision backbone %q", cfg.ModelType)
	}
	// Decision checkpoints never generate and ship no generation_config.json.
	m, err := newModelFromConfig(root, textConfig, []byte(fmt.Sprintf(`{"bos_token_id":%d,"eos_token_id":%d,"pad_token_id":%d}`, bosTokenID, imEndTokenID, padTokenID)))
	if err != nil {
		return nil, err
	}
	m.prefix = "model.language_model."
	m.Vision = vision
	m.Audio = audio
	return &VLModel{Model: m}, nil
}

// CapImagePixels shrinks a picture larger than maxPixels before the image
// processor, as decision checkpoints are trained: bicubic to the truncated
// scaled size, keeping the aspect ratio. Smaller pictures are returned as is.
func CapImagePixels(data []byte, maxPixels int) ([]byte, error) {
	source, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	bounds := source.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w*h <= maxPixels {
		return data, nil
	}
	s := math.Sqrt(float64(maxPixels) / float64(w*h))
	rgb := image.NewNRGBA(bounds)
	draw.Draw(rgb, bounds, source, bounds.Min, draw.Src)
	for i := 3; i < len(rgb.Pix); i += 4 {
		rgb.Pix[i] = 0xff
	}
	resized := resizeBicubic(rgb, max(1, int(float64(w)*s)), max(1, int(float64(h)*s)))
	var out bytes.Buffer
	if err := png.Encode(&out, resized); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}
