package lfm3

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/mlxrunner/model"
)

type visionConfig struct {
	HiddenSize        int32    `json:"hidden_size"`
	IntermediateSize  int32    `json:"intermediate_size"`
	NumHiddenLayers   int32    `json:"num_hidden_layers"`
	NumAttentionHeads int32    `json:"num_attention_heads"`
	NumKeyValueHeads  int32    `json:"num_key_value_heads"`
	PatchSize         int32    `json:"patch_size"`
	TemporalPatchSize int32    `json:"temporal_patch_size"`
	SpatialMergeSize  int32    `json:"spatial_merge_size"`
	ConvKernelSize    int32    `json:"conv_kernel_size"`
	NormEps           float32  `json:"norm_eps"`
	LayerTypes        []string `json:"layer_types"`
	RopeParameters    struct {
		Theta float32 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

// VLModel adds still-image preparation to the shared language decoder.
type VLModel struct{ *Model }

func newVLModel(root *model.Root) (model.Model, error) {
	data, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("load VL config: %w", err)
	}
	textConfig, vision, err := parseVisionModel(root, data)
	if err != nil {
		return nil, err
	}
	m, err := newModelFromConfig(root, textConfig, nil)
	if err != nil {
		return nil, err
	}
	m.prefix = "model.language_model."
	m.Vision = vision
	return &VLModel{Model: m}, nil
}

// parseVisionModel reads the vision tower, projector, and image processor
// settings shared by LFM3 VL and Omni, returning the nested text config.
func parseVisionModel(root *model.Root, data []byte) (json.RawMessage, *VisionModel, error) {
	var cfg struct {
		TextConfig            json.RawMessage `json:"text_config"`
		VisionConfig          visionConfig    `json:"vision_config"`
		ImageTokenID          int32           `json:"image_token_id"`
		DownsampleFactor      int32           `json:"downsample_factor"`
		ProjectorHiddenSize   int32           `json:"projector_hidden_size"`
		ProjectorLayerNormEps float32         `json:"projector_layer_norm_eps"`
		ProjectorHiddenAct    string          `json:"projector_hidden_act"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, nil, fmt.Errorf("parse VL config: %w", err)
	}
	if len(cfg.TextConfig) == 0 || cfg.VisionConfig.NumHiddenLayers <= 0 {
		return nil, nil, fmt.Errorf("incomplete VL text or vision config")
	}
	processorData, err := root.Manifest.ReadConfig("processor_config.json")
	if err != nil {
		return nil, nil, fmt.Errorf("load VL image processor: %w", err)
	}
	var processor struct {
		Image struct {
			PatchSize         int32     `json:"patch_size"`
			MergeSize         int32     `json:"merge_size"`
			TemporalPatchSize int32     `json:"temporal_patch_size"`
			Resample          int       `json:"resample"`
			Mean              []float32 `json:"image_mean"`
			Std               []float32 `json:"image_std"`
			Size              struct {
				Min int32 `json:"shortest_edge"`
				Max int32 `json:"longest_edge"`
			} `json:"size"`
		} `json:"image_processor"`
	}
	if err := json.Unmarshal(processorData, &processor); err != nil {
		return nil, nil, fmt.Errorf("parse VL image processor: %w", err)
	}
	imageCfg := processor.Image
	if imageCfg.PatchSize != cfg.VisionConfig.PatchSize || imageCfg.MergeSize != cfg.VisionConfig.SpatialMergeSize ||
		imageCfg.TemporalPatchSize != cfg.VisionConfig.TemporalPatchSize || imageCfg.Resample != 3 ||
		imageCfg.Size.Min <= 0 || imageCfg.Size.Max < imageCfg.Size.Min ||
		len(imageCfg.Mean) != 3 || len(imageCfg.Std) != 3 {
		return nil, nil, fmt.Errorf("unsupported VL image processor configuration")
	}
	for i := range 3 {
		if imageCfg.Mean[i] != 0.5 || imageCfg.Std[i] != 0.5 {
			return nil, nil, fmt.Errorf("unsupported VL image normalization")
		}
	}
	for name := range root.AllTensorQuant() {
		if strings.HasPrefix(name, "model.vision_tower.") || strings.HasPrefix(name, "model.multi_modal_projector.") {
			return nil, nil, fmt.Errorf("quantized LFM3 VL vision weight %q is not supported", name)
		}
	}
	return cfg.TextConfig, &VisionModel{
		Config:              cfg.VisionConfig,
		ImageTokenID:        cfg.ImageTokenID,
		DownsampleFactor:    cfg.DownsampleFactor,
		ProjectorHiddenSize: cfg.ProjectorHiddenSize,
		ProjectorNormEps:    cfg.ProjectorLayerNormEps,
		ProjectorAct:        cfg.ProjectorHiddenAct,
		MinPixels:           imageCfg.Size.Min,
		MaxPixels:           imageCfg.Size.Max,
	}, nil
}
