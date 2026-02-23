package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// lfm2VLTextModel converts the language model component of LFM2 VL checkpoints.
type lfm2VLTextModel struct {
	TextConfig            lfm2Model `json:"text_config"`
	DoImageSplitting      *bool     `json:"do_image_splitting"`
	DownsampleFactor      uint32    `json:"downsample_factor"`
	EncoderPatchSize      uint32    `json:"encoder_patch_size"`
	ImageTokenID          uint32    `json:"image_token_id"`
	MaxImageTokens        uint32    `json:"max_image_tokens"`
	MinImageTokens        uint32    `json:"min_image_tokens"`
	MaxTiles              uint32    `json:"max_tiles"`
	MinTiles              uint32    `json:"min_tiles"`
	TileSize              uint32    `json:"tile_size"`
	MaxPixelsTolerance    float32   `json:"max_pixels_tolerance"`
	ProjectorUseLayernorm bool      `json:"projector_use_layernorm"`
	ProjectorHiddenSize   uint32    `json:"projector_hidden_size"`
	ProjectorHiddenAct    string    `json:"projector_hidden_act"`
	UseImageSpecialTokens *bool     `json:"use_image_special_tokens"`
	UseThumbnail          *bool     `json:"use_thumbnail"`
	VisionConfig          struct {
		HiddenSize        uint32  `json:"hidden_size"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		NumChannels       uint32  `json:"num_channels"`
		PatchSize         uint32  `json:"patch_size"`
		LayerNormEpsilon  float32 `json:"layer_norm_eps"`
	} `json:"vision_config"`
	Processor struct {
		ImageProcessor struct {
			DoImageSplitting *bool     `json:"do_image_splitting"`
			DownsampleFactor uint32    `json:"downsample_factor"`
			MaxImageTokens   uint32    `json:"max_image_tokens"`
			MinImageTokens   uint32    `json:"min_image_tokens"`
			MaxTiles         uint32    `json:"max_tiles"`
			MinTiles         uint32    `json:"min_tiles"`
			MaxPixelsTol     float32   `json:"max_pixels_tolerance"`
			TileSize         uint32    `json:"tile_size"`
			UseThumbnail     *bool     `json:"use_thumbnail"`
			ImageMean        []float32 `json:"image_mean"`
			ImageStd         []float32 `json:"image_std"`
			Size             struct {
				Height uint32 `json:"height"`
				Width  uint32 `json:"width"`
			} `json:"size"`
		} `json:"image_processor"`
	}
}

func (p *lfm2VLTextModel) textModel() *lfm2Model {
	return &p.TextConfig
}

func (p *lfm2VLTextModel) specialTokenTypes() []string {
	return p.textModel().specialTokenTypes()
}

func (p *lfm2VLTextModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "processor_config.json")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}

	return json.Unmarshal(bts, &p.Processor)
}

func (p *lfm2VLTextModel) visionImageSize() uint32 {
	// LFM2-VL image processor operates on 512 tiles and downsamples by factor 2
	// before projection. Keep a fixed square image size compatible with position
	// embeddings and the simplified runtime image pipeline.
	tile := cmp.Or(
		p.Processor.ImageProcessor.TileSize,
		p.Processor.ImageProcessor.Size.Height,
		p.Processor.ImageProcessor.Size.Width,
		uint32(512),
	)
	downsample := cmp.Or(p.DownsampleFactor, p.Processor.ImageProcessor.DownsampleFactor, uint32(2))
	if downsample == 0 {
		return tile
	}

	return max(uint32(1), tile/downsample)
}

func (p *lfm2VLTextModel) KV(t *Tokenizer) KV {
	kv := p.textModel().KV(t)

	boolOr := func(defaultValue bool, values ...*bool) bool {
		for _, v := range values {
			if v != nil {
				return *v
			}
		}
		return defaultValue
	}

	kv["vision.block_count"] = cmp.Or(p.VisionConfig.NumHiddenLayers, uint32(27))
	kv["vision.embedding_length"] = cmp.Or(p.VisionConfig.HiddenSize, uint32(1152))
	kv["vision.feed_forward_length"] = cmp.Or(p.VisionConfig.IntermediateSize, uint32(4304))
	kv["vision.attention.head_count"] = cmp.Or(p.VisionConfig.NumAttentionHeads, uint32(16))
	kv["vision.attention.layer_norm_epsilon"] = cmp.Or(p.VisionConfig.LayerNormEpsilon, float32(1e-6))
	kv["vision.patch_size"] = cmp.Or(p.VisionConfig.PatchSize, p.EncoderPatchSize, uint32(16))
	kv["vision.num_channels"] = cmp.Or(p.VisionConfig.NumChannels, uint32(3))
	kv["vision.image_size"] = p.visionImageSize()
	kv["vision.projector.scale_factor"] = cmp.Or(p.DownsampleFactor, p.Processor.ImageProcessor.DownsampleFactor, uint32(2))
	kv["vision.projector.use_layernorm"] = p.ProjectorUseLayernorm
	kv["vision.do_image_splitting"] = boolOr(true, p.DoImageSplitting, p.Processor.ImageProcessor.DoImageSplitting)
	kv["vision.min_tiles"] = cmp.Or(p.MinTiles, p.Processor.ImageProcessor.MinTiles, uint32(2))
	kv["vision.max_tiles"] = cmp.Or(p.MaxTiles, p.Processor.ImageProcessor.MaxTiles, uint32(10))
	kv["vision.tile_size"] = cmp.Or(p.TileSize, p.Processor.ImageProcessor.TileSize, uint32(512))
	kv["vision.min_image_tokens"] = cmp.Or(p.MinImageTokens, p.Processor.ImageProcessor.MinImageTokens, uint32(64))
	kv["vision.max_image_tokens"] = cmp.Or(p.MaxImageTokens, p.Processor.ImageProcessor.MaxImageTokens, uint32(256))
	kv["vision.max_pixels_tolerance"] = cmp.Or(p.MaxPixelsTolerance, p.Processor.ImageProcessor.MaxPixelsTol, float32(2.0))
	kv["vision.use_thumbnail"] = boolOr(true, p.UseThumbnail, p.Processor.ImageProcessor.UseThumbnail)
	kv["vision.use_image_special_tokens"] = boolOr(true, p.UseImageSpecialTokens)
	kv["vision.image_mean"] = slices.Clone(defaultFloat32Slice(p.Processor.ImageProcessor.ImageMean, []float32{0.5, 0.5, 0.5}))
	kv["vision.image_std"] = slices.Clone(defaultFloat32Slice(p.Processor.ImageProcessor.ImageStd, []float32{0.5, 0.5, 0.5}))
	kv["vision.image_token_id"] = cmp.Or(p.ImageTokenID, uint32(396))

	setVisionTokenID := func(k, token string) {
		if t == nil || t.Vocabulary == nil {
			return
		}
		for i, v := range t.Vocabulary.Tokens {
			if v == token {
				kv[k] = uint32(i)
				return
			}
		}
	}
	setVisionTokenID("vision.image_start_token_id", "<|image_start|>")
	setVisionTokenID("vision.image_end_token_id", "<|image_end|>")
	setVisionTokenID("vision.image_thumbnail_token_id", "<|img_thumbnail|>")

	return kv
}

func (p *lfm2VLTextModel) Tensors(ts []Tensor) []*ggml.Tensor {
	patchSize := int(cmp.Or(p.VisionConfig.PatchSize, p.EncoderPatchSize, uint32(16)))
	numChannels := int(cmp.Or(p.VisionConfig.NumChannels, uint32(3)))

	for _, t := range ts {
		if t.Name() == "v.patch_embd.weight" {
			shape := t.Shape()
			if len(shape) == 2 {
				inputDim := uint64(numChannels * patchSize * patchSize)
				if shape[1] == inputDim {
					channels := numChannels
					patch := patchSize
					t.SetRepacker(func(_ string, data []float32, srcShape []uint64) ([]float32, error) {
						return repackPatchEmbeddingWeight(data, srcShape, channels, patch)
					})
				}
			}
		}
	}

	out := p.textModel().Tensors(ts)
	for _, t := range out {
		if t.Name == "v.patch_embd.weight" && len(t.Shape) == 2 {
			t.Shape = []uint64{t.Shape[0], uint64(numChannels), uint64(patchSize), uint64(patchSize)}
		}
	}
	return out
}

func (p *lfm2VLTextModel) Replacements() []string {
	out := make([]string, 0, 96)

	addText := func(from, to string) {
		out = append(out, from, to)
		if strings.HasPrefix(from, "model.") {
			suffix := strings.TrimPrefix(from, "model.")
			out = append(out,
				"model.language_model."+suffix, to,
				"model.language_model.model."+suffix, to,
			)
		}
	}

	base := p.textModel().Replacements()
	for i := 0; i+1 < len(base); i += 2 {
		addText(base[i], base[i+1])
	}

	// Vision tower + multimodal projector tensors (single-file conversion).
	out = append(out,
		"model.vision_tower.vision_model.embeddings.patch_embedding", "v.patch_embd",
		"model.vision_tower.vision_model.embeddings.position_embedding", "v.position_embd",
		"model.vision_tower.vision_model.encoder.layers", "v.blk",
		"model.vision_tower.vision_model.post_layernorm", "v.post_ln",
		"model.multi_modal_projector.layer_norm", "mm.layer_norm",
		"model.multi_modal_projector.linear_1", "mm.1",
		"model.multi_modal_projector.linear_2", "mm.2",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.out_proj", "attn_out",
		"layer_norm1", "ln1",
		"layer_norm2", "ln2",
		"mlp.fc1", "ffn_up",
		"mlp.fc2", "ffn_down",
	)

	return out
}

// lfm2VLProjectorModel converts the vision encoder + projector component of LFM2 VL checkpoints.
type lfm2VLProjectorModel struct {
	ModelParameters
	DownsampleFactor   uint32 `json:"downsample_factor"`
	ProjectorHiddenDim uint32 `json:"projector_hidden_size"`
	VisionModel        struct {
		HiddenSize        uint32  `json:"hidden_size"`
		IntermediateSize  uint32  `json:"intermediate_size"`
		NumAttentionHeads uint32  `json:"num_attention_heads"`
		NumHiddenLayers   uint32  `json:"num_hidden_layers"`
		NumChannels       uint32  `json:"num_channels"`
		PatchSize         uint32  `json:"patch_size"`
		LayerNormEpsilon  float32 `json:"layer_norm_eps"`
		ImageSize         uint32  `json:"image_size"`
	} `json:"vision_config"`
	Processor struct {
		ImageProcessor struct {
			DownsampleFactor uint32    `json:"downsample_factor"`
			TileSize         uint32    `json:"tile_size"`
			ImageMean        []float32 `json:"image_mean"`
			ImageStd         []float32 `json:"image_std"`
			Size             struct {
				Height uint32 `json:"height"`
				Width  uint32 `json:"width"`
			} `json:"size"`
		} `json:"image_processor"`
	}
}

var (
	_ ModelConverter = (*lfm2VLTextModel)(nil)
	_ ModelConverter = (*lfm2VLProjectorModel)(nil)
	_ moreParser     = (*lfm2VLTextModel)(nil)
	_ moreParser     = (*lfm2VLProjectorModel)(nil)
)

func (p *lfm2VLProjectorModel) parseMore(fsys fs.FS) error {
	bts, err := fs.ReadFile(fsys, "processor_config.json")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil
		}
		return err
	}

	return json.Unmarshal(bts, &p.Processor)
}

func (p *lfm2VLProjectorModel) imageSize() uint32 {
	if p.VisionModel.ImageSize > 0 {
		return p.VisionModel.ImageSize
	}

	downsample := cmp.Or(p.DownsampleFactor, p.Processor.ImageProcessor.DownsampleFactor, uint32(2))
	baseSize := cmp.Or(
		p.Processor.ImageProcessor.TileSize,
		p.Processor.ImageProcessor.Size.Height,
		p.Processor.ImageProcessor.Size.Width,
		uint32(256),
	)
	if downsample == 0 {
		return baseSize
	}

	return max(uint32(1), baseSize/downsample)
}

func (p *lfm2VLProjectorModel) KV(_ *Tokenizer) KV {
	kv := KV{
		"general.architecture":         "clip",
		"general.type":                 "mmproj",
		"general.file_type":            uint32(1),
		"general.quantization_version": uint32(2),
		"clip.has_vision_encoder":      true,
		"clip.projector_type":          "lfm2",
		"clip.use_gelu":                true,
	}

	kv["clip.vision.block_count"] = cmp.Or(p.VisionModel.NumHiddenLayers, uint32(27))
	kv["clip.vision.embedding_length"] = cmp.Or(p.VisionModel.HiddenSize, uint32(1152))
	kv["clip.vision.feed_forward_length"] = cmp.Or(p.VisionModel.IntermediateSize, uint32(4304))
	kv["clip.vision.attention.head_count"] = cmp.Or(p.VisionModel.NumAttentionHeads, uint32(16))
	kv["clip.vision.attention.layer_norm_epsilon"] = cmp.Or(p.VisionModel.LayerNormEpsilon, float32(1e-6))
	kv["clip.vision.patch_size"] = cmp.Or(p.VisionModel.PatchSize, uint32(16))
	kv["clip.vision.image_size"] = p.imageSize()
	kv["clip.vision.projection_dim"] = cmp.Or(p.ProjectorHiddenDim, uint32(2048))
	kv["clip.vision.projector.scale_factor"] = cmp.Or(p.DownsampleFactor, p.Processor.ImageProcessor.DownsampleFactor, uint32(2))
	kv["clip.vision.image_mean"] = slices.Clone(defaultFloat32Slice(p.Processor.ImageProcessor.ImageMean, []float32{0.5, 0.5, 0.5}))
	kv["clip.vision.image_std"] = slices.Clone(defaultFloat32Slice(p.Processor.ImageProcessor.ImageStd, []float32{0.5, 0.5, 0.5}))

	return kv
}

func defaultFloat32Slice(v, fallback []float32) []float32 {
	if len(v) > 0 {
		return v
	}

	return fallback
}

func (p *lfm2VLProjectorModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	numChannels := cmp.Or(p.VisionModel.NumChannels, uint32(3))
	patchSize := cmp.Or(p.VisionModel.PatchSize, uint32(16))

	for _, t := range ts {
		name := t.Name()
		if !(strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.")) {
			continue
		}

		shape := t.Shape()
		if name == "v.patch_embd.weight" && len(shape) == 2 {
			inputDim := uint64(numChannels * patchSize * patchSize)
			if shape[1] == inputDim {
				shape = []uint64{shape[0], uint64(numChannels), uint64(patchSize), uint64(patchSize)}
				channels := int(numChannels)
				patch := int(patchSize)
				t.SetRepacker(func(_ string, data []float32, srcShape []uint64) ([]float32, error) {
					return repackPatchEmbeddingWeight(data, srcShape, channels, patch)
				})
			}
		}

		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    slices.Clone(shape),
			WriterTo: t,
		})
	}

	return out
}

func (p *lfm2VLProjectorModel) Replacements() []string {
	return []string{
		"model.multi_modal_projector.linear_1", "mm.1",
		"model.multi_modal_projector.linear_2", "mm.2",
		"model.vision_tower.vision_model.embeddings.patch_embedding", "v.patch_embd",
		"model.vision_tower.vision_model.embeddings.position_embedding", "v.position_embd",
		"model.vision_tower.vision_model.encoder.layers", "v.blk",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.out_proj", "attn_out",
		"layer_norm1", "ln1",
		"layer_norm2", "ln2",
		"mlp.fc1", "ffn_up",
		"mlp.fc2", "ffn_down",
		"model.vision_tower.vision_model.post_layernorm", "v.post_ln",
	}
}

func repackPatchEmbeddingWeight(data []float32, srcShape []uint64, channels, patch int) ([]float32, error) {
	if len(srcShape) != 2 {
		return nil, fmt.Errorf("invalid patch embedding shape rank: %d", len(srcShape))
	}

	outDim := int(srcShape[0])
	flatInputDim := int(srcShape[1])
	expectedInputDim := channels * patch * patch
	if flatInputDim != expectedInputDim {
		return nil, fmt.Errorf("invalid patch embedding input dim: got %d, want %d", flatInputDim, expectedInputDim)
	}

	expectedSize := outDim * flatInputDim
	if len(data) != expectedSize {
		return nil, fmt.Errorf("invalid patch embedding data size: got %d, want %d", len(data), expectedSize)
	}

	repacked := make([]float32, len(data))
	perChannel := patch * patch

	for o := range outDim {
		inBase := o * flatInputDim
		outBase := o * flatInputDim

		for y := range patch {
			for x := range patch {
				inPixelBase := inBase + (y*patch+x)*channels
				for c := 0; c < channels; c++ {
					src := inPixelBase + c
					dst := outBase + c*perChannel + y*patch + x
					repacked[dst] = data[src]
				}
			}
		}
	}

	return repacked, nil
}
