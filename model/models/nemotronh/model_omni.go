package nemotronh

import (
	"bytes"
	"errors"
	"image"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type OmniModel struct {
	*Model
	*VisionModel         `gguf:"v"`
	*AudioModel          `gguf:"a"`
	*MultiModalProjector `gguf:"mm"`
	*AudioProjector      `gguf:"mm.a"`

	ImageProcessor

	imageTokenID    int32
	imageStartToken int32
	imageEndToken   int32
	audioTokenID    int32
}

var _ model.MultimodalProcessor = (*OmniModel)(nil)

func NewOmni(c fs.Config) (model.Model, error) {
	textModel, err := newTextModel(c)
	if err != nil {
		return nil, err
	}

	imageTokenID := int32(c.Uint("vision.image_token_id", 18))
	imageStartToken := int32(c.Uint("vision.image_start_token_id", 19))
	imageEndToken := int32(c.Uint("vision.image_end_token_id", 20))
	audioTokenID := int32(c.Uint("audio.sound_token_id", 27))

	return &OmniModel{
		Model:               textModel,
		VisionModel:         newVisionModel(c),
		AudioModel:          newAudioModel(c),
		MultiModalProjector: newMultiModalProjector(c),
		AudioProjector:      newAudioProjector(c),
		ImageProcessor:      newImageProcessor(c),
		imageTokenID:        imageTokenID,
		imageStartToken:     imageStartToken,
		imageEndToken:       imageEndToken,
		audioTokenID:        audioTokenID,
	}, nil
}

func (m *OmniModel) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if isAudioData(multimodalData) {
		return m.encodeAudioMultimodal(ctx, multimodalData)
	}

	if m.VisionModel == nil || m.MultiModalProjector == nil || len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	tiles, err := m.ImageProcessor.ProcessImage(img)
	if err != nil {
		return nil, err
	}

	mm := make([]input.Multimodal, 0, len(tiles))
	for _, tile := range tiles {
		patches := visionPatchGrid{
			Width:  tile.size.X / m.ImageProcessor.patchSize,
			Height: tile.size.Y / m.ImageProcessor.patchSize,
		}
		if patches.Width == 0 || patches.Height == 0 {
			return nil, errors.New("nemotron_h_omni: invalid resized image dimensions")
		}

		patchInput := packVisionPatchesCHW(tile.data, tile.size.X, tile.size.Y, m.ImageProcessor.numChannels, m.ImageProcessor.patchSize)
		visionOutputs := m.VisionModel.ForwardPacked(ctx, patchInput, patches)
		projected := m.MultiModalProjector.Forward(ctx, visionOutputs, patches)
		mm = append(mm, input.Multimodal{Tensor: projected})
	}

	return mm, nil
}

type audioTag struct{}

func (m *OmniModel) encodeAudioMultimodal(ctx ml.Context, data []byte) ([]input.Multimodal, error) {
	if m.AudioModel == nil || m.AudioProjector == nil || len(m.AudioModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	samples, err := decodeWAV(data, m.AudioModel.sampleRate)
	if err != nil {
		return nil, err
	}

	melData, frames, validFrames, err := computeParakeetMelSpectrogram(samples, m.AudioModel.FeatureExtractor, m.AudioModel.AudioOptions)
	if err != nil {
		return nil, err
	}

	melTensor := ctx.Input().FromFloats(melData, m.AudioModel.melBins, frames)
	audioOutputs := m.AudioModel.ForwardAudio(ctx, melTensor, validFrames, m.AudioProjector)
	return []input.Multimodal{{Tensor: audioOutputs, Data: audioTag{}}}, nil
}

func (m *OmniModel) PostLoad() error {
	return nil
}

func (m *OmniModel) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	imageToken := m.imageTokenID
	if imageToken == 0 {
		imageToken = 18
	}

	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
			continue
		}

		totalTokens := 0
		for _, mm := range inp.Multimodal {
			if mm.Tensor == nil {
				continue
			}
			totalTokens += mm.Tensor.Dim(1)
		}
		if totalTokens <= 0 {
			return nil, errors.New("nemotron_h_omni: multimodal input has no tokens")
		}

		if _, ok := inp.Multimodal[0].Data.(audioTag); ok {
			audioToken := m.audioTokenID
			if audioToken == 0 {
				audioToken = 27
			}

			for i, mm := range inp.Multimodal {
				tokenCount := 0
				if mm.Tensor != nil {
					tokenCount = mm.Tensor.Dim(1)
				}
				if tokenCount <= 0 {
					return nil, errors.New("nemotron_h_omni: multimodal input has no tokens")
				}

				first := &input.Input{Token: audioToken, SameBatch: tokenCount - 1}
				if i == 0 {
					first.MultimodalHash = inp.MultimodalHash
				}
				first.Multimodal = []input.Multimodal{mm}
				result = append(result, first)
				if tokenCount > 1 {
					result = append(result, slices.Repeat([]*input.Input{{Token: audioToken}}, tokenCount-1)...)
				}
			}
			continue
		}

		if m.imageStartToken > 0 {
			result = append(result, &input.Input{
				Token:     m.imageStartToken,
				SameBatch: totalTokens + btoi(m.imageEndToken > 0),
			})
		}

		for _, mm := range inp.Multimodal {
			tokenCount := 0
			if mm.Tensor != nil {
				tokenCount = mm.Tensor.Dim(1)
			}
			if tokenCount <= 0 {
				return nil, errors.New("nemotron_h_omni: multimodal input has no tokens")
			}

			result = append(result, &input.Input{
				Token:          imageToken,
				Multimodal:     []input.Multimodal{mm},
				MultimodalHash: inp.MultimodalHash,
			})
			if tokenCount > 1 {
				result = append(result, slices.Repeat([]*input.Input{{Token: imageToken}}, tokenCount-1)...)
			}
		}

		if m.imageEndToken > 0 {
			result = append(result, &input.Input{Token: m.imageEndToken})
		}
	}

	return result, nil
}

func btoi(v bool) int {
	if v {
		return 1
	}
	return 0
}

func (m *OmniModel) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	if len(batch.Multimodal) > 0 {
		hiddenStates = hiddenStates.Duplicate(ctx)
	}

	for _, mm := range batch.Multimodal {
		offset := mm.Index
		for _, multimodal := range mm.Multimodal {
			if multimodal.Tensor == nil {
				continue
			}

			tensor := multimodal.Tensor
			ctx.Forward(tensor.Copy(ctx, hiddenStates.View(ctx, offset*hiddenStates.Stride(1), tensor.Dim(0)*tensor.Dim(1))))
			offset += tensor.Dim(1)
		}
	}

	return m.forwardLogits(ctx, batch, hiddenStates)
}

func init() {
	model.Register("nemotron_h_omni", NewOmni)
}
