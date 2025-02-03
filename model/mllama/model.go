package mllama

import (
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Model struct {
	model.Base

	*VisionModel `gguf:"v,vision"`
	*TextModel

	Projector *nn.Linear `gguf:"mm.0"`

	ImageProcessor
	TextProcessor
}

func New(c ml.Config) (model.Model, error) {
	return &Model{
		ImageProcessor: newImageProcessor(c),
		VisionModel:    newVisionModel(c),
		TextProcessor:  newTextProcessor(c),
		TextModel:      newTextModel(c),
	}, nil
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	var crossAttentionStates ml.Tensor
	if opts.Images != nil {
		f32s, aspectRatioID, err := m.ImageProcessor.ProcessImage(opts.Images[0])
		if err != nil {
			return nil, err
		}

		pixelValues, err := ctx.FromFloatSlice(f32s,
			m.ImageProcessor.imageSize,
			m.ImageProcessor.imageSize,
			m.ImageProcessor.numChannels,
			m.ImageProcessor.maxNumTiles,
		)
		if err != nil {
			return nil, err
		}

		aspectRatio, err := ctx.FromIntSlice([]int32{int32(aspectRatioID)}, 1)
		if err != nil {
			return nil, err
		}

		positions := make([]int32, 1601)
		for i := range positions {
			positions[i] = int32(i)
		}

		positionIDs, err := ctx.FromIntSlice(positions, len(positions))
		if err != nil {
			return nil, err
		}

		crossAttentionStates = m.VisionModel.Forward(ctx, pixelValues, positionIDs, aspectRatio)
		crossAttentionStates = m.Projector.Forward(ctx, crossAttentionStates)
	}

	inputs, err := ctx.FromIntSlice(opts.Inputs(), len(opts.Inputs()))
	if err != nil {
		return nil, err
	}

	positions, err := ctx.FromIntSlice(opts.Positions(), len(opts.Positions()))
	if err != nil {
		return nil, err
	}

	// TODO: attention mask, cross attention mask
	hiddenState := m.TextModel.Forward(ctx, inputs, positions, nil, crossAttentionStates, nil, opts.Cache)

	outputs, err := ctx.FromIntSlice([]int32{int32(len(opts.Positions())) - 1}, 1)
	if err != nil {
		return nil, err
	}

	return hiddenState.Rows(ctx, outputs), nil
}

func init() {
	model.Register("mllama", New)
}
