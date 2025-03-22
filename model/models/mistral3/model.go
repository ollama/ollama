package mistral3

import (
	"bytes"
	"image"
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	*TextModel
	*VisionModel         `gguf:"v,vision"`
	*MultiModalProjector `gguf:"mm"`

	ImageProcessor
}

// Implement MultimodalProcessor interface
var _ model.MultimodalProcessor = (*Model)(nil)

func New(c ml.Config) (model.Model, error) {
	textModel, err := NewTextModel(c)
	if err != nil {
		return nil, err
	}

	m := &Model{
		TextModel:           textModel,
		VisionModel:         newVisionModel(c),
		ImageProcessor:      newImageProcessor(c),
		MultiModalProjector: newMultiModalProjector(c),
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	// Create tensor from image data
	pixelValues, err := ctx.Input().FromFloatSlice(f32s,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.numChannels,
	)
	if err != nil {
		return nil, err
	}

	// Forward pass through vision model
	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)

	// Project to text embedding space
	visionOutputs = m.MultiModalProjector.Forward(ctx, visionOutputs, m.VisionModel.eps)

	return visionOutputs, nil
}

func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input

	for _, inp := range inputs {
		if inp.Multimodal == nil {
			result = append(result, inp)
		} else {
			inputMultimodal := inp.Multimodal.(ml.Tensor)

			// Add special image tokens - using the imageTokenIndex from config
			result = append(result,
				input.Input{Token: int32(m.MultiModalProjector.imageTokenIndex)},             // Image token
				input.Input{Multimodal: inputMultimodal, MultimodalHash: inp.MultimodalHash}, // Image data
			)

			// Add image token placeholders
			result = append(result, slices.Repeat([]input.Input{{Token: 0}}, inputMultimodal.Dim(1)-1)...)
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions, err := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
	if err != nil {
		return nil, err
	}

	return m.TextModel.Forward(ctx, batch.Inputs, positions, outputs, batch, m.Cache), nil
}

func init() {
	model.Register("mistral3", New)
}
