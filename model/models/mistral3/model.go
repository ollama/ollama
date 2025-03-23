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
		1036, // TODO (jmorganca): this should be returned from ProcessImage
		m.ImageProcessor.numChannels,
	)
	if err != nil {
		return nil, err
	}

	// fmt.Println("pixelValues", "shape", pixelValues.Shape(), "data", ml.Dump(ctx, pixelValues))

	// Forward pass through vision model
	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)

	// fmt.Println("visionOutputs", "shape", visionOutputs.Shape(), "data", ml.Dump(ctx, visionOutputs))

	// Project to text embedding space
	visionOutputs = m.MultiModalProjector.Forward(ctx, visionOutputs, m.VisionModel.eps)

	// fmt.Println("visionOutputs after projector", "shape", visionOutputs.Shape(), "data", ml.Dump(ctx, visionOutputs))

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
			result = append(result, input.Input{Token: 10})                                                       // [IMG]
			result = append(result, input.Input{Multimodal: inputMultimodal, MultimodalHash: inp.MultimodalHash}) // image data
			result = append(result, slices.Repeat([]input.Input{{Token: 10}}, inputMultimodal.Dim(1)-1)...)       // [IMG] placeholders
			result = append(result, input.Input{Token: 13})                                                       // [IMG_END]
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
