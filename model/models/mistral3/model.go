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

	f32s, size, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	pixelValues, err := ctx.Input().FromFloatSlice(f32s, size.X, size.Y, m.ImageProcessor.numChannels)
	if err != nil {
		return nil, err
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	features, size := m.MultiModalProjector.Forward(ctx, visionOutputs, size)

	// split into patches to be sent to the text transformer
	rows := make([]ml.Tensor, size.Y)
	for i := range rows {
		rows[i] = features.View(ctx, features.Stride(1)*(i+size.X), features.Dim(0), features.Stride(1), size.X)
	}

	return rows, nil
}

// PostTokenize arranges Mistral 3's inputs for the forward pass
// In Mistral 3 and Pixtral, the input patches are arranged as follows:
// [IMG]...[IMG][IMG_BREAK][IMG]...[IMG][IMG_BREAK][IMG]...[IMG][IMG_END]
// Each sequence of [IMG]...[IMG] is a set of patches of vision embeddings
// that can be processed together.
func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input

	for _, inp := range inputs {
		if inp.Multimodal == nil {
			result = append(result, inp)
		} else {
			inputMultimodal := inp.Multimodal.([]ml.Tensor)
			for i, row := range inputMultimodal {
				result = append(result, input.Input{Token: 10, Multimodal: row, MultimodalHash: inp.MultimodalHash, SameBatch: row.Dim(1)}) // Image data
				result = append(result, slices.Repeat([]input.Input{{Token: 10}}, row.Dim(1)-1)...)                                         // [IMG]
				if i == len(inputMultimodal)-1 {
					result = append(result, input.Input{Token: 13}) // [IMG_END]
				} else {
					result = append(result, input.Input{Token: 12}) // [IMG_BREAK]
				}
			}
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
