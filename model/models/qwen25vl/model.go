package qwen25vl

import (
	"bytes"
	"fmt"
	"image"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	*TextModel
	*VisionModel `gguf:"v,vision"`

	ImageProcessor
}

// Implement MultimodalProcessor interface
var _ model.MultimodalProcessor = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	m := &Model{
		TextModel:      NewTextModel(c),
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
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

	f32s, grid, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	// Calculate tensor dimensions
	patchDim := m.ImageProcessor.numChannels * m.ImageProcessor.temporalPatchSize *
		m.ImageProcessor.patchSize * m.ImageProcessor.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width

	pixelValues, err := ctx.Input().FromFloatSlice(f32s, patchDim, numPatches)
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor from image: %w", err)
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues, grid)
	return visionOutputs, nil
}

// PostTokenize arranges Qwen-2.5-VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input

	// Get image token IDs from config
	var (
		imageToken       int32 = 151655
		visionStartToken int32 = 151652
		visionEndToken   int32 = 151653
	)

	for _, inp := range inputs {
		if inp.Multimodal == nil {
			// If not a multimodal input, add it to the result unchanged
			result = append(result, inp)
		} else {
			// This is an image token with multimodal data
			visionOutputs := inp.Multimodal.(ml.Tensor)

			// Calculate tokens per grid based on grid dimensions

			// First add the vision start token
			result = append(result, input.Input{Token: visionStartToken, SameBatch: visionOutputs.Dim(1) + 2})

			// Add the image token with the multimodal tensor data at the first position
			result = append(result, input.Input{Token: imageToken, Multimodal: visionOutputs, MultimodalHash: inp.MultimodalHash})

			// Add the placeholder tokens for the remaining positions (tokensPerGrid-1)
			result = append(result, slices.Repeat([]input.Input{{Token: imageToken}}, visionOutputs.Dim(1)-1)...)

			result = append(result, input.Input{Token: visionEndToken})
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

	return m.TextModel.Forward(ctx, batch.Inputs, positions, outputs, batch, m.Cache)
}

func init() {
	model.Register("qwen25vl", New)
	model.Register("qwen2", New)
	model.Register("qwen2vl", New)
}
