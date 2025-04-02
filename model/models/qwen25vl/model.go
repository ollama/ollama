package qwen25vl

import (
	"bytes"
	"fmt"
	"image"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	*TextModel
	*VisionModel `gguf:"v,vision"`
	*PatchMerger `gguf:"mm"`

	ImageProcessor
}

// Implement MultimodalProcessor interface
var _ model.MultimodalProcessor = (*Model)(nil)

type PatchMerger struct {
	MLPLayer1 *nn.Linear `gguf:"0"`
	MLPLayer2 *nn.Linear `gguf:"2"`
}

// Forward computes patch merging for the vision model
func (pm *PatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, eps float32) ml.Tensor {
	// Get dimensions
	hiddenSize := visionOutputs.Dim(0)
	numPositions := visionOutputs.Dim(1)
	batchSize := visionOutputs.Dim(2)

	reshaped := visionOutputs.Reshape(ctx, hiddenSize*4, numPositions/4, batchSize)

	// Apply first linear layer (mm_0_w, mm_0_b)
	hidden := pm.MLPLayer1.Forward(ctx, reshaped)

	activated := hidden.GELU(ctx)

	// Apply second linear layer (mm_1_w, mm_1_b)
	output := pm.MLPLayer2.Forward(ctx, activated)

	return output
}

func New(c fs.Config) (model.Model, error) {
	m := &Model{
		TextModel:      NewTextModel(c),
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

type imageFeatures struct {
	Tensor ml.Tensor
	GridT  int
	GridH  int
	GridW  int
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, gridT, gridH, gridW, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	// Calculate tensor dimensions
	patchDim := m.ImageProcessor.numChannels * m.ImageProcessor.temporalPatchSize *
		m.ImageProcessor.patchSize * m.ImageProcessor.patchSize
	numPatches := gridT * gridH * gridW

	pixelValues, err := ctx.Input().FromFloatSlice(f32s, patchDim, numPatches)
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor from image: %w", err)
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	visionOutputs = m.PatchMerger.Forward(ctx, visionOutputs, m.VisionModel.eps)

	return &imageFeatures{
		Tensor: visionOutputs,
		GridT:  gridT,
		GridH:  gridH,
		GridW:  gridW,
	}, nil
}

// PostTokenize arranges Qwen-2.5-VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input

	// Get image token IDs from config
	imageToken := 151655
	visionStartToken := 151652
	visionEndToken := 151653

	// Get merge size from config
	mergeSize := m.ImageProcessor.mergeSize

	for _, inp := range inputs {
		if inp.Multimodal == nil {
			// If not a multimodal input, add it to the result unchanged
			result = append(result, inp)
		} else {
			// This is an image token with multimodal data
			features := inp.Multimodal.(*imageFeatures)

			// Get grid dimensions from the features
			gridT := features.GridT
			gridH := features.GridH
			gridW := features.GridW

			// Calculate tokens per grid based on grid dimensions
			mergeLength := mergeSize * mergeSize
			gridProduct := gridT * gridH * gridW
			tokensPerGrid := gridProduct / mergeLength

			// First add the vision start token
			result = append(result, input.Input{Token: int32(visionStartToken)})

			// Add the image token with the multimodal tensor data at the first position
			result = append(result, input.Input{
				Token:          int32(imageToken),
				Multimodal:     features.Tensor,
				MultimodalHash: inp.MultimodalHash,
			})

			// Add the placeholder tokens for the remaining positions (tokensPerGrid-1)
			for range tokensPerGrid - 1 {
				result = append(result, input.Input{Token: int32(imageToken)})
			}

			result = append(result, input.Input{Token: int32(visionEndToken)})
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
	model.Register("qwen2vl", New)
}
