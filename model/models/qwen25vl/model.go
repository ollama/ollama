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
	model.BytePairEncoding

	*TextModel
	*VisionModel `gguf:"v"`

	ImageProcessor
}

// Implement MultimodalProcessor interface
var _ model.MultimodalProcessor = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	m := &Model{
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		TextModel:      NewTextModel(c),
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

func (m *Model) PixelValues(ctx ml.Context, multimodalData []byte) (ml.Tensor, *Grid, error) {
	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, nil, err
	}

	f32s, grid, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, nil, err
	}

	// Calculate tensor dimensions
	patchDim := m.ImageProcessor.numChannels * m.ImageProcessor.temporalPatchSize *
		m.ImageProcessor.patchSize * m.ImageProcessor.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width

	pixelValues := ctx.Input().FromFloats(f32s, patchDim, numPatches)

	return pixelValues, grid, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	pixels, grid, err := m.PixelValues(ctx, multimodalData)
	if err != nil {
		return nil, err
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixels, grid)
	return []input.Multimodal{{Tensor: visionOutputs}}, nil
}

// PostTokenize arranges Qwen-2.5-VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	var (
		imageToken       int32 = 151655
		visionStartToken int32 = 151652
		visionEndToken   int32 = 151653
	)

	nImg := 0
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			// If not a multimodal input, add it to the result unchanged
			result = append(result, inp)
		} else {
			// Adding the 'Picture' prefix is a hack, at the time of writing there is no way to prefix
			// the image tokens with a prompt, so we add a prefix here
			nImg++
			pre, err := m.Encode(fmt.Sprintf(" Picture %d: ", nImg), true)
			if err != nil {
				return nil, fmt.Errorf("failed to encode image prompt: %w", err)
			}
			for i := range pre {
				result = append(result, &input.Input{Token: pre[i]})
			}

			patchesPerChunk := inp.Multimodal[0].Tensor.Dim(1)

			// First add the vision start token
			result = append(result, &input.Input{Token: visionStartToken})

			// Add the image token with the multimodal tensor data at the first position
			result = append(result, &input.Input{
				Token:          imageToken,
				Multimodal:     inp.Multimodal,
				MultimodalHash: inp.MultimodalHash,
				SameBatch:      patchesPerChunk,
			})

			// Add the placeholder tokens for the remaining positions (tokensPerGrid-1)
			result = append(result, slices.Repeat([]*input.Input{{Token: imageToken}}, patchesPerChunk-1)...)

			result = append(result, &input.Input{Token: visionEndToken})
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	return m.TextModel.Forward(ctx, batch.Inputs, positions, batch.Outputs, batch, m.Cache)
}

func init() {
	model.Register("qwen25vl", New)
}
