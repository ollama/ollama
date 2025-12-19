package qwen25vl

import (
	"bytes"
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
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
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
	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, nil, err
	}

	f32s, grid, err := m.ImageProcessor.ProcessImage(img)
	if err != nil {
		return nil, nil, err
	}

	// Calculate tensor dimensions
	patchDim := m.numChannels * m.temporalPatchSize * m.patchSize * m.patchSize
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
	return []input.Multimodal{{Tensor: visionOutputs, Data: grid}}, nil
}

// PostTokenize arranges Qwen-2.5-VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	// Reset position cache
	m.positionCache = m.positionCache[:0]
	var result []*input.Input

	var (
		imageToken       int32 = 151655
		visionStartToken int32 = 151652
		visionEndToken   int32 = 151653
	)

	appendInput := func(i *input.Input, p int) int {
		result = append(result, i)
		m.positionCache = append(m.positionCache, int32(p))
		return p + 1
	}

	var p int
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			// If not a multimodal input, add it to the result unchanged
			p = appendInput(inp, p)
		} else {
			// First add the vision start token
			p = appendInput(&input.Input{Token: visionStartToken}, p)

			// Add the image token with the multimodal tensor data at the first position
			tokensPerGrid := inp.Multimodal[0].Tensor.Dim(1)
			appendInput(&input.Input{
				Token:          imageToken,
				Multimodal:     inp.Multimodal,
				MultimodalHash: inp.MultimodalHash,
				SameBatch:      tokensPerGrid,
			}, p)

			// Add the placeholder tokens for the remaining positions (tokensPerGrid-1)
			for range tokensPerGrid - 1 {
				appendInput(&input.Input{Token: imageToken}, p)
			}

			grid := inp.Multimodal[0].Data.(*Grid)
			p = appendInput(&input.Input{Token: visionEndToken}, p+max(grid.Width/m.spatialMergeSize, grid.Height/m.spatialMergeSize))
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	// Initial token embedding
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

	positionSlice := func() [][]int32 {
		s := [][]int32{
			make([]int32, len(batch.Positions)),
			make([]int32, len(batch.Positions)),
			make([]int32, len(batch.Positions)),
			make([]int32, len(batch.Positions)),
		}
		for i, position := range batch.Positions {
			if position < int32(len(m.positionCache)) {
				position = m.positionCache[position]
			} else if len(m.positionCache) > 0 {
				position = position - int32(len(m.positionCache)) + m.positionCache[len(m.positionCache)-1] + 1
			}

			s[0][i] = position
			s[1][i] = position
			s[2][i] = position
		}
		return s
	}()

	for _, mi := range batch.Multimodal {
		img := mi.Multimodal[0].Tensor
		ctx.Forward(img.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), img.Dim(0)*img.Dim(1))))
		if grid, ok := mi.Multimodal[0].Data.(*Grid); ok {
			for i := range img.Dim(1) {
				w := grid.Width / m.spatialMergeSize
				positionSlice[1][mi.Index+i] += int32(i / w)
				positionSlice[2][mi.Index+i] += int32(i % w)
			}
		}
	}

	positions := ctx.Input().FromInts(slices.Concat(positionSlice...), len(positionSlice[0])*len(positionSlice))

	// Process through transformer layers
	for i, layer := range m.TextModel.Layers {
		m.Cache.SetLayer(i)

		var lastLayerOutputs ml.Tensor
		if i == len(m.TextModel.Layers)-1 {
			lastLayerOutputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, lastLayerOutputs, m.Cache, m.TextOptions)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.TextModel.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("qwen25vl", New)
}
