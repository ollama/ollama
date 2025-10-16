package qwen3vl

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
	model.TextProcessor

	*TextModel
	*VisionModel `gguf:"v"`

	ImageProcessor
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	pixelValues, grid, err := m.ProcessImage(ctx, img)
	if err != nil {
		return nil, err
	}

	// Calculate tensor dimensions
	visionOutputs, deepstackVisualEmbeds := m.VisionModel.Forward(ctx, pixelValues, grid)
	mm := []input.Multimodal{{Tensor: visionOutputs}}
	for i := range deepstackVisualEmbeds {
		mm = append(mm, input.Multimodal{Tensor: deepstackVisualEmbeds[i]})
	}

	return mm, nil
}

// PostTokenize arranges Qwen 3 VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	var (
		imageToken       int32 = 151655
		visionStartToken int32 = 151652
		visionEndToken   int32 = 151653
	)

	var n int
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			// If not a multimodal input, add it to the result unchanged
			result = append(result, inp)
		} else {
			n++
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
	hiddenStates := m.TextModel.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)

	var deepstackVisualEmbeds []ml.Tensor
	for _, mi := range batch.Multimodal {
		visionOutputs := mi.Multimodal[0].Tensor
		for _, mm := range mi.Multimodal[1:] {
			deepstackVisualEmbeds = append(deepstackVisualEmbeds, mm.Tensor)
		}

		indices := ctx.Input().Arange(float32(mi.Index), float32(mi.Index+visionOutputs.Dim(1)), 1, ml.DTypeI32)
		ctx.Forward(hiddenStates.SetRows(ctx, visionOutputs, indices))
	}

	positionIDs := make([]int32, len(batch.Positions)*4)
	for i, id := range batch.Positions {
		positionIDs[i+len(batch.Positions)*0] = id
		positionIDs[i+len(batch.Positions)*1] = id
		positionIDs[i+len(batch.Positions)*2] = id
		positionIDs[i+len(batch.Positions)*3] = 0
	}

	positions := ctx.Input().FromIntSlice(positionIDs, len(positionIDs))
	for i, layer := range m.TextModel.Layers {
		if m.Cache != nil {
			m.Cache.SetLayer(i)
		}

		var outputs ml.Tensor
		if i == len(m.TextModel.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.TextModel.Cache, m.TextModel.Options)
		if j := slices.Index(m.deepstackVisualIndexes, int32(i)); len(deepstackVisualEmbeds) > 0 && j >= 0 {
			visualEmbeds := deepstackVisualEmbeds[j].Pad(ctx, 0, hiddenStates.Dim(1)-deepstackVisualEmbeds[j].Dim(1), 0, 0)
			hiddenStates = hiddenStates.Add(ctx, visualEmbeds)
		}
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, 1e-06)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func New(c fs.Config) (model.Model, error) {
	m := Model{
		TextProcessor: model.NewBytePairEncoding(
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
		TextModel:      newTextModel(c),
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

func init() {
	model.Register("qwen3vl", New)
	model.Register("qwen3vlmoe", New)
}
