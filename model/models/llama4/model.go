package llama4

import (
	"bytes"
	"image"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.BytePairEncoding
	ImageProcessor

	*VisionModel `gguf:"v,vision"`
	*Projector   `gguf:"mm"`
	*TextModel
}

type Projector struct {
	Linear1 *nn.Linear `gguf:"linear_1"`
}

func (p *Projector) Forward(ctx ml.Context, visionOutputs ml.Tensor) ml.Tensor {
	return p.Linear1.Forward(ctx, visionOutputs)
}

func New(c fs.Config) (model.Model, error) {
	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			c.String("tokenizer.ggml.pretokenizer",
				`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
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
		),
		ImageProcessor: newImageProcessor(c),
		VisionModel:    newVisionModel(c),
		TextModel:      newTextModel(c),
	}

	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewChunkedAttentionCache(int32(c.Uint("attention.chunk_size", 8192)), m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) < 1 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	pixelsLocal, pixelsGlobal, size, err := m.ProcessImage(img)
	if err != nil {
		return nil, err
	}

	tilesLocal := ctx.Input().FromFloatSlice(pixelsLocal, size.X, size.Y, m.numChannels)

	ratioW, ratioH := size.X/m.imageSize, size.Y/m.imageSize

	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW, ratioW, size.Y, m.numChannels).Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW*size.Y/ratioH, ratioH, ratioW, m.numChannels).Permute(ctx, 0, 3, 2, 1).Contiguous(ctx)
	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW, size.Y/ratioH, m.numChannels, ratioH*ratioW)

	pixelValues := tilesLocal

	if len(pixelsGlobal) > 0 {
		tilesGlobal := ctx.Input().FromFloatSlice(pixelsGlobal, m.imageSize, m.imageSize, m.numChannels)
		pixelValues = pixelValues.Concat(ctx, tilesGlobal, 3)
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	visionOutputs = visionOutputs.Reshape(ctx, visionOutputs.Dim(0), visionOutputs.Dim(1)*visionOutputs.Dim(2)*visionOutputs.Dim(3))
	projectedOutputs := m.Projector.Forward(ctx, visionOutputs)

	var multimodal []input.Multimodal
	aspectRatio := image.Point{ratioW, ratioH}

	var offset int
	patchesPerChunk := projectedOutputs.Dim(1)
	if aspectRatio.Y*aspectRatio.X > 1 {
		patchesPerChunk = projectedOutputs.Dim(1) / (aspectRatio.X*aspectRatio.Y + 1)

		for range aspectRatio.Y {
			for x := range aspectRatio.X {
				view := projectedOutputs.View(ctx, projectedOutputs.Stride(1)*offset,
					projectedOutputs.Dim(0), projectedOutputs.Stride(1),
					patchesPerChunk)
				var separator separator
				if x < aspectRatio.X-1 {
					separator.x = true // <|tile_x_separator|>
				} else {
					separator.y = true // <|tile_y_separator|>
				}
				multimodal = append(multimodal, input.Multimodal{Tensor: view, Data: &separator})
				offset += patchesPerChunk
			}
		}
	}

	view := projectedOutputs.View(ctx, projectedOutputs.Stride(1)*offset,
		projectedOutputs.Dim(0), projectedOutputs.Stride(1),
		patchesPerChunk)
	multimodal = append(multimodal, input.Multimodal{Tensor: view, Data: &separator{}})

	return multimodal, nil
}

type separator struct {
	x bool
	y bool
}

func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input
	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
			continue
		}

		var imageInputs []input.Input
		imageInputs = append(imageInputs, input.Input{Token: 200080}) // <|image_start|>

		for i, mm := range inp.Multimodal {
			patchesPerChunk := mm.Tensor.Dim(1)

			if i < len(inp.Multimodal)-1 {
				separator := mm.Data.(*separator)

				imageInputs = append(imageInputs, input.Input{Token: 200092, Multimodal: []input.Multimodal{{Tensor: mm.Tensor}}, MultimodalHash: inp.MultimodalHash, SameBatch: patchesPerChunk}) // <|patch|>
				imageInputs = append(imageInputs, slices.Repeat([]input.Input{{Token: 200092}}, patchesPerChunk-1)...)

				if separator.x {
					imageInputs = append(imageInputs, input.Input{Token: 200084}) // <|tile_x_separator|>
				}
				if separator.y {
					imageInputs = append(imageInputs, input.Input{Token: 200085}) // <|tile_y_separator|>
				}
			} else {
				imageInputs = append(imageInputs, input.Input{Token: 200090})                                                                                                                      // <|image|>
				imageInputs = append(imageInputs, input.Input{Token: 200092, Multimodal: []input.Multimodal{{Tensor: mm.Tensor}}, MultimodalHash: inp.MultimodalHash, SameBatch: patchesPerChunk}) // <|patch|>
				imageInputs = append(imageInputs, slices.Repeat([]input.Input{{Token: 200092}}, patchesPerChunk-1)...)
				imageInputs = append(imageInputs, input.Input{Token: 200080}) // <|image_end|>
			}
		}

		result = append(result, imageInputs...)
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))
	outputs := ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))

	return m.TextModel.Forward(ctx, batch.Inputs, positions, outputs, batch, m.Cache), nil
}

func init() {
	model.Register("llama4", New)
}
