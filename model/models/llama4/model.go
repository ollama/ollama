package llama4

import (
	"bytes"
	"image"
	"slices"
	"sync"

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
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
			},
		),
		ImageProcessor: newImageProcessor(c),
		VisionModel:    newVisionModel(c),
		TextModel:      newTextModel(c),
	}

	m.Cache = kvcache.NewWrapperCache(
		// TODO: pretend this is chunked attention for now
		kvcache.NewSWACache(8192, m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
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

	tilesLocal, err := ctx.Input().FromFloatSlice(pixelsLocal, size.X, size.Y, m.numChannels)
	if err != nil {
		return nil, err
	}

	ratioW, ratioH := size.X/m.imageSize, size.Y/m.imageSize

	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW, ratioW, size.Y, m.numChannels).Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW*size.Y/ratioH, ratioH, ratioW, m.numChannels).Permute(ctx, 0, 3, 2, 1).Contiguous(ctx)
	tilesLocal = tilesLocal.Reshape(ctx, size.X/ratioW, size.Y/ratioH, m.numChannels, ratioH*ratioW)

	pixelValues := tilesLocal

	if len(pixelsGlobal) > 0 {
		tilesGlobal, err := ctx.Input().FromFloatSlice(pixelsGlobal, m.imageSize, m.imageSize, m.numChannels)
		if err != nil {
			return nil, err
		}

		pixelValues = pixelValues.Concat(ctx, tilesGlobal, 3)
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	visionOutputs = visionOutputs.Reshape(ctx, visionOutputs.Dim(0), visionOutputs.Dim(1)*visionOutputs.Dim(2)*visionOutputs.Dim(3))
	projectedOutputs := m.Projector.Forward(ctx, visionOutputs)
	return &chunks{Model: m, Tensor: projectedOutputs, aspectRatio: image.Point{ratioW, ratioH}}, nil
}

type chunks struct {
	*Model
	ml.Tensor
	aspectRatio image.Point

	dataOnce sync.Once
	data     []float32
}

type chunk struct {
	*chunks
	s, n int
}

func (r *chunk) floats() []float32 {
	r.dataOnce.Do(func() {
		temp := r.Backend().NewContext()
		defer temp.Close()
		temp.Forward(r.Tensor).Compute(r.Tensor)
		r.data = r.Floats()
	})

	return r.data[r.s*r.Dim(0) : (r.s+r.n)*r.Dim(0)]
}

func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			result = append(result, inp)
			continue
		}

		t := inp.Multimodal.(*chunks)
		var imageInputs []input.Input
		imageInputs = append(imageInputs, input.Input{Token: 200080}) // <|image_start|>

		var offset int
		patchesPerChunk := t.Dim(1)
		if t.aspectRatio.Y*t.aspectRatio.X > 1 {
			patchesPerChunk = t.Dim(1) / (t.aspectRatio.X*t.aspectRatio.Y + 1)

			for range t.aspectRatio.Y {
				for x := range t.aspectRatio.X {
					imageInputs = append(imageInputs, input.Input{Token: 200092, Multimodal: &chunk{t, offset, patchesPerChunk}, MultimodalHash: inp.MultimodalHash, SameBatch: patchesPerChunk}) // <|patch|>
					imageInputs = append(imageInputs, slices.Repeat([]input.Input{{Token: 200092}}, patchesPerChunk-1)...)
					if x < t.aspectRatio.X-1 {
						imageInputs = append(imageInputs, input.Input{Token: 200084}) // <|tile_x_separator|>
					}
					offset += patchesPerChunk
				}

				imageInputs = append(imageInputs, input.Input{Token: 200085}) // <|tile_y_separator|>
			}
		}

		imageInputs = append(imageInputs, input.Input{Token: 200090})                                                                                                                 // <|image|>
		imageInputs = append(imageInputs, input.Input{Token: 200092, Multimodal: &chunk{t, offset, patchesPerChunk}, MultimodalHash: inp.MultimodalHash, SameBatch: patchesPerChunk}) // <|patch|>
		imageInputs = append(imageInputs, slices.Repeat([]input.Input{{Token: 200092}}, patchesPerChunk-1)...)
		imageInputs = append(imageInputs, input.Input{Token: 200080}) // <|image_end|>

		result = append(result, imageInputs...)
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
	model.Register("llama4", New)
}
