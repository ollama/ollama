package mllama

import (
	"bytes"
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
	model.BytePairEncoding

	*VisionModel `gguf:"v,vision"`
	*TextModel

	Projector *nn.Linear `gguf:"mm.0"`

	ImageProcessor
}

const (
	crossAttentionLayer = iota
	selfAttentionLayer
)

func New(c fs.Config) (model.Model, error) {
	m := Model{
		BytePairEncoding: model.NewBytePairEncoding(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				// TODO: set EOT to EOS otherwise 0 will stop generation
				EOT:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
				AddEOT: c.Bool("tokenizer.ggml.add_eos_token", false),
			},
		),
		ImageProcessor: newImageProcessor(c),
		VisionModel:    newVisionModel(c),
		TextModel:      newTextModel(c),
	}

	encoderCache := kvcache.NewEncoderCache()
	encoderCache.SetConfig(ml.CacheConfig{})
	m.Cache = kvcache.NewWrapperCache(encoderCache, kvcache.NewCausalCache(m.TextModel.Shift))

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	if len(m.VisionModel.Transformer.Layers) == 0 || len(m.GlobalTransformer.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, ratio, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	pixelValues, err := ctx.Input().FromFloatSlice(f32s, m.imageSize, m.imageSize, m.numChannels, ratio.numTiles())
	if err != nil {
		return nil, err
	}

	pixelValues = pixelValues.Pad(ctx, 0, 0, 0, m.ImageProcessor.maxNumTiles-ratio.numTiles())

	aspectRatio, err := ctx.Input().FromIntSlice([]int32{int32(ratio.rank)}, 1)
	if err != nil {
		return nil, err
	}

	positionIDs := ctx.Arange(0, 1601, 1, ml.DTypeI32)
	crossAttentionStates := m.VisionModel.Forward(ctx, pixelValues, positionIDs, aspectRatio)
	return m.Projector.Forward(ctx, crossAttentionStates), nil
}

func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	for i := range inputs {
		if inputs[i].Multimodal != nil {
			inputs[i].Token = 128256 // <|image|>
		}
	}

	return inputs, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	var crossAttentionStates ml.Tensor
	if len(batch.Multimodal) > 0 {
		crossAttentionStates = batch.Multimodal[len(batch.Multimodal)-1].Multimodal.(ml.Tensor)
	}

	positions, err := ctx.Input().FromIntSlice(batch.Positions, len(batch.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Input().FromIntSlice(batch.Outputs, len(batch.Outputs))
	if err != nil {
		return nil, err
	}

	// TODO: attention mask, cross attention mask
	return m.TextModel.Forward(ctx, batch.Inputs, positions, outputs, crossAttentionStates, nil, m.Cache.(*kvcache.WrapperCache)), nil
}

func init() {
	model.Register("mllama", New)
}
