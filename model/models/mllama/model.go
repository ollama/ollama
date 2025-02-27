package mllama

import (
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
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

func New(c ml.Config) (model.Model, error) {
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

	m.Cache = kvcache.NewWrapperCache(kvcache.NewEncoderCache(), kvcache.NewCausalCache(m.TextModel.Shift))

	return &m, nil
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	var crossAttentionStates ml.Tensor
	if opts.Images != nil {
		f32s, aspectRatioID, err := m.ImageProcessor.ProcessImage(opts.Images[0])
		if err != nil {
			return nil, err
		}

		pixelValues, err := ctx.FromFloatSlice(f32s,
			m.ImageProcessor.imageSize,
			m.ImageProcessor.imageSize,
			m.ImageProcessor.numChannels,
			m.ImageProcessor.maxNumTiles,
		)
		if err != nil {
			return nil, err
		}

		aspectRatio, err := ctx.FromIntSlice([]int32{int32(aspectRatioID)}, 1)
		if err != nil {
			return nil, err
		}

		positions := make([]int32, 1601)
		for i := range positions {
			positions[i] = int32(i)
		}

		positionIDs, err := ctx.FromIntSlice(positions, len(positions))
		if err != nil {
			return nil, err
		}

		crossAttentionStates = m.VisionModel.Forward(ctx, pixelValues, positionIDs, aspectRatio)
		crossAttentionStates = m.Projector.Forward(ctx, crossAttentionStates)
	}

	inputs, err := ctx.FromIntSlice(opts.Inputs, len(opts.Inputs))
	if err != nil {
		return nil, err
	}

	positions, err := ctx.FromIntSlice(opts.Positions, len(opts.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.FromIntSlice(opts.Outputs, len(opts.Outputs))
	if err != nil {
		return nil, err
	}

	// TODO: attention mask, cross attention mask
	return m.TextModel.Forward(ctx, inputs, positions, outputs, nil, crossAttentionStates, nil, m.Cache.(*kvcache.WrapperCache)), nil
}

func init() {
	model.Register("mllama", New)
}
