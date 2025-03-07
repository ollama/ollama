package mllama

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"image"
	"slices"

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
	// Verify unified config
	if c.Uint("vision.block_count") == 0 {
		return nil, fmt.Errorf("non-unified vision model not supported")
	}
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

	encoderCache := kvcache.NewEncoderCache()
	encoderCache.SetConfig(ml.CacheConfig{})
	m.Cache = kvcache.NewWrapperCache(encoderCache, kvcache.NewCausalCache(m.TextModel.Shift))

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, aspectRatioID, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	pixelValues, err := ctx.Input().FromFloatSlice(f32s,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.numChannels,
		m.ImageProcessor.maxNumTiles,
	)
	if err != nil {
		return nil, err
	}

	aspectRatio, err := ctx.Input().FromIntSlice([]int32{int32(aspectRatioID)}, 1)
	if err != nil {
		return nil, err
	}

	positions := make([]int32, 1601)
	for i := range positions {
		positions[i] = int32(i)
	}

	positionIDs, err := ctx.Input().FromIntSlice(positions, len(positions))
	if err != nil {
		return nil, err
	}

	crossAttentionStates := m.VisionModel.Forward(ctx, pixelValues, positionIDs, aspectRatio)
	return m.Projector.Forward(ctx, crossAttentionStates), nil
}

func (m *Model) PostTokenize(ctx ml.Context, inputs []model.Input) ([]model.Input, error) {
	var images []model.Input
	fnvHash := fnv.New64a()

	for i := range inputs {
		if inputs[i].Multimodal == nil {
			if len(images) > 0 {
				inputs[i].Multimodal = images[0].Multimodal
				inputs[i].MultimodalHash = images[0].MultimodalHash
				for j := 1; j < len(images); j++ {
					inputs[i].Multimodal = inputs[i].Multimodal.(ml.Tensor).Concat(ctx, images[j].Multimodal.(ml.Tensor), 3)
					fnvHash.Reset()
					binary.Write(fnvHash, binary.NativeEndian, inputs[i].MultimodalHash)
					binary.Write(fnvHash, binary.NativeEndian, inputs[j].MultimodalHash)
					inputs[i].MultimodalHash = fnvHash.Sum64()
				}
				images = nil
			}
		} else {
			images = append(images, inputs[i])
			inputs[i].Token = -1
		}
	}

	inputs = slices.DeleteFunc(inputs, func(input model.Input) bool { return input.Token == -1 })

	return inputs, nil
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	var crossAttentionStates ml.Tensor
	if opts.Multimodal != nil {
		crossAttentionStates = opts.Multimodal[0].Multimodal.(ml.Tensor)
	}

	inputs, err := ctx.Input().FromIntSlice(opts.Inputs, len(opts.Inputs))
	if err != nil {
		return nil, err
	}

	positions, err := ctx.Input().FromIntSlice(opts.Positions, len(opts.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Output().FromIntSlice(opts.Outputs, len(opts.Outputs))
	if err != nil {
		return nil, err
	}

	// TODO: attention mask, cross attention mask
	return m.TextModel.Forward(ctx, inputs, positions, outputs, nil, crossAttentionStates, nil, m.Cache.(*kvcache.WrapperCache)), nil
}

func init() {
	model.Register("mllama", New)
}
