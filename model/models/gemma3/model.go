package gemma3

import (
	"bytes"
	"image"
	"math"
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
	model.TextProcessor

	*VisionModel `gguf:"v"`
	*TextModel

	*MultiModalProjector `gguf:"mm"`

	ImageProcessor
}

var _ model.MultimodalProcessor = (*Model)(nil)

type MultiModalProjector struct {
	SoftEmbNorm     *nn.RMSNorm `gguf:"mm_soft_emb_norm"`
	InputProjection *nn.Linear  `gguf:"mm_input_projection"`

	tokensPerImage int
}

func (p *MultiModalProjector) Forward(ctx ml.Context, visionOutputs ml.Tensor, imageSize, patchSize int, eps float32) ml.Tensor {
	l := visionOutputs.Dim(0)

	visionOutputs = visionOutputs.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	patchesPerImage := imageSize / patchSize
	visionOutputs = visionOutputs.Reshape(ctx, patchesPerImage, patchesPerImage, l)

	kernelSize := patchesPerImage / int(math.Sqrt(float64(p.tokensPerImage)))
	visionOutputs = visionOutputs.AvgPool2D(ctx, kernelSize, kernelSize, 0)
	visionOutputs = visionOutputs.Reshape(ctx, visionOutputs.Dim(0)*visionOutputs.Dim(1), l)
	visionOutputs = visionOutputs.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	visionOutputs = p.SoftEmbNorm.Forward(ctx, visionOutputs, eps)

	// TODO: inputProjection must be transposed since they're incompatible with visionOutputs
	visionOutputs = p.InputProjection.Weight.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mulmat(ctx, visionOutputs)
	return visionOutputs
}

func New(c fs.Config) (model.Model, error) {
	vocabulary := model.Vocabulary{
		Values: c.Strings("tokenizer.ggml.tokens"),
		Scores: c.Floats("tokenizer.ggml.scores"),
		Types:  c.Ints("tokenizer.ggml.token_type"),
		Merges: c.Strings("tokenizer.ggml.merges"),
		AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
		BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{
				int32(c.Uint("tokenizer.ggml.eos_token_id")),
			},
			c.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	var processor model.TextProcessor
	switch c.String("tokenizer.ggml.model") {
	case "gpt2":
		processor = model.NewBytePairEncoding(&vocabulary)
	default:
		// Previous uploads of Gemma 3 on Ollama did not have token 106
		// (i.e. "<end_of_turn>") so we need to add in case it's not already present
		vocabulary.EOS = append(vocabulary.EOS, int32(c.Uint("tokenizer.ggml.eot_token_id", 106)))
		processor = model.NewSentencePiece(&vocabulary)
	}

	m := Model{
		TextProcessor:  processor,
		ImageProcessor: newImageProcessor(c),
		VisionModel:    newVisionModel(c),
		TextModel:      newTextModel(c),
		MultiModalProjector: &MultiModalProjector{
			tokensPerImage: int(c.Uint("mm_tokens_per_image", 256)),
		},
	}

	slidingWindowLen := int32(c.Uint("attention.sliding_window"))
	m.Cache = kvcache.NewWrapperCache(kvcache.NewSWACache(slidingWindowLen, m.Shift), kvcache.NewCausalCache(m.Shift))

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, err
	}

	pixelValues := ctx.Input().FromFloats(f32s,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.numChannels,
	)

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	visionOutputs = m.MultiModalProjector.Forward(ctx, visionOutputs, m.imageSize, m.patchSize, m.VisionModel.eps)
	return []input.Multimodal{{Tensor: visionOutputs}}, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
		} else {
			inputMultimodal := inp.Multimodal[0].Tensor

			result = append(result,
				&input.Input{Token: 108, SameBatch: inputMultimodal.Dim(1) + 3}, // "\n\n"
				&input.Input{Token: 255999},                                     // "<start_of_image>""
				&input.Input{Multimodal: []input.Multimodal{{Tensor: inputMultimodal}}, MultimodalHash: inp.MultimodalHash}, // image data is on the first placeholder
			)

			// add image token placeholders
			result = append(result, slices.Repeat([]*input.Input{{Token: 0}}, inputMultimodal.Dim(1)-1)...)

			result = append(result,
				&input.Input{Token: 256000}, // <end_of_image>
				&input.Input{Token: 108},    // "\n\n"
			)
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenState := m.TextModel.Forward(ctx, batch, m.Cache)
	hiddenState = m.Output.Forward(ctx, hiddenState)

	if m.TextConfig.finalLogitSoftcap > 0.0 {
		hiddenState = hiddenState.Scale(ctx, 1.0/float64(m.TextConfig.finalLogitSoftcap))
		hiddenState = hiddenState.Tanh(ctx)
		hiddenState = hiddenState.Scale(ctx, float64(m.TextConfig.finalLogitSoftcap))
	}

	return hiddenState, nil
}

func init() {
	model.Register("gemma3", New)
	model.Register("gemma3_embed", newEmbedModel)
}
