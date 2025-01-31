package gemma3

import (
	"bytes"
	"image"
	"math"
	"slices"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.SentencePieceModel

	*VisionModel `gguf:"v,vision"`
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
	l := visionOutputs.Dim(1)

	visionOutputs = visionOutputs.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	patchesPerImage := imageSize / patchSize
	visionOutputs = visionOutputs.Reshape(ctx, l, patchesPerImage, patchesPerImage)

	kernelSize := patchesPerImage / int(math.Sqrt(float64(p.tokensPerImage)))
	visionOutputs = visionOutputs.AvgPool2D(ctx, kernelSize, kernelSize, 0)
	visionOutputs = visionOutputs.Reshape(ctx, l, visionOutputs.Dim(2)*visionOutputs.Dim(1))
	visionOutputs = visionOutputs.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	visionOutputs = p.SoftEmbNorm.Forward(ctx, visionOutputs, eps)

	// TODO: inputProjection must be transposed since they're incompatible with visionOutputs
	visionOutputs = p.InputProjection.Weight.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Mulmat(ctx, visionOutputs)
	return visionOutputs
}

func New(c ml.Config) (model.Model, error) {
	m := Model{
		SentencePieceModel: model.NewSentencePieceModel(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Scores: c.Floats("tokenizer.ggml.scores"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				EOS:    int32(1),
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOT:    int32(106),
				AddEOT: c.Bool("tokenizer.ggml.add_eot_token", false),
			},
		),
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

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
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

	pixelValues, err := ctx.Input().FromFloatSlice(f32s,
		m.ImageProcessor.numChannels,
		m.ImageProcessor.imageSize,
		m.ImageProcessor.imageSize,
	)
	if err != nil {
		return nil, err
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues)
	visionOutputs = m.MultiModalProjector.Forward(ctx, visionOutputs, m.imageSize, m.patchSize, m.VisionModel.eps)
	return visionOutputs, nil
}

func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input

	for _, inp := range inputs {
		if inp.Multimodal == nil {
			result = append(result, inp)
		} else {
			inputMultimodal := inp.Multimodal.(ml.Tensor)

			result = append(result,
				input.Input{Token: 108, SameBatch: inputMultimodal.Dim(0) + 3},               // "\n\n"
				input.Input{Token: 255999},                                                   // "<start_of_image>""
				input.Input{Multimodal: inputMultimodal, MultimodalHash: inp.MultimodalHash}, // image data is on the first placeholder
			)

			// add image token placeholders
			result = append(result, slices.Repeat([]input.Input{{Token: 0}}, inputMultimodal.Dim(0)-1)...)

			result = append(result,
				input.Input{Token: 256000}, // <end_of_image>
				input.Input{Token: 108},    // "\n\n"
			)
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, opts input.Options) (ml.Tensor, error) {
	inputs, err := ctx.Input().FromIntSlice(opts.Inputs, len(opts.Inputs))
	if err != nil {
		return nil, err
	}

	positions, err := ctx.Input().FromIntSlice(opts.Positions, len(opts.Positions))
	if err != nil {
		return nil, err
	}

	outputs, err := ctx.Input().FromIntSlice(opts.Outputs, len(opts.Outputs))
	if err != nil {
		return nil, err
	}

	return m.TextModel.Forward(ctx, inputs, positions, outputs, opts, m.Cache), nil
}

func init() {
	model.Register("gemma3", New)
}
