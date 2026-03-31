package gemma4

import (
	"bytes"
	"image"
	"log/slog"
	"slices"
	"time"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

type Model struct {
	model.Base
	tokenizer.Tokenizer

	*VisionModel `gguf:"v"`
	*TextModel

	*MultiModalProjector `gguf:"mm"`

	ImageProcessor

	imageTokenID    int32
	imageEndTokenID int32
}

var _ model.MultimodalProcessor = (*Model)(nil)

type MultiModalProjector struct {
	Projection *ClippableLinear `gguf:"input_projection"`
}

func (p *MultiModalProjector) Forward(ctx ml.Context, visionOutputs ml.Tensor, eps float32) ml.Tensor {
	visionOutputs = p.Projection.Forward(ctx, visionOutputs)
	// Post-projection RMSNorm without learned weight
	visionOutputs = visionOutputs.RMSNorm(ctx, nil, eps)
	return visionOutputs
}

func New(c fs.Config) (model.Model, error) {
	vocabulary := tokenizer.Vocabulary{
		Values: c.Strings("tokenizer.ggml.tokens"),
		Scores: c.Floats("tokenizer.ggml.scores"),
		Types:  c.Ints("tokenizer.ggml.token_type"),
		Merges: c.Strings("tokenizer.ggml.merges"),
		AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
		BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{
				int32(c.Uint("tokenizer.ggml.eos_token_id")),
			},
			c.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	vocabulary.EOS = append(vocabulary.EOS, int32(c.Uint("tokenizer.ggml.eot_token_id", 106)))

	// Gemma 4 uses BPE with SentencePiece-style ▁ space markers (not GPT-2 byte-level encoding).
	// The tokenizer.json has merges and a Replace normalizer (space → ▁), with no pre-tokenizer.
	t := tokenizer.NewBytePairEncodingWithOptions(&vocabulary, []string{},
		tokenizer.WithSentencePieceNormalizer())

	// Look up special token IDs for vision
	imageTokenID := int32(-1)
	imageEndTokenID := int32(-1)
	for i, tok := range vocabulary.Values {
		switch tok {
		case "<|image>":
			imageTokenID = int32(i)
		case "<image|>":
			imageEndTokenID = int32(i)
		}
	}

	m := Model{
		Tokenizer:            t,
		TextModel:            newTextModel(c),
		VisionModel:          newVisionModel(c),
		MultiModalProjector:  &MultiModalProjector{},
		ImageProcessor:       newImageProcessor(c),
		imageTokenID:         imageTokenID,
		imageEndTokenID:      imageEndTokenID,
	}

	slidingWindowLen := int32(c.Uint("attention.sliding_window"))
	m.Cache = kvcache.NewWrapperCache(kvcache.NewSWACache(slidingWindowLen, m.Shift), kvcache.NewCausalCache(m.Shift))

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	// Initialize clamp values from model tensors (lazy, once, after model is fully loaded)
	m.VisionModel.InitClamp(m.MultiModalProjector)

	t0 := time.Now()
	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}
	slog.Info("vision: decode", "elapsed", time.Since(t0), "bounds", img.Bounds())

	t1 := time.Now()
	f32s, imgW, imgH, err := m.ImageProcessor.ProcessImage(img)
	if err != nil {
		return nil, err
	}
	slog.Info("vision: preprocess", "elapsed", time.Since(t1), "size", [2]int{imgW, imgH})

	pixelValues := ctx.Input().FromFloats(f32s, imgW, imgH, m.ImageProcessor.numChannels)
	slog.Info("vision: pixelValues", "shape", pixelValues.Shape(), "dim0", pixelValues.Dim(0), "dim1", pixelValues.Dim(1), "dim2", pixelValues.Dim(2))

	numPatchesX := imgW / m.ImageProcessor.patchSize
	numPatchesY := imgH / m.ImageProcessor.patchSize
	slog.Info("vision: patches", "patchesX", numPatchesX, "patchesY", numPatchesY, "total", numPatchesX*numPatchesY, "patchSize", m.ImageProcessor.patchSize)

	visionOutputs := m.VisionModel.Forward(ctx, pixelValues, numPatchesX, numPatchesY)
	visionOutputs = visionPoolAndProject(ctx, visionOutputs, numPatchesX, numPatchesY, m.VisionModel.VisionModelOptions, m.MultiModalProjector)
	slog.Info("vision: encoded", "elapsed", time.Since(t0), "shape", visionOutputs.Shape())

	return []input.Multimodal{{Tensor: visionOutputs}}, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
		} else {
			inputMultimodal := inp.Multimodal[0].Tensor
			numImageTokens := inputMultimodal.Dim(1)

			// <|image>
			if m.imageTokenID >= 0 {
				result = append(result, &input.Input{Token: m.imageTokenID, SameBatch: numImageTokens + 2})
			}

			// Image embedding placeholder tokens
			result = append(result,
				&input.Input{Multimodal: []input.Multimodal{{Tensor: inputMultimodal}}, MultimodalHash: inp.MultimodalHash},
			)
			result = append(result, slices.Repeat([]*input.Input{{Token: 0}}, numImageTokens-1)...)

			// <image|>
			if m.imageEndTokenID >= 0 {
				result = append(result, &input.Input{Token: m.imageEndTokenID})
			}
		}
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenState := m.TextModel.Forward(ctx, batch, m.Cache)

	hiddenState = m.TextModel.Output.Forward(ctx, hiddenState)

	if m.TextModel.TextOptions.finalLogitSoftcap > 0.0 {
		hiddenState = hiddenState.Scale(ctx, 1.0/float64(m.TextModel.TextOptions.finalLogitSoftcap))
		hiddenState = hiddenState.Tanh(ctx)
		hiddenState = hiddenState.Scale(ctx, float64(m.TextModel.TextOptions.finalLogitSoftcap))
	}

	return hiddenState, nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase, ropeDims := m.TextModel.ropeForLayer(layer)
	return nn.RoPE(ctx, key, shift, ropeDims, ropeBase, 1.0, rope.WithTypeNeoX()), nil
}

func init() {
	model.Register("gemma4", New)
}
