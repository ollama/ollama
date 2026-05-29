package gemma4

import (
	"bytes"
	"fmt"
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
	*AudioModel `gguf:"a"`
	*DraftModel `gguf:"draft"`

	*MultiModalProjector      `gguf:"mm"`
	*AudioMultimodalProjector `gguf:"mm.a"`

	ImageProcessor

	imageTokenID    int32
	imageEndTokenID int32
	audioTokenID    int32
	audioEndTokenID int32

	audioOpts *AudioModelOptions
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

	// Look up special token IDs for vision and audio
	imageTokenID := int32(-1)
	imageEndTokenID := int32(-1)
	audioTokenID := int32(-1)
	audioEndTokenID := int32(-1)
	for i, tok := range vocabulary.Values {
		switch tok {
		case "<|image>":
			imageTokenID = int32(i)
		case "<image|>":
			imageEndTokenID = int32(i)
		case "<|audio>":
			audioTokenID = int32(i)
		case "<audio|>":
			audioEndTokenID = int32(i)
		}
	}

	slog.Info("gemma4: token IDs", "image", imageTokenID, "image_end", imageEndTokenID, "audio", audioTokenID, "audio_end", audioEndTokenID)

	m := Model{
		Tokenizer:                t,
		TextModel:                newTextModel(c),
		VisionModel:              newVisionModel(c),
		AudioModel:               newAudioModel(c),
		DraftModel:               newDraftModel(c),
		MultiModalProjector:      &MultiModalProjector{},
		AudioMultimodalProjector: &AudioMultimodalProjector{},
		ImageProcessor:           newImageProcessor(c),
		imageTokenID:             imageTokenID,
		imageEndTokenID:          imageEndTokenID,
		audioTokenID:             audioTokenID,
		audioEndTokenID:          audioEndTokenID,
		audioOpts:                newAudioModelOptions(c),
	}

	slidingWindowLen := int32(c.Uint("attention.sliding_window"))
	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewSWAMemCache(slidingWindowLen, 4096, m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)

	return &m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	// Audio input: detect WAV format and route to audio encoder.
	if isAudioData(multimodalData) {
		return m.encodeAudioMultimodal(ctx, multimodalData)
	}

	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

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
	visionOutputs = visionPoolAndProject(ctx, visionOutputs, numPatchesX, numPatchesY, m.VisionModel.VisionModelOptions, m.MultiModalProjector, m.VisionModel.StdBias, m.VisionModel.StdScale)
	slog.Info("vision: encoded", "elapsed", time.Since(t0), "shape", visionOutputs.Shape())

	return []input.Multimodal{{Tensor: visionOutputs}}, nil
}

func (m *Model) PostLoad() error {
	m.VisionModel.InitClamp(m.MultiModalProjector)
	return nil
}

func (m *Model) encodeAudioMultimodal(ctx ml.Context, data []byte) ([]input.Multimodal, error) {
	if m.AudioModel == nil || m.audioOpts == nil {
		return nil, model.ErrNoVisionModel
	}

	t0 := time.Now()
	samples, err := decodeWAV(data)
	if err != nil {
		return nil, err
	}
	slog.Info("audio: decode", "elapsed", time.Since(t0), "samples", len(samples), "duration_s", float64(len(samples))/audioSampleRate)

	// Pad waveform to next multiple of 128.
	if rem := len(samples) % 128; rem != 0 {
		samples = append(samples, make([]float32, 128-rem)...)
	}

	// Compute mel spectrogram.
	melData, numFrames := computeMelSpectrogram(samples)
	if numFrames == 0 {
		return nil, fmt.Errorf("audio too short to encode")
	}
	slog.Info("audio: mel", "frames", numFrames, "elapsed", time.Since(t0))

	// Create input tensor [melBins, numFrames] (GGML ne order). FromFloats creates F32.
	melTensor := ctx.Input().FromFloats(melData, melBins, numFrames)

	// Run audio encoder.
	audioOutputs := m.AudioModel.ForwardAudio(ctx, melTensor, m.AudioMultimodalProjector, m.audioOpts)
	slog.Info("audio: encoded", "elapsed", time.Since(t0), "shape", audioOutputs.Shape())

	return []input.Multimodal{{Tensor: audioOutputs, Data: audioTag{}}}, nil
}

// audioTag marks multimodal data as audio (vs vision) for PostTokenize.
type audioTag struct{}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	for _, inp := range inputs {
		if len(inp.Multimodal) == 0 {
			result = append(result, inp)
			continue
		}

		inputMultimodal := inp.Multimodal[0].Tensor
		numTokens := inputMultimodal.Dim(1)

		// Determine if this is audio or vision based on the tag.
		_, isAudio := inp.Multimodal[0].Data.(audioTag)

		var beginToken, endToken int32
		if isAudio {
			beginToken = m.audioTokenID
			endToken = m.audioEndTokenID
		} else {
			beginToken = m.imageTokenID
			endToken = m.imageEndTokenID
		}

		if beginToken >= 0 {
			result = append(result, &input.Input{Token: beginToken, SameBatch: numTokens + 2})
		}

		result = append(result,
			&input.Input{Multimodal: []input.Multimodal{{Tensor: inputMultimodal}}, MultimodalHash: inp.MultimodalHash},
		)
		result = append(result, slices.Repeat([]*input.Input{{Token: 0}}, numTokens-1)...)

		if endToken >= 0 {
			result = append(result, &input.Input{Token: endToken})
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

func (m *Model) HasDraft() bool {
	return m.DraftModel != nil
}

func (m *Model) ForwardMTP(ctx ml.Context, batch input.Batch) (ml.Tensor, ml.Tensor, error) {
	if cache := m.Config().Cache; cache != nil {
		if err := cache.StartForward(ctx, batch, false); err != nil {
			return nil, nil, err
		}
	}

	normed, hidden := m.TextModel.ForwardWithHidden(ctx, batch, m.Cache)

	logits := m.TextModel.Output.Forward(ctx, normed)
	if m.TextModel.TextOptions.finalLogitSoftcap > 0 {
		logits = logits.Scale(ctx, 1.0/float64(m.TextModel.TextOptions.finalLogitSoftcap))
		logits = logits.Tanh(ctx)
		logits = logits.Scale(ctx, float64(m.TextModel.TextOptions.finalLogitSoftcap))
	}

	if t, ok := hidden.(interface{ SetOutput() }); ok {
		t.SetOutput()
	}
	ctx.Forward(logits)
	ctx.Forward(hidden)
	return logits, hidden, nil
}

func (m *Model) MTPDraft(ctx ml.Context, token int32, hiddenFloats []float32, hiddenDim int, position int32, seqID int, cache kvcache.Cache, maxDraft int) ([]int32, error) {
	var draftTokens []int32
	lastToken := token
	lastHiddenFloats := hiddenFloats
	backend := m.Backend()

	for range maxDraft {
		iterCtx := backend.NewContext()

		draftBatch := input.Batch{
			Inputs:    iterCtx.Input().Empty(ml.DTypeI32, 1),
			Positions: []int32{position},
			Sequences: []int{seqID},
			Outputs:   iterCtx.Input().FromInts([]int32{0}, 1),
		}

		if err := cache.StartForward(iterCtx, draftBatch, true); err != nil {
			iterCtx.Close()
			return draftTokens, err
		}

		lastHidden := iterCtx.Input().FromFloats(lastHiddenFloats, hiddenDim)
		tokenTensor := iterCtx.Input().FromInts([]int32{lastToken}, 1)
		embedding := m.TextModel.TokenEmbeddings(iterCtx, tokenTensor)
		inputEmbeds := embedding.Concat(iterCtx, lastHidden, 0)

		logits, projected := m.DraftModel.Draft(iterCtx, inputEmbeds, position, cache, &m.TextModel.TextOptions)

		iterCtx.Forward(logits)
		iterCtx.Forward(projected)
		iterCtx.Compute(logits, projected)

		logitValues := logits.Floats()
		lastHiddenFloats = projected.Floats()

		iterCtx.Close()

		nextToken := argmaxSlice(logitValues)
		draftTokens = append(draftTokens, nextToken)
		lastToken = nextToken
	}

	return draftTokens, nil
}

func (m *Model) MTPVerify(ctx ml.Context, baseLogits []float32, token int32, draftTokens []int32, seqID int, position int32, cache kvcache.Cache) (int, int32, error) {
	N := len(draftTokens)
	vocabSize := len(baseLogits)

	baseChoice := argmaxSlice(baseLogits)
	if N == 0 || baseChoice != draftTokens[0] {
		return 0, baseChoice, nil
	}

	// draftTokens[0] matched baseChoice (both predict P+1 = token).
	// Build verify batch: [token, draftTokens[1], ..., draftTokens[N-1]]
	// at positions [P+1, P+2, ..., P+N].
	remaining := draftTokens[1:]
	M := 1 + len(remaining)
	verifyInputs := make([]int32, M)
	verifyInputs[0] = token
	copy(verifyInputs[1:], remaining)

	positions := make([]int32, M)
	sequences := make([]int, M)
	outputs := make([]int32, M)
	for i := range M {
		positions[i] = position + 1 + int32(i)
		sequences[i] = seqID
		outputs[i] = int32(i)
	}

	verifyBatch := input.Batch{
		Inputs:    ctx.Input().FromInts(verifyInputs, M),
		Positions: positions,
		Sequences: sequences,
		Outputs:   ctx.Input().FromInts(outputs, M),
	}

	if err := cache.StartForward(ctx, verifyBatch, false); err != nil {
		return 0, baseChoice, err
	}

	normed := m.TextModel.Forward(ctx, verifyBatch, cache)
	logitsTensor := m.TextModel.Output.Forward(ctx, normed)

	if m.TextModel.TextOptions.finalLogitSoftcap > 0 {
		logitsTensor = logitsTensor.Scale(ctx, 1.0/float64(m.TextModel.TextOptions.finalLogitSoftcap))
		logitsTensor = logitsTensor.Tanh(ctx)
		logitsTensor = logitsTensor.Scale(ctx, float64(m.TextModel.TextOptions.finalLogitSoftcap))
	}

	ctx.Forward(logitsTensor)
	ctx.Compute(logitsTensor)
	allLogits := logitsTensor.Floats()

	// Output[0] at P+1 (input=token) predicts P+2 → compare with remaining[0]=draftTokens[1]
	accepted := 1
	for i := range remaining {
		posLogits := allLogits[i*vocabSize : (i+1)*vocabSize]
		if argmaxSlice(posLogits) != remaining[i] {
			break
		}
		accepted++
	}

	var nextToken int32
	if accepted > len(remaining) {
		lastLogits := allLogits[len(remaining)*vocabSize : (len(remaining)+1)*vocabSize]
		nextToken = argmaxSlice(lastLogits)
	} else {
		mismatchLogits := allLogits[(accepted-1)*vocabSize : accepted*vocabSize]
		nextToken = argmaxSlice(mismatchLogits)
	}

	return accepted, nextToken, nil
}

func argmaxSlice(s []float32) int32 {
	best := int32(0)
	bestVal := s[0]
	for i := int32(1); i < int32(len(s)); i++ {
		if s[i] > bestVal {
			bestVal = s[i]
			best = i
		}
	}
	return best
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase, ropeDims := m.TextModel.ropeForLayer(layer)
	return nn.RoPE(ctx, key, shift, ropeDims, ropeBase, 1.0, rope.WithTypeNeoX()), nil
}

func init() {
	model.Register("gemma4", New)
}
