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

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase, ropeDims := m.TextModel.ropeForLayer(layer)
	return nn.RoPE(ctx, key, shift, ropeDims, ropeBase, 1.0, rope.WithTypeNeoX()), nil
}

func init() {
	model.Register("gemma4", New)
}
