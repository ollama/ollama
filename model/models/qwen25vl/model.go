package qwen25vl

import (
	"bytes"
	"fmt"
	"image"
	"slices"
	"sync"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Model struct {
	model.Base
	model.BytePairEncoding

	*TextModel
	*VisionModel `gguf:"v,vision"`

	ImageProcessor
}

// Implement MultimodalProcessor interface
var _ model.MultimodalProcessor = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	m := &Model{
		BytePairEncoding: model.NewBytePairEncoding(
			c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    int32(c.Uint("tokenizer.ggml.bos_token_id")),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				EOS:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOT:    int32(c.Uint("tokenizer.ggml.eos_token_id")),
				AddEOT: c.Bool("tokenizer.ggml.add_eos_token", false),
			},
		),
		TextModel:      NewTextModel(c),
		VisionModel:    newVisionModel(c),
		ImageProcessor: newImageProcessor(c),
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

func (m *Model) PixelValues(ctx ml.Context, multimodalData []byte) (ml.Tensor, *Grid, error) {
	image, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, nil, err
	}

	f32s, grid, err := m.ImageProcessor.ProcessImage(image)
	if err != nil {
		return nil, nil, err
	}

	// Calculate tensor dimensions
	patchDim := m.ImageProcessor.numChannels * m.ImageProcessor.temporalPatchSize *
		m.ImageProcessor.patchSize * m.ImageProcessor.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width

	pixelValues, err := ctx.Input().FromFloatSlice(f32s, patchDim, numPatches)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create tensor from image: %w", err)
	}

	return pixelValues, grid, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) (any, error) {
	if len(m.VisionModel.Layers) == 0 {
		return nil, model.ErrNoVisionModel
	}

	pixels, grid, err := m.PixelValues(ctx, multimodalData)
	if err != nil {
		return nil, err
	}

	visionOutputs := m.VisionModel.Forward(ctx, pixels, grid)
	return &chunks{Model: m, Tensor: visionOutputs}, nil
}

type chunks struct {
	*Model
	ml.Tensor

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

// PostTokenize arranges Qwen-2.5-VL's inputs for the forward pass
func (m *Model) PostTokenize(inputs []input.Input) ([]input.Input, error) {
	var result []input.Input

	var (
		imageToken       int32 = 151655
		visionStartToken int32 = 151652
		visionEndToken   int32 = 151653
	)

	nImg := 0
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			// If not a multimodal input, add it to the result unchanged
			result = append(result, inp)
		} else {
			// Adding the 'Picture' prefix is a hack, at the time of writing there is no way to prefix
			// the image tokens with a prompt, so we add a prefix here
			nImg++
			pre, err := m.Encode(fmt.Sprintf(" Picture %d: ", nImg), true)
			if err != nil {
				return nil, fmt.Errorf("failed to encode image prompt: %w", err)
			}
			for i := range pre {
				result = append(result, input.Input{Token: pre[i]})
			}

			// This is an image token with multimodal data
			chunksData := inp.Multimodal.(*chunks)
			patchesPerChunk := chunksData.Dim(1)

			// First add the vision start token
			result = append(result, input.Input{Token: visionStartToken, SameBatch: patchesPerChunk + 2})

			// Add the image token with the multimodal tensor data at the first position
			// Create a chunk with proper s and n values
			result = append(result, input.Input{
				Token:          imageToken,
				Multimodal:     &chunk{chunks: chunksData, s: 0, n: patchesPerChunk},
				MultimodalHash: inp.MultimodalHash,
				SameBatch:      patchesPerChunk,
			})

			// Add the placeholder tokens for the remaining positions (tokensPerGrid-1)
			result = append(result, slices.Repeat([]input.Input{{Token: imageToken}}, patchesPerChunk-1)...)

			result = append(result, input.Input{Token: visionEndToken})
		}
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

	return m.TextModel.Forward(ctx, batch.Inputs, positions, outputs, batch, m.Cache)
}

func init() {
	model.Register("qwen25vl", New)
}
