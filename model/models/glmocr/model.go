package glmocr

import (
	"bytes"
	"errors"
	"image"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

type Model struct {
	model.Base
	tokenizer.Tokenizer

	*TextModel
	*VisionModel     `gguf:"v"`
	VisionDownsample *VisionDownsample `gguf:"mm.patch_merger"`
	PatchMerger      *PatchMerger      `gguf:"mm"`

	ImageProcessor

	imageTokenID      int32
	imageStartTokenID int32
	imageEndTokenID   int32
}

var _ model.MultimodalProcessor = (*Model)(nil)

func New(c fs.Config) (model.Model, error) {
	eosTokenID := int32(c.Uint("tokenizer.ggml.eos_token_id"))
	eosTokenIDs := c.Ints("tokenizer.ggml.eos_token_ids")
	allEOS := append([]int32{eosTokenID}, eosTokenIDs...)

	m := &Model{
		Tokenizer: tokenizer.NewBytePairEncoding(
			&tokenizer.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS:    allEOS,
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		TextModel:         newTextModel(c),
		VisionModel:       newVisionModel(c),
		ImageProcessor:    newImageProcessor(c),
		imageTokenID:      int32(c.Uint("image_token_id", 59280)),
		imageStartTokenID: int32(c.Uint("image_start_token_id", 59256)),
		imageEndTokenID:   int32(c.Uint("image_end_token_id", 59257)),
	}

	m.Cache = kvcache.NewCausalCache(m.TextModel.Shift)

	return m, nil
}

func (m *Model) EncodeMultimodal(ctx ml.Context, multimodalData []byte) ([]input.Multimodal, error) {
	if len(m.VisionModel.Blocks) == 0 {
		return nil, model.ErrNoVisionModel
	}

	img, _, err := image.Decode(bytes.NewReader(multimodalData))
	if err != nil {
		return nil, err
	}

	f32s, grid, err := m.ImageProcessor.ProcessImage(img)
	if err != nil {
		return nil, err
	}

	// Create pixel values tensor from flattened patches
	// Shape: [patchDim, numPatches]
	patchDim := m.VisionModel.numChannels * m.temporalPatchSize * m.patchSize * m.patchSize
	numPatches := grid.Temporal * grid.Height * grid.Width
	pixelValues := ctx.Input().FromFloats(f32s, patchDim, numPatches)

	// Forward through vision encoder
	visionOutputs := m.VisionModel.Forward(ctx, pixelValues, grid)

	// Forward through downsample (patch merger)
	if m.VisionDownsample == nil || m.VisionDownsample.Weight == nil {
		return nil, errors.New("glmocr: missing vision downsample weights")
	}
	visionOutputs = m.VisionDownsample.Forward(ctx, visionOutputs, grid, m.VisionModel.VisionModelOptions)

	// Forward through patch merger (FC + LayerNorm + GELU + SwiGLU FFN)
	if m.PatchMerger == nil {
		return nil, errors.New("glmocr: missing patch merger weights")
	}
	visionOutputs = m.PatchMerger.Forward(ctx, visionOutputs, m.VisionModel.VisionModelOptions)

	return []input.Multimodal{{Tensor: visionOutputs, Data: grid}}, nil
}

func (m *Model) PostTokenize(inputs []*input.Input) ([]*input.Input, error) {
	var result []*input.Input

	// Reset position cache
	m.TextModel.positionCache = m.TextModel.positionCache[:0]
	m.TextModel.ropeDelta = 0

	pos := int32(0)
	for _, inp := range inputs {
		if inp.Multimodal == nil {
			result = append(result, inp)
			m.TextModel.positionCache = append(m.TextModel.positionCache, pos)
			pos++
			continue
		}

		// Get grid info for position calculation
		grid := inp.Multimodal[0].Data.(*Grid)
		mergedH := grid.Height / m.VisionModel.spatialMergeSize
		mergedW := grid.Width / m.VisionModel.spatialMergeSize

		// Add image start token
		result = append(result, &input.Input{Token: m.imageStartTokenID})
		m.TextModel.positionCache = append(m.TextModel.positionCache, pos)
		pos++

		// Add image tokens with multimodal data
		// All image tokens share the same base position for temporal dimension
		tokensPerGrid := inp.Multimodal[0].Tensor.Dim(1)
		basePos := pos
		sameBatch := tokensPerGrid - 1
		if sameBatch < 0 {
			sameBatch = 0
		}
		result = append(result, &input.Input{
			Token:          m.imageTokenID,
			Multimodal:     inp.Multimodal,
			MultimodalHash: inp.MultimodalHash,
			SameBatch:      sameBatch,
		})
		m.TextModel.positionCache = append(m.TextModel.positionCache, basePos)

		// Add placeholder tokens for remaining positions
		// All image tokens use the same base position (temporal stays constant)
		for range tokensPerGrid - 1 {
			result = append(result, &input.Input{Token: m.imageTokenID})
			m.TextModel.positionCache = append(m.TextModel.positionCache, basePos)
		}

		// Advance position by max(mergedH, mergedW) after image tokens
		pos = basePos + int32(max(mergedH, mergedW))

		// Add image end token
		result = append(result, &input.Input{Token: m.imageEndTokenID})
		m.TextModel.positionCache = append(m.TextModel.positionCache, pos)
		pos++
	}

	// Compute rope delta for continuation after the prefill segment:
	// delta = (max_position_id + 1) - sequence_length
	if len(m.TextModel.positionCache) > 0 {
		last := m.TextModel.positionCache[len(m.TextModel.positionCache)-1]
		m.TextModel.ropeDelta = last + 1 - int32(len(m.TextModel.positionCache))
	}

	return result, nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	// Initial token embedding
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs).Duplicate(ctx)
	ctx.Forward(hiddenStates)

	// Build position slices for M-RoPE
	positionSlice := func() [][]int32 {
		s := [][]int32{
			make([]int32, len(batch.Positions)), // temporal
			make([]int32, len(batch.Positions)), // height
			make([]int32, len(batch.Positions)), // width
			make([]int32, len(batch.Positions)), // unused (zeros)
		}
		for i, position := range batch.Positions {
			// Translate through position cache or continue sequence
			if position < int32(len(m.TextModel.positionCache)) {
				position = m.TextModel.positionCache[position]
			} else if len(m.TextModel.positionCache) > 0 {
				// Continue sequence after cached positions using ropeDelta
				position = position + m.TextModel.ropeDelta
			}

			s[0][i] = position
			s[1][i] = position
			s[2][i] = position
		}
		return s
	}()

	// Inject vision embeddings and adjust positions for image tokens
	for _, mi := range batch.Multimodal {
		img := mi.Multimodal[0].Tensor
		ctx.Forward(img.Copy(ctx, hiddenStates.View(ctx, mi.Index*hiddenStates.Stride(1), img.Dim(0)*img.Dim(1))))

		if grid, ok := mi.Multimodal[0].Data.(*Grid); ok {
			w := grid.Width / m.VisionModel.spatialMergeSize
			for i := range img.Dim(1) {
				positionSlice[1][mi.Index+i] += int32(i / w)
				positionSlice[2][mi.Index+i] += int32(i % w)
			}
		}
	}

	positions := ctx.Input().FromInts(slices.Concat(positionSlice...), len(positionSlice[0])*len(positionSlice))

	// Process through transformer layers
	for i, layer := range m.TextModel.Layers {
		m.Cache.SetLayer(i)

		var lastLayerOutputs ml.Tensor
		if i == len(m.TextModel.Layers)-1 {
			lastLayerOutputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, lastLayerOutputs, m.Cache, m.TextModel.TextModelOptions)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.TextModel.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func init() {
	model.Register("glmocr", New)
}
