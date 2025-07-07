package qwen25vl

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// We only support batch size of 1
var batchSize int = 1

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.View(ctx, 0, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3))
	x2 := t.View(ctx, t.Stride(0)*t.Dim(0)/2, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3)).Contiguous(ctx)
	return x2.Neg(ctx).Concat(ctx, x1, 0)
}

func applyRotaryPositionalEmbedding(ctx ml.Context, t, cos, sin ml.Tensor) ml.Tensor {
	return t.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, t).Mul(ctx, sin))
}

func blockDiagonalMask(ctx ml.Context, seqLength int, bounds []int, numHeads int) ml.Tensor {
	// Create a flat slice for the mask (all -inf initially to block all attention)
	flat := make([]float32, seqLength*seqLength)
	for i := range flat {
		flat[i] = float32(math.Inf(-1)) // Negative infinity to block attention
	}

	// Fill in the mask with zeros for tokens that CAN attend to each other
	for i := 1; i < len(bounds); i++ {
		start := bounds[i-1]
		end := bounds[i]

		// Enable attention within this sequence block by setting values to 0
		for row := start; row < end; row++ {
			for col := start; col < end; col++ {
				idx := row*seqLength + col
				flat[idx] = 0.0 // 0 allows attention, -inf blocks it
			}
		}
	}

	mask := ctx.Input().FromFloatSlice(flat, seqLength, seqLength)

	// Reshape to match [seqLength, seqLength, 1] for broadcasting
	mask = mask.Reshape(ctx, seqLength, seqLength, 1)

	return mask
}

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, cos, sin, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1), batchSize)
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1), batchSize)
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1), batchSize)

	query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)
	key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)

	// Scale factor for scaled dot-product attention
	scale := 1.0 / math.Sqrt(float64(opts.headDim))

	// Scaled dot-product attention
	query = query.Permute(ctx, 0, 2, 1, 3)
	key = key.Permute(ctx, 0, 2, 1, 3)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
	kq := key.MulmatFullPrec(ctx, query)
	kq = kq.Scale(ctx, scale)
	if mask != nil {
		kq = kq.Add(ctx, mask)
	}
	kq = kq.Softmax(ctx)
	kqv := value.Mulmat(ctx, kq)
	attention := kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)

	return sa.Output.Forward(ctx, attention)
}

// VisionMLP implements the multi-layer perceptron
type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	// Using activation as specified in config (likely GELU or SiLU/Swish)
	gateOutput := mlp.Gate.Forward(ctx, hiddenStates)
	upOutput := mlp.Up.Forward(ctx, hiddenStates)
	hiddenStates = gateOutput.SILU(ctx).Mul(ctx, upOutput)

	return mlp.Down.Forward(ctx, hiddenStates)
}

type VisionEncoderLayer struct {
	Norm1         *nn.RMSNorm `gguf:"ln1"`
	SelfAttention *VisionSelfAttention
	Norm2         *nn.RMSNorm `gguf:"ln2"`
	MLP           *VisionMLP
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, cos, sin, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, cos, sin, mask, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = e.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

// VisionModelOptions contains configuration options
type VisionModelOptions struct {
	hiddenSize        int
	numHeads          int
	headDim           int
	patchSize         int
	numChannels       int
	eps               float32
	ropeTheta         float32
	spatialMergeSize  int
	windowSize        int
	fullAttnBlocks    []int32
	temporalPatchSize int
}

type PatchEmbedding struct {
	PatchConv0 *nn.Conv2D `gguf:"patch_embd_0"`
	PatchConv1 *nn.Conv2D `gguf:"patch_embd_1"`
}

func (pe *PatchEmbedding) Forward(ctx ml.Context, pixelValues ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	numPatches := pixelValues.Shape()[1]

	// Reshape the input tensor to match the expected dimensions
	pixelValues = pixelValues.Reshape(ctx, opts.patchSize*opts.patchSize, opts.temporalPatchSize, opts.numChannels, numPatches)

	// Permute the tensor to bring the temporal dimension to the front
	pixelValues = pixelValues.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// Split the tensor into parts for the temporal convolutions
	in0 := pixelValues.View(ctx, 0, 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
	in0 = in0.Reshape(ctx, opts.patchSize, opts.patchSize, opts.numChannels, numPatches)
	in1 := pixelValues.View(ctx, pixelValues.Stride(0), 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
	in1 = in1.Reshape(ctx, opts.patchSize, opts.patchSize, opts.numChannels, numPatches)

	s0, s1 := opts.patchSize, opts.patchSize // Use full stride
	p0, p1 := 0, 0                           // padding
	d0, d1 := 1, 1                           // dilation
	out0 := pe.PatchConv0.Forward(ctx, in0, s0, s1, p0, p1, d0, d1)
	out1 := pe.PatchConv1.Forward(ctx, in1, s0, s1, p0, p1, d0, d1)

	// Add the outputs from the two temporal convolutions
	out := out0.Add(ctx, out1)

	// Reshape the output tensor to match the expected dimensions
	return out.Reshape(ctx, opts.hiddenSize, numPatches)
}

// VisionPatchMerger implements patch merging for the Qwen vision model
type VisionPatchMerger struct {
	LNQ  *nn.RMSNorm `gguf:"ln_q"`
	MLP0 *nn.Linear  `gguf:"mlp.0"`
	MLP2 *nn.Linear  `gguf:"mlp.2"`
}

// Forward computes patch merging for the vision model
func (pm *VisionPatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	normalized := pm.LNQ.Forward(ctx, visionOutputs, opts.eps)

	hiddenSize := visionOutputs.Dim(0) * (opts.spatialMergeSize * opts.spatialMergeSize)

	// Reshape the normalized output to view the hidden size dimension
	reshaped := normalized.Reshape(ctx, hiddenSize, normalized.Dim(1)/(opts.spatialMergeSize*opts.spatialMergeSize), batchSize)
	hidden := pm.MLP0.Forward(ctx, reshaped)
	activated := hidden.GELU(ctx)

	output := pm.MLP2.Forward(ctx, activated)

	return output
}

// VisionModel implements the Qwen vision model
type VisionModel struct {
	PatchEmbedding *PatchEmbedding
	Layers         []VisionEncoderLayer `gguf:"blk"`
	PatchMerger    *VisionPatchMerger   `gguf:"merger"`

	*VisionModelOptions
}

// Forward computes the vision model for an input tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) ml.Tensor {
	// Extract patch embeddings
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.VisionModelOptions)

	positionEmbedding := m.PositionalEmbedding(ctx, grid)

	windowIndex, bounds := m.WindowIndex(ctx, grid)

	spatialMergeUnit := m.spatialMergeSize * m.spatialMergeSize

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)*spatialMergeUnit, hiddenStates.Dim(1)/spatialMergeUnit)
	hiddenStates = hiddenStates.Rows(ctx, windowIndex)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)/spatialMergeUnit, hiddenStates.Dim(1)*spatialMergeUnit)

	positionEmbedding = positionEmbedding.Reshape(ctx, positionEmbedding.Dim(0)*spatialMergeUnit, positionEmbedding.Dim(1)/spatialMergeUnit)
	positionEmbedding = positionEmbedding.Rows(ctx, windowIndex)
	positionEmbedding = positionEmbedding.Reshape(ctx, positionEmbedding.Dim(0)/spatialMergeUnit, positionEmbedding.Dim(1)*spatialMergeUnit)
	positionEmbedding = positionEmbedding.Concat(ctx, positionEmbedding, 0)

	cos, sin := positionEmbedding.Cos(ctx), positionEmbedding.Sin(ctx)
	cos = cos.Reshape(ctx, cos.Dim(0), 1, cos.Dim(1))
	sin = sin.Reshape(ctx, sin.Dim(0), 1, sin.Dim(1))

	mask := blockDiagonalMask(ctx, hiddenStates.Dim(1), bounds, m.VisionModelOptions.numHeads)
	// Apply encoder layers
	for i, layer := range m.Layers {
		if slices.Contains(m.fullAttnBlocks, int32(i)) {
			hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, nil, m.VisionModelOptions)
		} else {
			hiddenStates = layer.Forward(
				ctx,
				hiddenStates,
				cos,
				sin,
				mask,
				m.VisionModelOptions,
			)
		}
	}

	hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, m.VisionModelOptions)
	reverseWindowIndex := windowIndex.Argsort(ctx)
	return hiddenStates.Rows(ctx, reverseWindowIndex)
}

// WindowIndex divides the grid into windows and returns:
//  1. A tensor containing flattened indices of all grid points organized by windows
//  2. A slice of boundaries that mark where each window's data begins and ends
//     in the flattened representation, scaled by spatialMergeSize squared
//
// The boundaries slice always starts with 0 and contains cumulative ending
// positions for each window, allowing downstream processing to identify
// window boundaries in the tensor data.
func (m *VisionModel) WindowIndex(ctx ml.Context, grid *Grid) (ml.Tensor, []int) {
	vitMergerWindowSize := m.windowSize / m.spatialMergeSize / m.patchSize

	llmGridH := grid.Height / m.spatialMergeSize
	llmGridW := grid.Width / m.spatialMergeSize

	// Calculate window parameters
	numWindowsH := int(math.Ceil(float64(llmGridH) / float64(vitMergerWindowSize)))
	numWindowsW := int(math.Ceil(float64(llmGridW) / float64(vitMergerWindowSize)))

	// Initialize index_new slice
	var index []int32

	// Initialize bounds with the first element as 0
	bounds := []int{0}
	totalSeqLen := 0

	// Process each window without padding
	for wh := range numWindowsH {
		for ww := range numWindowsW {
			// Calculate window boundaries
			hStart := wh * vitMergerWindowSize
			wStart := ww * vitMergerWindowSize
			hEnd := min(hStart+vitMergerWindowSize, llmGridH)
			wEnd := min(wStart+vitMergerWindowSize, llmGridW)

			// Calculate sequence length for this window
			seqLen := (hEnd - hStart) * (wEnd - wStart)

			// Collect indices for this window
			for h := hStart; h < hEnd; h++ {
				for w := wStart; w < wEnd; w++ {
					index = append(index, int32(h*llmGridW+w))
				}
			}

			totalSeqLen += seqLen
			bounds = append(bounds, totalSeqLen*(m.spatialMergeSize*m.spatialMergeSize)+bounds[0])
		}
	}

	t := ctx.Input().FromIntSlice(index, len(index))

	return t, bounds
}

// PositionalEmbedding generates rotary position embeddings for attention mechanisms
func (m *VisionModel) PositionalEmbedding(ctx ml.Context, grid *Grid) ml.Tensor {
	dim := m.headDim / 2
	freq := dim / 2
	theta := float64(m.ropeTheta)
	merge := m.spatialMergeSize

	// Create frequency patterns for position encoding
	maxGridSize := max(grid.Height, grid.Width)
	freqVals := make([]float32, freq*maxGridSize)
	for i := range maxGridSize {
		for j := range freq {
			freqVals[i*freq+j] = float32(i) / float32(math.Pow(theta, float64(j*2)/float64(dim)))
		}
	}
	freqs := ctx.Input().FromFloatSlice(freqVals, freq, maxGridSize)

	// Create position coordinates (y,x pairs) for the grid
	// In PyTorch: Equivalent to generating position ids with torch.arange()
	coords := make([]int32, 0, grid.Height*grid.Width*2)
	for y := range grid.Height {
		for x := range grid.Width {
			coords = append(coords, int32(y), int32(x))
		}
	}
	pos := ctx.Input().FromIntSlice(coords, 2, grid.Width, grid.Height)

	// Reshape and permute positions to match spatial merging pattern
	pos = pos.Reshape(ctx, 2, grid.Width, merge, grid.Height/merge)
	pos = pos.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	pos = pos.Reshape(ctx, 2, merge, merge, grid.Width/merge*grid.Height/merge)
	pos = pos.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	pos = pos.Reshape(ctx, 2*merge*merge*grid.Width/merge*grid.Height/merge)

	// Use position indices to look up corresponding frequency values
	positionalEmbedding := freqs.Rows(ctx, pos)
	positionalEmbedding = positionalEmbedding.Reshape(ctx, positionalEmbedding.Dim(0)*2, positionalEmbedding.Dim(1)/2)
	return positionalEmbedding
}

// newVisionModel creates a new instance of the Qwen vision model
func newVisionModel(c fs.Config) *VisionModel {
	patchSize := int(c.Uint("vision.patch_size", 14))
	hiddenSize := int(c.Uint("vision.embedding_length", 1280))
	numHeads := int(c.Uint("vision.attention.head_count", 16))
	numChannels := int(c.Uint("vision.num_channels", 3))
	eps := c.Float("vision.attention.layer_norm_epsilon", 1e-6)
	ropeTheta := c.Float("vision.rope.freq_base", 10000.0)
	spatialMergeSize := int(c.Uint("vision.spatial_merge_size", 2))
	windowSize := int(c.Uint("vision.window_size", 112))
	fullAttnBlocks := c.Ints("qwen25vl.vision.fullatt_block_indexes", []int32{7, 15, 23, 31})
	temporalPatchSize := int(c.Uint("vision.temporal_patch_size", 2))

	model := &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 32)),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:        hiddenSize,
			numHeads:          numHeads,
			headDim:           hiddenSize / numHeads,
			patchSize:         patchSize,
			numChannels:       numChannels,
			eps:               eps,
			ropeTheta:         ropeTheta,
			spatialMergeSize:  spatialMergeSize,
			windowSize:        windowSize,
			temporalPatchSize: temporalPatchSize,
			fullAttnBlocks:    fullAttnBlocks,
		},
	}

	return model
}
