package qwen25vl

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

func blockDiagonalMask(ctx ml.Context, seqLength int, bounds []int) ml.Tensor {
	// Initialize a 2D mask with -Inf
	s := make([][]float32, seqLength)
	for i := range s {
		s[i] = slices.Repeat([]float32{float32(math.Inf(-1))}, seqLength)
	}

	// Fill in the mask with zeros for tokens that CAN attend to each other
	for i := 1; i < len(bounds); i++ {
		start, end := bounds[i-1], bounds[i]
		// Enable attention within this sequence block
		for row := start; row < end; row++ {
			for col := start; col < end; col++ {
				s[row][col] = 0.0
			}
		}
	}

	return ctx.Input().FromFloats(slices.Concat(s...), seqLength, seqLength)
}

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1))
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1))
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1))

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

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
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))

	return sa.Output.Forward(ctx, attention)
}

// VisionMLP implements the multi-layer perceptron
type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type VisionEncoderLayer struct {
	Norm1         *nn.RMSNorm `gguf:"ln1"`
	SelfAttention *VisionSelfAttention
	Norm2         *nn.RMSNorm `gguf:"ln2"`
	MLP           *VisionMLP
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, positions, mask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, positions, mask, opts)
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

func (o VisionModelOptions) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, states, positions, o.headDim/2, o.ropeTheta, 1,
		rope.WithVision([]int{
			o.headDim / 4,
			o.headDim / 4,
			o.headDim / 4,
			o.headDim / 4,
		}),
	)
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
	reshaped := normalized.Reshape(ctx, hiddenSize, normalized.Dim(1)/(opts.spatialMergeSize*opts.spatialMergeSize))
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

	index, bounds := m.windowIndex(grid)
	spatialMergeUnit := m.spatialMergeSize * m.spatialMergeSize

	windowIndex := ctx.Input().FromInts(index, len(index))
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)*spatialMergeUnit, hiddenStates.Dim(1)/spatialMergeUnit)
	hiddenStates = hiddenStates.Rows(ctx, windowIndex.Argsort(ctx))
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0)/spatialMergeUnit, hiddenStates.Dim(1)*spatialMergeUnit)

	positions := ctx.Input().FromInts(func() []int32 {
		s := [][]int32{
			make([]int32, grid.Height*grid.Width),
			make([]int32, grid.Height*grid.Width),
			make([]int32, grid.Height*grid.Width),
			make([]int32, grid.Height*grid.Width),
		}

		var cur int
		for y := 0; y < grid.Height; y += m.spatialMergeSize {
			for x := 0; x < grid.Width; x += m.spatialMergeSize {
				for dy := range 2 {
					for dx := range 2 {
						i := int(index[cur/spatialMergeUnit]) * spatialMergeUnit
						i += cur % spatialMergeUnit
						s[0][i] = int32(y + dy)
						s[1][i] = int32(x + dx)
						s[2][i] = int32(y + dy)
						s[3][i] = int32(x + dx)
						cur++
					}
				}
			}
		}

		return slices.Concat(s...)
	}(), grid.Height*grid.Width*4)

	mask := blockDiagonalMask(ctx, hiddenStates.Dim(1), bounds)

	// Apply encoder layers
	for i, layer := range m.Layers {
		if slices.Contains(m.fullAttnBlocks, int32(i)) {
			hiddenStates = layer.Forward(ctx, hiddenStates, positions, nil, m.VisionModelOptions)
		} else {
			hiddenStates = layer.Forward(
				ctx,
				hiddenStates,
				positions,
				mask,
				m.VisionModelOptions,
			)
		}
	}

	hiddenStates = m.PatchMerger.Forward(ctx, hiddenStates, m.VisionModelOptions)
	return hiddenStates.Rows(ctx, windowIndex)
}

// windowIndex divides the grid into windows and returns:
//  1. A slice of grid point indices organized by windows
//  2. A slice of boundaries that mark where each window's data begins and ends
//     in the flattened representation, scaled by spatialMergeSize squared
//
// The boundaries slice always starts with 0 and contains cumulative ending
// positions for each window, allowing downstream processing to identify
// window boundaries in the tensor data.
func (m *VisionModel) windowIndex(grid *Grid) (index []int32, bounds []int) {
	height := grid.Height / m.spatialMergeSize
	width := grid.Width / m.spatialMergeSize
	window := m.windowSize / m.patchSize / m.spatialMergeSize

	index = make([]int32, height*width)

	bounds = make([]int, 0, ((height+window-1)/window)*((width+window-1)/window)+1)
	bounds = append(bounds, 0)

	var cur int32
	for y := 0; y < height; y += window {
		for x := 0; x < width; x += window {
			h1 := min(window, height-y)
			w1 := min(window, width-x)
			for dy := range h1 {
				for dx := range w1 {
					win := (y+dy)*width + (x + dx)
					index[win] = cur
					cur++
				}
			}
			bounds = append(bounds, int(cur)*window)
		}
	}
	return index, bounds
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
