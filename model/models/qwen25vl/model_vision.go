package qwen25vl

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

// VisionSelfAttention implements self-attention for the Qwen vision model
type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

// Forward computes self-attention for the vision model
func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates ml.Tensor, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1), batchSize)
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1), batchSize)
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1), batchSize)

	config := ml.RoPEConfig{
		Dim:        uint32(opts.headDim / 2),
		Type:       ml.RopeTypeMRoPE,
		Base:       opts.ropeTheta,
		Scale:      1.0,
		YarnConfig: ml.DefaultYarnConfig(128000),
	}

	query = query.RoPEMulti(
		ctx,
		positionIDs,
		nil,
		[4]int{opts.headDim / 4, opts.headDim / 4, opts.headDim / 4, opts.headDim / 4},
		config,
	)
	key = key.RoPEMulti(
		ctx,
		positionIDs,
		nil,
		[4]int{opts.headDim / 4, opts.headDim / 4, opts.headDim / 4, opts.headDim / 4},
		config,
	)

	// Scale factor for scaled dot-product attention
	scale := 1.0 / math.Sqrt(float64(opts.headDim))

	attention := nn.Attention(ctx, query, key, value, scale, nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)

	return sa.Output.Forward(ctx, attention)
}

// VisionMLP implements the MLP for the Qwen vision model
type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

// Forward computes the MLP for the vision model
func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	// Using GEGLU activation: (Gate * Up) * GELU(Gate)
	gateOutput := mlp.Gate.Forward(ctx, hiddenStates)
	upOutput := mlp.Up.Forward(ctx, hiddenStates)
	hiddenStates = gateOutput.GELU(ctx).Mul(ctx, upOutput)

	return mlp.Down.Forward(ctx, hiddenStates)
}

// VisionEncoderLayer implements an encoder layer for the Qwen vision model
type VisionEncoderLayer struct {
	Norm1         *nn.RMSNorm `gguf:"ln1"`
	Norm2         *nn.RMSNorm `gguf:"ln2"`
	SelfAttention *VisionSelfAttention
	MLP           *VisionMLP
}

// Forward computes an encoder layer for the vision model
func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates ml.Tensor, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, positionIDs, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = e.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

// VisionModelOptions contains configuration options for the Qwen vision model
type VisionModelOptions struct {
	hiddenSize       int
	numHeads         int
	headDim          int
	intermediateSize int
	imageSize        int
	patchSize        int
	numChannels      int
	eps              float32
	ropeTheta        float32
	outHiddenSize    int
}

type PatchEmbedding struct {
	PatchConv0 *nn.Conv2D `gguf:"patch_embd_0"` // TODO: `gguf:"patch_embed_0"`
	PatchConv1 *nn.Conv2D `gguf:"patch_embd_1"`
}

func (pe *PatchEmbedding) Forward(ctx ml.Context, pixelValues ml.Tensor, numChannels, embedDim, patchSize int) ml.Tensor {
	temporalPatchSize := 2 // we have two temporal convolutions
	numPatches := pixelValues.Shape()[1]

	// Reshape the input tensor to match the expected dimensions
	pixelValues = pixelValues.Reshape(ctx, patchSize*patchSize, temporalPatchSize, numChannels, numPatches)

	// Permute the tensor to bring the temporal dimension to the front
	pixelValues = pixelValues.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// Split the tensor into two parts for the two temporal convolutions
	in0 := pixelValues.View(ctx, 0, 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
	in0 = in0.Reshape(ctx, patchSize, patchSize, numChannels, numPatches)
	in1 := pixelValues.View(ctx, pixelValues.Stride(0), 1, pixelValues.Stride(1), pixelValues.Dim(1), pixelValues.Stride(2), pixelValues.Dim(2), pixelValues.Stride(3), pixelValues.Dim(3)).Contiguous(ctx)
	in1 = in1.Reshape(ctx, patchSize, patchSize, numChannels, numPatches)

	s0, s1 := patchSize, patchSize // Use full stride
	p0, p1 := 0, 0                 // padding
	d0, d1 := 1, 1                 // dilation
	out0 := pe.PatchConv0.Forward(ctx, in0, s0, s1, p0, p1, d0, d1)
	out1 := pe.PatchConv1.Forward(ctx, in1, s0, s1, p0, p1, d0, d1)

	// Add the outputs from the two temporal convolutions
	out := out0.Add(ctx, out1)

	// Reshape the output tensor to match the expected dimensions
	return out.Reshape(ctx, embedDim, numPatches)
}

// VisionPatchMerger implements patch merging for the Qwen vision model
type VisionPatchMerger struct {
	LNQ *nn.RMSNorm `gguf:"ln_q"`
	MLP *nn.Linear  `gguf:"mlp"`
}

// Forward computes patch merging for the vision model
func (pm *VisionPatchMerger) Forward(ctx ml.Context, x ml.Tensor, outDim, contextDim, spatialMergeSize int) ml.Tensor {
	hiddenSize := contextDim * (spatialMergeSize * spatialMergeSize)

	// Normalize and reshape
	x = pm.LNQ.Forward(ctx, x, 1e-6)
	x = x.Reshape(ctx, -1, hiddenSize)

	// Apply MLP for merging
	x = pm.MLP.Forward(ctx, x)

	return x
}

func rope(ctx ml.Context, grid *Grid) ml.Tensor {
	dim := 80 / 2             // TODO: get this from config
	theta := float64(10000.0) // TODO: get this from config ropeTheta
	merge := 2                // Merging factor for spatial dimensions

	// Calculate inverse frequencies for rotation
	inv := freqInv(ctx, dim, theta)

	// Generate and stack position IDs for height and width dimensions
	hPos := heightPos(ctx, grid, merge)
	wPos := widthPos(ctx, grid, merge)
	// Reshape both and stack them
	tmp := hPos.Reshape(ctx, 1, hPos.Dim(0))
	pos := tmp.Stack(ctx, 0, wPos.Reshape(ctx, 1, wPos.Dim(0)))

	// Generate rotary embeddings
	return rotEmbed(ctx, inv, grid.Width, pos)
}

// freqInv calculates the inverse frequencies for rotary embeddings
func freqInv(ctx ml.Context, dim int, theta float64) ml.Tensor {
	logBase, err := ctx.Input().FromFloatSlice([]float32{float32(math.Log(theta))}, 1)
	if err != nil {
		panic(err) // TODO: handle error
	}

	// Create powers divided by dimension (0, 2, 4, ..., dim-2) / dim
	powers := ctx.Arange(0, float32(dim), 2, ml.DTypeF32)
	dims, err := ctx.Input().FromFloatSlice([]float32{float32(dim)}, 1)
	if err != nil {
		panic(err) // TODO: handle error
	}
	powers = powers.Div(ctx, dims)

	// Calculate inverse frequencies: 1 / (theta ^ (powers/dim))
	dims = powers.Mul(ctx, logBase).Exp(ctx)
	ones, err := ctx.Input().FromFloatSlice(slices.Repeat([]float32{1.0}, dims.Shape()[0]), dims.Shape()...)
	if err != nil {
		panic(err) // TODO: handle error
	}
	return ones.Div(ctx, dims)
}

// heightPos generates position IDs for the height dimension
func heightPos(ctx ml.Context, grid *Grid, merge int) ml.Tensor {
	// Create a slice where each row contains the same height value repeated width times
	data := make([]float32, 0, grid.Height*grid.Width)
	for i := 0; i < grid.Height; i++ {
		data = append(data, slices.Repeat([]float32{float32(i)}, grid.Width)...)
	}

	// Create pos with shape [height, width]
	pos, err := ctx.Input().FromFloatSlice(data, grid.Height, grid.Width)
	if err != nil {
		panic(err)
	}

	// Reshape and permute for spatial merging
	pos = pos.Reshape(
		ctx,
		merge,
		grid.Width/merge,
		merge,
		grid.Height/merge,
	)
	pos = pos.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	// Flatten to 1D tensor
	return pos.Reshape(ctx, pos.Dim(0)*pos.Dim(1)*pos.Dim(2)*pos.Dim(3))
}

// widthPos generates position IDs for the width dimension
func widthPos(ctx ml.Context, grid *Grid, merge int) ml.Tensor {
	// Create a slice containing width values in column-major order
	data := make([]float32, 0, grid.Height*grid.Width)
	for i := 0; i < grid.Height; i++ {
		for j := 0; j < grid.Width; j++ {
			data = append(data, float32(j))
		}
	}

	// Create pos with shape [width, height]
	pos, err := ctx.Input().FromFloatSlice(data, grid.Width, grid.Height)
	if err != nil {
		panic(err)
	}

	// Reshape and permute for spatial merging
	pos = pos.Reshape(
		ctx,
		merge,
		grid.Width/merge,
		merge,
		grid.Height/merge,
	)
	pos = pos.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	// Flatten to 1D tensor
	return pos.Reshape(ctx, pos.Dim(0)*pos.Dim(1)*pos.Dim(2)*pos.Dim(3))
}

// rotEmbed generates rotary embeddings using inverse frequencies and position IDs
func rotEmbed(ctx ml.Context, freqInv ml.Tensor, maxSize int, pos ml.Tensor) ml.Tensor {
	// Create sequence tensor [0, 1, 2, ..., maxGridSize-1]
	seq := ctx.Arange(0, float32(maxSize), 1, ml.DTypeF32)

	// Reshape for matrix multiplication and calculate outer product
	outer := freqInv.Reshape(ctx, 1, freqInv.Shape()[0]).Mulmat(ctx, seq.Reshape(ctx, 1, maxSize))

	// Flatten position IDs and use as indices to select rows from outer product
	return outer.Rows(ctx, pos.Reshape(ctx, pos.Dim(0)*pos.Dim(1)))

	// TODO: index position IDs and flatten
}

// VisionModel implements the Qwen vision model
type VisionModel struct {
	PatchEmbedding *PatchEmbedding
	Layers         []VisionEncoderLayer `gguf:"blk"`
	PostLayerNorm  *nn.LayerNorm        `gguf:"post_ln"`
	PatchMerger    *VisionPatchMerger   `gguf:"patch_merger"`

	*VisionModelOptions
}

// Forward computes the vision model for an input tensor
func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, grid *Grid) ml.Tensor {
	// Extract patch embeddings
	hiddenStates := m.PatchEmbedding.Forward(
		ctx,
		pixelValues,   // processed image tensor
		m.numChannels, // number of channels, e.g., 3 for RGB
		m.hiddenSize,  // embedding size
		m.patchSize,   // patch size, e.g., 14
	)

	rope := rope(ctx, grid)

	// spatialMergeSize := 2 // TODO: get this from config
	// // Create the position IDs tensor with correct dimensions
	// positions := []int32{}

	// // Apply encoder layers
	// for _, layer := range m.Layers {
	// 	hiddenStates = layer.Forward(ctx, hiddenStates, positionIDs, m.VisionModelOptions)
	// }

	// hiddenStates = m.PostLayerNorm.Forward(ctx, hiddenStates, m.eps)
	return hiddenStates
}

// newVisionModel creates a new instance of the Qwen vision model
func newVisionModel(c fs.Config) *VisionModel {
	patchSize := int(c.Uint("vision.patch_size", 14))
	hiddenSize := int(c.Uint("vision.embedding_length", 1280))
	ropeTheta := c.Float("vision.rope_theta", 10000.0)             // not set
	outHiddenSize := int(c.Uint("vision.out_embedding_length", 0)) // not set
	numHeads := int(c.Uint("vision.attention.head_count", 16))

	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 24)),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:       hiddenSize,
			numHeads:         numHeads,
			headDim:          hiddenSize / numHeads,
			intermediateSize: int(c.Uint("vision.feed_forward_length", 0)),
			imageSize:        int(c.Uint("vision.image_size", 560)),
			patchSize:        patchSize,
			numChannels:      int(c.Uint("vision.num_channels", 3)), // not set
			eps:              c.Float("vision.attention.layer_norm_epsilon", 1e-6),
			ropeTheta:        ropeTheta,
			outHiddenSize:    outHiddenSize,
		},
	}
}
