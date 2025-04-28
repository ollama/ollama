package qwen25vl

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.View(ctx, 0, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3))
	x2 := t.View(ctx, t.Stride(0)*t.Dim(0)/2, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3)).Contiguous(ctx)
	return x2.Neg(ctx).Concat(ctx, x1, 0)
}

func applyRotaryPositionalEmbedding(ctx ml.Context, t, cos, sin ml.Tensor) ml.Tensor {
	return t.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, t).Mul(ctx, sin))
}

// VisionSelfAttention implements self-attention for the Qwen vision model
type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

// Forward computes self-attention for the vision model
func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts *VisionModelOptions) ml.Tensor {
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
func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, cos, sin, opts)
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

	positionEmbedding := m.positionalEmbedding(ctx, grid)
	cos, sin := positionEmbedding.Cos(ctx), positionEmbedding.Sin(ctx)

	// Apply encoder layers
	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionModelOptions)
	}

	// hiddenStates = m.PostLayerNorm.Forward(ctx, hiddenStates, m.eps)
	return hiddenStates
}

// positionalEmbedding generates rotary position embeddings for attention mechanisms
// This implements rotary embeddings using spatial merging patterns for grid-based
// vision transformers
func (m *VisionModel) positionalEmbedding(ctx ml.Context, grid *Grid) ml.Tensor {
	// Configuration parameters
	dim := 80 / 2    // Head dimension divided by 2
	freq := dim / 2  // Frequency dimension (half of head dimension)
	theta := 10000.0 // Base for frequency scaling
	merge := 2       // Spatial merge size for rearranging coordinates

	// Create frequency patterns for position encoding
	// These are scaled position values based on frequency
	// In PyTorch: Similar to inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
	freqVals := make([]float32, freq*grid.Width)
	for i := range grid.Width {
		for j := range freq {
			freqVals[i*freq+j] = float32(i) / float32(math.Pow(theta, float64(j*2)/float64(dim)))
		}
	}
	freqs, err := ctx.Input().FromFloatSlice(freqVals, freq, grid.Width)
	if err != nil {
		panic(err) // TODO: handle error
	}

	// Create position coordinates (y,x pairs) for the grid
	// In PyTorch: Equivalent to generating position ids with torch.arange()
	coords := make([]int32, 0, grid.Height*grid.Width*2)
	for y := range grid.Height {
		for x := range grid.Width {
			coords = append(coords, int32(y), int32(x))
		}
	}
	pos, err := ctx.Input().FromIntSlice(coords, 2, grid.Width, grid.Height)
	if err != nil {
		panic(err) // TODO: handle error
	}

	// Reshape and permute positions to match spatial merging pattern
	// This rearranges positions to group spatially related coordinates
	pos = pos.Reshape(ctx, 2, grid.Width, merge, grid.Height/merge)
	pos = pos.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	pos = pos.Reshape(ctx, 2, merge, merge, grid.Width/merge*grid.Height/merge)
	pos = pos.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	pos = pos.Reshape(ctx, 2*merge*merge*grid.Width/merge*grid.Height/merge)

	// Use position indices to look up corresponding frequency values
	positionalEmbedding := freqs.Rows(ctx, pos)
	positionalEmbedding = positionalEmbedding.Reshape(ctx, positionalEmbedding.Dim(0)*2, positionalEmbedding.Dim(1)/2)
	positionalEmbedding = positionalEmbedding.Concat(ctx, positionalEmbedding, 0)

	return positionalEmbedding
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
