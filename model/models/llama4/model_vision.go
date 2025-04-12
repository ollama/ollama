package llama4

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type VisionAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.View(ctx, 0, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3))
	x2 := t.View(ctx, t.Stride(0)*t.Dim(0)/2, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3)).Contiguous(ctx)
	return x2.Neg(ctx).Concat(ctx, x1, 0)
}

func applyRotaryPositionalEmbedding(ctx ml.Context, t, cos, sin ml.Tensor) ml.Tensor {
	return t.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, t).Mul(ctx, sin))
}

func (sa *VisionAttention) Forward(ctx ml.Context, hiddenState, cos, sin ml.Tensor, opts *VisionOptions) ml.Tensor {
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	key := sa.Key.Forward(ctx, hiddenState)
	value := sa.Value.Forward(ctx, hiddenState)

	query = query.Reshape(ctx, headDim, opts.numHeads, query.Dim(1))
	key = key.Reshape(ctx, headDim, opts.numHeads, key.Dim(1))
	value = value.Reshape(ctx, headDim, opts.numHeads, value.Dim(1))

	query = applyRotaryPositionalEmbedding(ctx, query, cos, sin)
	key = applyRotaryPositionalEmbedding(ctx, key, cos, sin)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2))
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	FC1 *nn.Linear `gguf:"fc1"`
	FC2 *nn.Linear `gguf:"fc2"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionOptions) ml.Tensor {
	hiddenStates = mlp.FC1.Forward(ctx, hiddenStates).GELU(ctx)
	hiddenStates = mlp.FC2.Forward(ctx, hiddenStates)
	return hiddenStates
}

type VisionLayer struct {
	InputLayerNorm *nn.LayerNorm `gguf:"attn_norm"`
	*VisionAttention

	PostAttentionNorm *nn.LayerNorm `gguf:"ffn_norm"`
	*VisionMLP        `gguf:"mlp"`
}

func (e *VisionLayer) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts *VisionOptions) ml.Tensor {
	residual := hiddenStates

	// self attention
	hiddenStates = e.InputLayerNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.VisionAttention.Forward(ctx, hiddenStates, cos, sin, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	// MLP
	residual = hiddenStates
	hiddenStates = e.PostAttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.VisionMLP.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	return hiddenStates
}

type VisionAdapter struct {
	FC1 *nn.Linear `gguf:"mlp.fc1"`
	FC2 *nn.Linear `gguf:"mlp.fc2"`
}

func (a *VisionAdapter) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionOptions) ml.Tensor {
	channels, numPatches, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)
	patchSize := int(math.Sqrt(float64(numPatches)))

	hiddenStates = hiddenStates.Reshape(ctx, channels, patchSize, patchSize, batchSize)
	channels, width, height, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2), hiddenStates.Dim(3)

	hiddenStates = hiddenStates.Reshape(ctx, int(float32(channels)/opts.pixelShuffleRatio), int(float32(width)*opts.pixelShuffleRatio), height, batchSize)
	hiddenStates = hiddenStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	hiddenStates = hiddenStates.Reshape(ctx, int(float32(channels)/opts.pixelShuffleRatio/opts.pixelShuffleRatio), int(float32(width)*opts.pixelShuffleRatio), int(float32(height)/opts.pixelShuffleRatio), batchSize)
	hiddenStates = hiddenStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	hiddenStates = hiddenStates.Reshape(ctx, channels, numPatches, batchSize)

	hiddenStates = a.FC1.Forward(ctx, hiddenStates).GELU(ctx)
	hiddenStates = a.FC2.Forward(ctx, hiddenStates).GELU(ctx)
	return hiddenStates
}

type VisionOptions struct {
	hiddenSize, numHeads, headDim int
	imageSize, patchSize          int

	ropeTheta         float32
	eps               float32
	pixelShuffleRatio float32
}

type VisionModel struct {
	Layers []VisionLayer `gguf:"blk"`

	PatchEmbedding      *nn.Conv2D `gguf:"patch_embedding"`
	ClassEmbedding      ml.Tensor  `gguf:"class_embedding"`
	PositionalEmbedding ml.Tensor  `gguf:"position_embedding"`

	LayerNormPost *nn.LayerNorm `gguf:"layernorm_post"`
	LayerNormPre  *nn.LayerNorm `gguf:"layernorm_pre"`

	*VisionAdapter `gguf:"vision_adapter"`

	*VisionOptions
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Layers: make([]VisionLayer, c.Uint("vision.block_count")),
		VisionOptions: &VisionOptions{
			hiddenSize:        int(c.Uint("vision.embedding_length")),
			numHeads:          int(c.Uint("vision.attention.head_count")),
			imageSize:         int(c.Uint("vision.image_size")),
			patchSize:         int(c.Uint("vision.patch_size")),
			ropeTheta:         float32(c.Float("vision.rope.freq_base")),
			eps:               c.Float("vision.layer_norm_epsilon"),
			pixelShuffleRatio: float32(c.Float("vision.pixel_shuffle_ratio")),
		},
	}
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor) ml.Tensor {
	// TODO: this may need to be change to im2col, permute, nn.Linear
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, 16, 16, 0, 0, 1, 1)
	hiddenStates = hiddenStates.Reshape(ctx)

	hiddenStates = hiddenStates.Concat(ctx, m.ClassEmbedding.Repeat(ctx, 2, hiddenStates.Dim(1)), 0)
	hiddenStates = hiddenStates.Reshape(ctx)

	hiddenStates = hiddenStates.Add(ctx, m.PositionalEmbedding)
	hiddenStates = m.LayerNormPre.Forward(ctx, hiddenStates, m.eps)

	cos, sin := m.rotaryEmbedding(ctx, pixelValues)
	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionOptions)
	}

	hiddenStates = m.LayerNormPost.Forward(ctx, hiddenStates, m.eps)
	hiddenStates = m.VisionAdapter.Forward(ctx, hiddenStates, m.VisionOptions)
	return hiddenStates
}

// floorDiv is a helper function to perform floor division. This mimics PyTorch's div(round_mode='floor') function
// which in turn mimics Python's // operator.
func floorDiv[T int | int16 | int32 | int64 | uint | uint16 | uint32 | uint64](a, b T) T {
	if b == 0 {
		panic("division by zero")
	}

	if (a >= 0 && b > 0) || (a <= 0 && b < 0) || a%b == 0 {
		return a / b
	}

	return a/b - 1
}

func (m *VisionModel) rotaryEmbedding(ctx ml.Context, pixelValues ml.Tensor) (ml.Tensor, ml.Tensor) {
	patchesPerSide := m.imageSize / m.patchSize
	numPatches := patchesPerSide*patchesPerSide + 1
	headDim := m.hiddenSize / m.numHeads

	freqsX, freqsY := make([]float32, numPatches*headDim/2), make([]float32, numPatches*headDim/2)
	for i := range numPatches {
		if i >= patchesPerSide*patchesPerSide {
			// ID_CLS_TOKEN
			break
		}

		for j := range headDim / 2 {
			ropeFreq := math.Pow(float64(m.ropeTheta), float64(j)*2/float64(headDim))
			freqsX[int(i)*headDim/2+j] = float32(float64(1+i-floorDiv(i, patchesPerSide)*patchesPerSide) / ropeFreq)
			freqsY[int(i)*headDim/2+j] = float32(float64(1+floorDiv(i, patchesPerSide)) / ropeFreq)
		}
	}

	ropeFreqs, err := ctx.FromFloatSlice(append(freqsX, freqsY...), headDim/2, numPatches, 2)
	if err != nil {
		panic(err)
	}

	ropeFreqs = ropeFreqs.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx).Reshape(ctx, headDim, numPatches)
	return ropeFreqs.Cos(ctx), ropeFreqs.Sin(ctx)
}
