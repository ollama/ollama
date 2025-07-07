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

// applyVisionRotaryEmbedding applies 2D rotary embedding to the input tensor.
// This is equivalent to the Pytorch implmentation using half rotations:
//
//	cos, sin = torch.cos(freqs), torch.sin(freqs)
//	cos = cos.unsqueeze(-1)
//	sin = sin.unsqueeze(-1)
//	t = t.reshape(*t.shape[:-1], -1, 2)
//	t_out = (t * cos) + (_rotate_half(t) * sin)
//	t_out = t_out.flatten(3)
//
// Which is equivalent to the Pytorch implementation using complex numbers:
//
//	t_ = torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))
//	freqs_ci = reshape_for_broadcast(freqs_ci=freq_cis, t=t_)  # freqs_ci[:,:,None,:]
//	freqs_ci = freqs_ci.to(t_.device)
//	t_out = torch.view_as_real(t_ * freqs_ci).flatten(3)
//
// Due to the 1) the dimensional and 2) the datatype limitations of current backends,
// we need to use a different approach to achieve the same result.
func applyVisionRotaryEmbedding(ctx ml.Context, t, cos, sin ml.Tensor) ml.Tensor {
	width, height, channels, tiles := t.Dim(0), t.Dim(1), t.Dim(2), t.Dim(3)

	t = t.Reshape(ctx, 2, t.Dim(0)/2, t.Dim(1)*t.Dim(2)*t.Dim(3))

	// t1 = t[..., 0::2]
	t1 := t.View(ctx, 0, 1, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2)).Contiguous(ctx)
	t1 = t1.Reshape(ctx, width/2, height, channels, tiles)

	// t2 = t[..., 1::2]
	t2 := t.View(ctx, t.Stride(0), 1, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2)).Contiguous(ctx)
	t2 = t2.Reshape(ctx, width/2, height, channels, tiles)

	// cos_out = torch.stack((t1 * cos, t2 * cos), dim=-1)
	cosOut := t1.Mul(ctx, cos).Concat(ctx, t2.Mul(ctx, cos), 0)
	cosOut = cosOut.Reshape(ctx, cosOut.Dim(0)/2, 2, cosOut.Dim(1)*cosOut.Dim(2)*cosOut.Dim(3))
	cosOut = cosOut.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	cosOut = cosOut.Reshape(ctx, width, height, channels, tiles)

	// sin_out = torch.stack((-t2 * sin, t1 * sin), dim=-1)
	sinOut := t2.Neg(ctx).Mul(ctx, sin).Concat(ctx, t1.Mul(ctx, sin), 0)
	sinOut = sinOut.Reshape(ctx, sinOut.Dim(0)/2, 2, sinOut.Dim(1)*sinOut.Dim(2)*sinOut.Dim(3))
	sinOut = sinOut.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	sinOut = sinOut.Reshape(ctx, width, height, channels, tiles)

	return cosOut.Add(ctx, sinOut)
}

func (sa *VisionAttention) Forward(ctx ml.Context, hiddenState, cos, sin ml.Tensor, opts *VisionOptions) ml.Tensor {
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	key := sa.Key.Forward(ctx, hiddenState)
	value := sa.Value.Forward(ctx, hiddenState)

	query = query.Reshape(ctx, headDim, opts.numHeads, query.Dim(1), query.Dim(2))
	key = key.Reshape(ctx, headDim, opts.numHeads, key.Dim(1), key.Dim(2))
	value = value.Reshape(ctx, headDim, opts.numHeads, value.Dim(1), value.Dim(2))

	query = applyVisionRotaryEmbedding(ctx, query, cos, sin)
	key = applyVisionRotaryEmbedding(ctx, key, cos, sin)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), attention.Dim(3))
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
	patches := hiddenStates.Dim(1)
	patchSize := int(math.Sqrt(float64(patches)))

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), patchSize, patchSize, hiddenStates.Dim(2))

	channels, width, height, tiles := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2), hiddenStates.Dim(3)

	channels, width = int(float32(channels)/opts.pixelShuffleRatio), int(float32(width)*opts.pixelShuffleRatio)
	hiddenStates = hiddenStates.Reshape(ctx, channels, width, height, tiles)
	hiddenStates = hiddenStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	channels, height = int(float32(channels)/opts.pixelShuffleRatio), int(float32(height)*opts.pixelShuffleRatio)
	hiddenStates = hiddenStates.Reshape(ctx, channels, width, height, tiles)
	hiddenStates = hiddenStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	hiddenStates = hiddenStates.Reshape(ctx, channels, width*height, tiles)

	hiddenStates = a.FC1.Forward(ctx, hiddenStates).GELU(ctx)
	hiddenStates = a.FC2.Forward(ctx, hiddenStates).GELU(ctx)
	return hiddenStates
}

type VisionOptions struct {
	hiddenSize, numHeads int
	imageSize, patchSize int

	ropeTheta         float32
	eps               float32
	pixelShuffleRatio float32
}

type PatchEmbedding struct {
	*nn.Linear
}

func (p *PatchEmbedding) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionOptions) ml.Tensor {
	kernel := ctx.Input().Empty(ml.DTypeF32, opts.patchSize, opts.patchSize, hiddenStates.Dim(2))
	hiddenStates = kernel.IM2Col(ctx, hiddenStates, opts.patchSize, opts.patchSize, 0, 0, 1, 1)
	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), hiddenStates.Dim(1)*hiddenStates.Dim(2), hiddenStates.Dim(3))
	return p.Linear.Forward(ctx, hiddenStates)
}

type VisionModel struct {
	Layers []VisionLayer `gguf:"blk"`

	*PatchEmbedding     `gguf:"patch_embedding"`
	ClassEmbedding      ml.Tensor `gguf:"class_embedding"`
	PositionalEmbedding ml.Tensor `gguf:"positional_embedding_vlm"`

	LayerNormPre  *nn.LayerNorm `gguf:"layernorm_pre"`
	LayerNormPost *nn.LayerNorm `gguf:"layernorm_post"`

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
	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.VisionOptions)
	hiddenStates = hiddenStates.Concat(ctx, m.ClassEmbedding.Repeat(ctx, 2, hiddenStates.Dim(2)), 1)

	hiddenStates = hiddenStates.Add(ctx, m.PositionalEmbedding)
	hiddenStates = m.LayerNormPre.Forward(ctx, hiddenStates, m.eps)

	cos, sin := m.rotaryEmbedding(ctx)
	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionOptions)
	}

	hiddenStates = m.LayerNormPost.Forward(ctx, hiddenStates, m.eps)
	hiddenStates = hiddenStates.Pad(ctx, 0, -1, 0, 0)
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

func (m *VisionModel) rotaryEmbedding(ctx ml.Context) (ml.Tensor, ml.Tensor) {
	patchesPerSide := m.imageSize / m.patchSize
	numPatches := patchesPerSide*patchesPerSide + 1

	headDim := m.hiddenSize / m.numHeads
	freqDim := headDim / 2

	freqs := make([]float32, numPatches*freqDim)
	for i := range numPatches - 1 {
		for j := 0; j < freqDim; j += 2 {
			positionX := i*freqDim/2 + j/2
			positionY := (i+numPatches)*freqDim/2 + j/2
			ropeFreq := math.Pow(float64(m.ropeTheta), float64(j)*2/float64(headDim))
			freqs[positionX] = float32(float64(1+i-floorDiv(i, patchesPerSide)*patchesPerSide) / ropeFreq)
			freqs[positionY] = float32(float64(1+floorDiv(i, patchesPerSide)) / ropeFreq)
		}
	}

	ropeFreqs := ctx.Input().FromFloatSlice(freqs, freqDim/2, numPatches, 2)

	ropeFreqs = ropeFreqs.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	ropeFreqs = ropeFreqs.Reshape(ctx, freqDim, 1, numPatches)
	return ropeFreqs.Cos(ctx), ropeFreqs.Sin(ctx)
}
