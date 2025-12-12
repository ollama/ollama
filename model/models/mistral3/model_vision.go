package mistral3

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.Slice(ctx, 0, 0, t.Dim(0)/2, 1)
	x2 := t.Slice(ctx, 0, t.Dim(0)/2, t.Dim(0), 1).Contiguous(ctx)
	return x2.Scale(ctx, -1).Concat(ctx, x1, 0)
}

func applyRotaryPositionEmbeddings(ctx ml.Context, states, cos, sin ml.Tensor) ml.Tensor {
	return states.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, states).Mul(ctx, sin))
}

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1), batchSize)
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1), batchSize)
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1), batchSize)

	query = applyRotaryPositionEmbeddings(ctx, query, cos, sin)
	key = applyRotaryPositionEmbeddings(ctx, key, cos, sin)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(opts.headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)
	return sa.Output.Forward(ctx, attention)
}

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
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *VisionSelfAttention
	FFNNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *VisionMLP
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = e.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.SelfAttention.Forward(ctx, hiddenStates, cos, sin, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = e.FFNNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = e.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type VisionModelOptions struct {
	hiddenSize       int
	numHeads         int
	headDim          int
	intermediateSize int
	imageSize        int
	patchSize        int
	numChannels      int
	eps              float32
	ropeBase         float32
}

type VisionModel struct {
	PatchEmbedding *nn.Conv2D           `gguf:"patch_conv"`
	EncoderNorm    *nn.RMSNorm          `gguf:"encoder_norm"`
	Layers         []VisionEncoderLayer `gguf:"blk"`

	*VisionModelOptions
}

func (m *VisionModel) positionalEmbedding(ctx ml.Context, positionIDs ml.Tensor) ml.Tensor {
	maxPatchesPerSide := m.imageSize / m.patchSize
	frequencies := m.headDim / 2
	frequenciesHeight := make([]float32, frequencies/2*maxPatchesPerSide)
	frequenciesWidth := make([]float32, frequencies/2*maxPatchesPerSide)
	for i := range frequencies {
		for j := range maxPatchesPerSide {
			frequency := float32(j) / float32(math.Pow(float64(m.ropeBase), float64(i)*2/float64(m.headDim)))
			if i%2 == 0 {
				frequenciesHeight[i/2*maxPatchesPerSide+j] = frequency
			} else {
				frequenciesWidth[i/2*maxPatchesPerSide+j] = frequency
			}
		}
	}

	h := ctx.Input().FromFloats(frequenciesHeight, maxPatchesPerSide, frequencies/2)
	w := ctx.Input().FromFloats(frequenciesWidth, maxPatchesPerSide, frequencies/2)

	h = h.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	w = w.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	h = h.Repeat(ctx, 1, maxPatchesPerSide)
	h = h.Reshape(ctx, frequencies/2, maxPatchesPerSide, maxPatchesPerSide).Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	w = w.Repeat(ctx, 2, maxPatchesPerSide)

	inverseFrequencies := h.Concat(ctx, w, 0).Reshape(ctx, frequencies, maxPatchesPerSide*maxPatchesPerSide)
	inverseFrequencies = inverseFrequencies.Concat(ctx, inverseFrequencies, 0)
	return inverseFrequencies.Rows(ctx, positionIDs)
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor) ml.Tensor {
	numPatchesW := pixelValues.Dim(0) / m.patchSize
	numPatchesH := pixelValues.Dim(1) / m.patchSize
	numPatches := numPatchesW * numPatchesH

	hiddenStates := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenStates = hiddenStates.Reshape(ctx, numPatches, m.hiddenSize)
	hiddenStates = hiddenStates.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	hiddenStates = m.EncoderNorm.Forward(ctx, hiddenStates, m.VisionModelOptions.eps)

	// Prepare position IDs for 2D rope
	positions := make([]int32, numPatches)
	for h := range numPatchesH {
		for w := range numPatchesW {
			idx := h*numPatchesW + w
			positions[idx] = int32(h*m.imageSize/m.patchSize + w)
		}
	}

	positionIDs := ctx.Input().FromInts(positions, len(positions))

	positionEmbedding := m.positionalEmbedding(ctx, positionIDs)
	cos, sin := positionEmbedding.Cos(ctx), positionEmbedding.Sin(ctx)
	cos = cos.Reshape(ctx, cos.Dim(0), 1, cos.Dim(1))
	sin = sin.Reshape(ctx, sin.Dim(0), 1, sin.Dim(1))

	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionModelOptions)
	}

	return hiddenStates
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count")),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize:       int(c.Uint("vision.embedding_length", 1024)),
			numHeads:         int(c.Uint("vision.attention.head_count", 16)),
			headDim:          int(c.Uint("vision.attention.key_length", 64)),
			intermediateSize: int(c.Uint("vision.feed_forward_length", 4096)),
			imageSize:        int(c.Uint("vision.image_size", 1540)),
			patchSize:        int(c.Uint("vision.patch_size", 14)),
			numChannels:      int(c.Uint("vision.num_channels", 3)),
			eps:              c.Float("vision.attention.layer_norm_epsilon", 1e-5),
			ropeBase:         c.Float("vision.rope.freq_base", 10000.0),
		},
	}
}
