//go:build mlx

package gemma3

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/x/ml"
	"github.com/ollama/ollama/x/ml/nn"
)

var batchSize int = 1

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"self_attn.q_proj"`
	Key    *nn.Linear `gguf:"self_attn.k_proj"`
	Value  *nn.Linear `gguf:"self_attn.v_proj"`
	Output *nn.Linear `gguf:"self_attn.out_proj"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	key := sa.Key.Forward(ctx, hiddenState)
	value := sa.Value.Forward(ctx, hiddenState)

	query = query.Reshape(ctx, headDim, opts.numHeads, query.Dim(1), batchSize)
	key = key.Reshape(ctx, headDim, opts.numHeads, key.Dim(1), batchSize)
	value = value.Reshape(ctx, headDim, opts.numHeads, value.Dim(1), batchSize)

	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)

	hiddenState = sa.Output.Forward(ctx, attention)
	return hiddenState
}

type VisionMLP struct {
	FC1 *nn.Linear `gguf:"fc1"`
	FC2 *nn.Linear `gguf:"fc2"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	hiddenState = mlp.FC1.Forward(ctx, hiddenState).GELU(ctx)
	hiddenState = mlp.FC2.Forward(ctx, hiddenState)
	return hiddenState
}

type VisionEncoderLayer struct {
	LayerNorm1    *nn.LayerNorm `gguf:"layer_norm1"`
	SelfAttention *VisionSelfAttention

	LayerNorm2 *nn.LayerNorm `gguf:"layer_norm2"`
	MLP        *VisionMLP    `gguf:"mlp"`
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenState

	// self attention
	hiddenState = e.LayerNorm1.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.SelfAttention.Forward(ctx, hiddenState, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	// feed forward
	hiddenState = e.LayerNorm2.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

type VisionModelOptions struct {
	hiddenSize, numHeads int
	imageSize, patchSize int
	eps                  float32
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv2D    `gguf:"embeddings.patch_embedding"`
	PositionEmbedding *nn.Embedding `gguf:"embeddings.position_embedding"`
	PostLayerNorm     *nn.LayerNorm `gguf:"post_layernorm"`

	Layers []VisionEncoderLayer `gguf:"encoder.layers"`

	*VisionModelOptions
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor) ml.Tensor {
	numPatches := (m.imageSize / m.patchSize) * (m.imageSize / m.patchSize)

	hiddenState := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, numPatches, m.hiddenSize)
	hiddenState = hiddenState.Transpose(ctx, 1, 0, 2, 3).Contiguous(ctx, false)

	positionIDs := ctx.Arange(0, float32(numPatches), 1, ml.DTypeInt32)
	hiddenState = hiddenState.Add(ctx, m.PositionEmbedding.Forward(ctx, positionIDs))

	for _, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, m.VisionModelOptions)
	}

	hiddenState = m.PostLayerNorm.Forward(ctx, hiddenState, m.eps)
	return hiddenState
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count")),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize: int(c.Uint("vision.embedding_length")),
			numHeads:   int(c.Uint("vision.attention.head_count")),

			imageSize: int(c.Uint("vision.image_size")),
			patchSize: int(c.Uint("vision.patch_size")),

			eps: c.Float("vision.attention.layer_norm_epsilon"),
		},
	}
}
