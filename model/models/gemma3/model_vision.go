package gemma3

import (
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	key := sa.Key.Forward(ctx, hiddenState)
	value := sa.Value.Forward(ctx, hiddenState)

	query = query.Reshape(ctx, headDim, opts.numHeads, query.Dim(1), batchSize).Permute(ctx, 0, 2, 1, 3)
	key = key.Reshape(ctx, headDim, opts.numHeads, key.Dim(1), batchSize).Permute(ctx, 0, 2, 1, 3)
	value = value.Reshape(ctx, headDim, opts.numHeads, value.Dim(1), batchSize).Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	scores := key.Mulmat(ctx, query)
	scores = scores.Scale(ctx, 1.0/math.Sqrt(float64(headDim)))
	scores = scores.Softmax(ctx)

	attention := value.Mulmat(ctx, scores)
	attention = attention.Reshape(ctx, headDim, attention.Dim(1), opts.numHeads, batchSize)
	attention = attention.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
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

type VisionEncoder struct {
	Layers []VisionEncoderLayer
}

func (e *VisionEncoder) Forward(ctx ml.Context, hiddenState ml.Tensor, intermediateLayersIndices []uint32, opts *VisionModelOptions) (ml.Tensor, []ml.Tensor) {
	var intermediateHiddenStates []ml.Tensor
	for i, layer := range e.Layers {
		if slices.Contains(intermediateLayersIndices, uint32(i)) {
			intermediateHiddenStates = append(intermediateHiddenStates, hiddenState.Reshape(ctx, append([]int{1}, hiddenState.Shape()...)...))
		}

		hiddenState = layer.Forward(ctx, hiddenState, opts)
	}

	return hiddenState, intermediateHiddenStates
}

type PrecomputedAspectRatioEmbedding struct {
	Embedding *nn.Embedding
	Gate      ml.Tensor `gguf:"gate"`
}

func (e *PrecomputedAspectRatioEmbedding) Forward(ctx ml.Context, hiddenState ml.Tensor, aspectRatioIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	embeddings := e.Embedding.Forward(ctx, aspectRatioIDs)
	embeddings = embeddings.Reshape(ctx, opts.hiddenSize, 1, opts.numTiles)
	if e.Gate != nil {
		embeddings = embeddings.Mul(ctx, e.Gate)
	}

	return hiddenState.Add(ctx, embeddings)
}

type PrecomputedPositionEmbedding struct {
	PositionEmbedding     *nn.Embedding `gguf:"position_embd"`
	PositionEmbeddingGate ml.Tensor     `gguf:"position_embd.gate"`
}

func (e *PrecomputedPositionEmbedding) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, numPositions int, opts *VisionModelOptions) ml.Tensor {
	positionEmbedding := e.PositionEmbedding.Forward(ctx, positionIDs)
	if e.PositionEmbeddingGate != nil {
		positionEmbedding = positionEmbedding.Mul(ctx, e.PositionEmbeddingGate)
	}

	return hiddenState.Add(ctx, positionEmbedding)
}

type VisionModelOptions struct {
	hiddenSize, numHeads, numTiles int
	imageSize, patchSize           int
	eps                            float32
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv2D    `gguf:"patch_embedding"`
	PositionEmbedding *nn.Embedding `gguf:"position_embedding"`
	PostLayerNorm     *nn.LayerNorm `gguf:"post_layernorm"`

	Encoder *VisionEncoder `gguf:"blk"`

	*VisionModelOptions
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues, positionIDs ml.Tensor) ml.Tensor {
	numPatches := (m.imageSize / m.patchSize) * (m.imageSize / m.patchSize)

	hiddenState := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, numPatches, m.hiddenSize)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	positions := m.PositionEmbedding.Forward(ctx, positionIDs)
	hiddenState = hiddenState.Add(ctx, positions)

	for _, layer := range m.Encoder.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, m.VisionModelOptions)
	}

	hiddenState = m.PostLayerNorm.Forward(ctx, hiddenState, m.eps)
	return hiddenState
}

func newVisionModel(c ml.Config) *VisionModel {
	return &VisionModel{
		Encoder: &VisionEncoder{Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count"))},
		VisionModelOptions: &VisionModelOptions{
			hiddenSize: int(c.Uint("vision.embedding_length")),
			numHeads:   int(c.Uint("vision.attention.head_count")),

			imageSize: int(c.Uint("vision.image_size")),
			patchSize: int(c.Uint("vision.patch_size")),

			eps: c.Float("vision.attention.layer_norm_epsilon"),
		},
	}
}
