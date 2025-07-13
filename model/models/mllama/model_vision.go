package mllama

import (
	"math"
	"slices"

	"github.com/ollama/ollama/fs"
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
	query = query.Reshape(ctx, headDim, opts.numHeads, query.Dim(1), batchSize)

	key := sa.Key.Forward(ctx, hiddenState)
	key = key.Reshape(ctx, headDim, opts.numHeads, key.Dim(1), batchSize)

	value := sa.Value.Forward(ctx, hiddenState)
	value = value.Reshape(ctx, headDim, opts.numHeads, value.Dim(1), batchSize)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	hiddenState = mlp.Up.Forward(ctx, hiddenState).GELU(ctx)
	hiddenState = mlp.Down.Forward(ctx, hiddenState)

	return hiddenState
}

type VisionEncoderLayer struct {
	AttentionNorm *nn.LayerNorm `gguf:"attn_norm"`
	SelfAttention *VisionSelfAttention
	AttentionGate ml.Tensor `gguf:"attn_gate"`

	MLPNorm *nn.LayerNorm `gguf:"ffn_norm"`
	MLP     *VisionMLP
	MLPGate ml.Tensor `gguf:"ffn_gate"`
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenState

	// self attention
	hiddenState = e.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.SelfAttention.Forward(ctx, hiddenState, opts)
	if e.AttentionGate != nil {
		hiddenState = hiddenState.Mul(ctx, e.AttentionGate)
	}
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = e.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.MLP.Forward(ctx, hiddenState, opts)
	if e.MLPGate != nil {
		hiddenState = hiddenState.Mul(ctx, e.MLPGate)
	}
	hiddenState = hiddenState.Add(ctx, residual)
	return hiddenState
}

type VisionEncoder struct {
	Layers []VisionEncoderLayer
}

func (e *VisionEncoder) Forward(ctx ml.Context, hiddenState ml.Tensor, intermediateLayersIndices []int32, opts *VisionModelOptions) (ml.Tensor, []ml.Tensor) {
	var intermediateHiddenStates []ml.Tensor
	for i, layer := range e.Layers {
		if slices.Contains(intermediateLayersIndices, int32(i)) {
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

func (e *PrecomputedAspectRatioEmbedding) Forward(ctx ml.Context, hiddenState ml.Tensor, aspectRatioIDs ml.Tensor, numTiles int, opts *VisionModelOptions) ml.Tensor {
	embeddings := e.Embedding.Forward(ctx, aspectRatioIDs)
	embeddings = embeddings.Reshape(ctx, opts.hiddenSize, 1, numTiles)
	if e.Gate != nil {
		embeddings = embeddings.Mul(ctx, e.Gate)
	}

	return hiddenState.Add(ctx, embeddings)
}

type PrecomputedPositionEmbedding struct {
	PositionEmbedding     *nn.Embedding `gguf:"position_embd"`
	PositionEmbeddingGate ml.Tensor     `gguf:"position_embd.gate"`

	TilePositionEmbedding     *nn.Embedding `gguf:"tile_position_embd"`
	TilePositionEmbeddingGate ml.Tensor     `gguf:"tile_position_embd.gate"`
}

func (e *PrecomputedPositionEmbedding) Forward(ctx ml.Context, hiddenState, positionIDs, aspectRatioIDs ml.Tensor, numPositions, numTiles int, opts *VisionModelOptions) ml.Tensor {
	positionEmbedding := e.PositionEmbedding.Forward(ctx, positionIDs)
	if e.PositionEmbeddingGate != nil {
		positionEmbedding = positionEmbedding.Mul(ctx, e.PositionEmbeddingGate)
	}

	hiddenState = hiddenState.Add(ctx, positionEmbedding)

	tilePositionEmbedding := e.TilePositionEmbedding.Forward(ctx, aspectRatioIDs)
	tilePositionEmbedding = tilePositionEmbedding.Reshape(ctx, opts.hiddenSize, numPositions, numTiles)
	if e.TilePositionEmbeddingGate != nil {
		tilePositionEmbedding = tilePositionEmbedding.Mul(ctx, e.TilePositionEmbeddingGate)
	}

	return hiddenState.Add(ctx, tilePositionEmbedding)
}

type VisionModelOptions struct {
	hiddenSize, numHeads int
	imageSize, patchSize int
	eps                  float32

	intermediateLayersIndices []int32
}

type VisionModel struct {
	PatchEmbeddings *nn.Conv2D `gguf:"patch_embd"`

	PreTilePositionEmbedding  *PrecomputedAspectRatioEmbedding `gguf:"pre_tile_position_embd"`
	PostTilePositionEmbedding *PrecomputedAspectRatioEmbedding `gguf:"post_tile_position_embd"`
	PositionEmbedding         *PrecomputedPositionEmbedding

	PreLayerNorm   *nn.LayerNorm `gguf:"pre_ln"`
	PostLayerNorm  *nn.LayerNorm `gguf:"post_ln"`
	ClassEmbedding ml.Tensor     `gguf:"class_embd"`

	Transformer       *VisionEncoder `gguf:"blk"`
	GlobalTransformer *VisionEncoder `gguf:"global.blk"`

	*VisionModelOptions
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues, positionIDs, aspectRatioIDs ml.Tensor) ml.Tensor {
	numPatches := (m.imageSize / m.patchSize) * (m.imageSize / m.patchSize)
	numPositions := numPatches
	if m.ClassEmbedding != nil {
		numPositions++
	}

	numTiles := pixelValues.Dim(3)

	hiddenState := m.PatchEmbeddings.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, numPatches, m.hiddenSize, numTiles)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	hiddenState = m.PreTilePositionEmbedding.Forward(ctx, hiddenState, aspectRatioIDs, numTiles, m.VisionModelOptions)
	hiddenState = m.ClassEmbedding.Repeat(ctx, 2, numTiles).Concat(ctx, hiddenState, 1)

	hiddenState = m.PositionEmbedding.Forward(ctx, hiddenState, positionIDs, aspectRatioIDs, numPositions, numTiles, m.VisionModelOptions)
	hiddenState = m.PreLayerNorm.Forward(ctx, hiddenState, m.eps)

	numPaddingPatches := 8 - (hiddenState.Dim(1)%8)%8
	hiddenState = hiddenState.Pad(ctx, 0, numPaddingPatches, 0, 0)

	hiddenState = hiddenState.Reshape(ctx, hiddenState.Dim(0), hiddenState.Dim(1)*hiddenState.Dim(2), batchSize)
	hiddenState, intermediateHiddenStates := m.Transformer.Forward(ctx, hiddenState, m.intermediateLayersIndices, m.VisionModelOptions)

	hiddenState = m.PostLayerNorm.Forward(ctx, hiddenState, m.eps)

	hiddenState = hiddenState.Reshape(ctx, m.hiddenSize, numPositions+numPaddingPatches, numTiles, batchSize)
	hiddenState = m.PostTilePositionEmbedding.Forward(ctx, hiddenState, aspectRatioIDs, numTiles, m.VisionModelOptions)

	hiddenState = hiddenState.Reshape(ctx, m.hiddenSize, numTiles*(numPositions+numPaddingPatches), batchSize)
	hiddenState, _ = m.GlobalTransformer.Forward(ctx, hiddenState, nil, m.VisionModelOptions)

	hiddenStates := intermediateHiddenStates[0].Stack(ctx, 0, intermediateHiddenStates[1:]...)
	hiddenStates = hiddenStates.Reshape(ctx, len(intermediateHiddenStates)*m.hiddenSize, numPositions+numPaddingPatches, numTiles, batchSize)
	hiddenStates = hiddenStates.Pad(ctx, 0, -numPaddingPatches, 0, 0)

	hiddenState = hiddenState.Reshape(ctx, m.hiddenSize, numPositions+numPaddingPatches, numTiles, batchSize)
	hiddenState = hiddenState.Pad(ctx, 0, -numPaddingPatches, 0, 0)
	return hiddenState.Concat(ctx, hiddenStates, 0)
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Transformer:       &VisionEncoder{Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count"))},
		GlobalTransformer: &VisionEncoder{Layers: make([]VisionEncoderLayer, c.Uint("vision.global.block_count"))},

		VisionModelOptions: &VisionModelOptions{
			hiddenSize: int(c.Uint("vision.embedding_length")),
			numHeads:   int(c.Uint("vision.attention.head_count")),

			imageSize: int(c.Uint("vision.image_size")),
			patchSize: int(c.Uint("vision.patch_size")),

			eps: c.Float("vision.attention.layer_norm_epsilon"),

			intermediateLayersIndices: c.Ints("vision.intermediate_layers_indices"),
		},
	}
}
