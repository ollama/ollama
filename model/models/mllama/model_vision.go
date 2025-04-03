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
	Output *nn.Linear `gguf:"attn_out"`

	Gate ml.Tensor `gguf:"attn_gate"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	query = query.Reshape(ctx, batchSize, query.Dim(1), opts.numHeads, headDim)
	query = query.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	key := sa.Key.Forward(ctx, hiddenState)
	key = key.Reshape(ctx, batchSize, key.Dim(1), opts.numHeads, headDim)
	key = key.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	value := sa.Value.Forward(ctx, hiddenState)
	value = value.Reshape(ctx, batchSize, value.Dim(1), opts.numHeads, headDim)
	value = value.Permute(ctx, 0, 2, 3, 1).Contiguous(ctx)

	scores := query.Matmul(ctx, key)
	scores = scores.Scale(ctx, 1.0/math.Sqrt(float64(headDim)))
	scores = scores.Softmax(ctx)

	attention := scores.Matmul(ctx, value)
	attention = attention.Reshape(ctx, batchSize, opts.numHeads, attention.Dim(2), headDim)
	attention = attention.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	attention = attention.Reshape(ctx, batchSize, attention.Dim(1), opts.hiddenSize)

	hiddenState = sa.Output.Forward(ctx, attention)
	if sa.Gate != nil {
		hiddenState = hiddenState.Mul(ctx, sa.Gate)
	}

	return hiddenState
}

type VisionMLP struct {
	Down *nn.Linear `gguf:"ffn_down"`
	Up   *nn.Linear `gguf:"ffn_up"`

	Gate ml.Tensor `gguf:"ffn_gate"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	hiddenState = mlp.Down.Forward(ctx, hiddenState).GELU(ctx)
	hiddenState = mlp.Up.Forward(ctx, hiddenState)
	if mlp.Gate != nil {
		hiddenState = hiddenState.Mul(ctx, mlp.Gate)
	}

	return hiddenState
}

type VisionEncoderLayer struct {
	AttentionNorm *nn.LayerNorm `gguf:"ln1"`
	SelfAttention *VisionSelfAttention

	MLPNorm *nn.LayerNorm `gguf:"ln2"`
	MLP     *VisionMLP
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenState

	// self attention
	hiddenState = e.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.SelfAttention.Forward(ctx, hiddenState, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	// feed forward
	hiddenState = e.MLPNorm.Forward(ctx, hiddenState, opts.eps)
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
			intermediateHiddenStates = append(intermediateHiddenStates, hiddenState.Reshape(ctx, append(hiddenState.Shape(), 1)...))
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
	embeddings = embeddings.Reshape(ctx, opts.numTiles, 1, opts.hiddenSize)
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

func (e *PrecomputedPositionEmbedding) Forward(ctx ml.Context, hiddenState, positionIDs, aspectRatioIDs ml.Tensor, numPositions int, opts *VisionModelOptions) ml.Tensor {
	positionEmbedding := e.PositionEmbedding.Forward(ctx, positionIDs)
	if e.PositionEmbeddingGate != nil {
		positionEmbedding = positionEmbedding.Mul(ctx, e.PositionEmbeddingGate)
	}

	hiddenState = hiddenState.Add(ctx, positionEmbedding)

	tilePositionEmbedding := e.TilePositionEmbedding.Forward(ctx, aspectRatioIDs)
	tilePositionEmbedding = tilePositionEmbedding.Reshape(ctx, opts.numTiles, numPositions, opts.hiddenSize)
	if e.TilePositionEmbeddingGate != nil {
		tilePositionEmbedding = tilePositionEmbedding.Mul(ctx, e.TilePositionEmbeddingGate)
	}

	return hiddenState.Add(ctx, tilePositionEmbedding)
}

type VisionModelOptions struct {
	hiddenSize, numHeads, numTiles int
	imageSize, patchSize           int
	eps                            float32

	intermediateLayersIndices []uint32
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

	hiddenState := m.PatchEmbeddings.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, m.numTiles, m.hiddenSize, numPatches)
	hiddenState = hiddenState.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	hiddenState = m.PreTilePositionEmbedding.Forward(ctx, hiddenState, aspectRatioIDs, m.VisionModelOptions)
	hiddenState = m.ClassEmbedding.Stack(ctx, 2, slices.Repeat([]ml.Tensor{m.ClassEmbedding}, m.numTiles-1)...).Concat(ctx, hiddenState, 1)

	hiddenState = m.PositionEmbedding.Forward(ctx, hiddenState, positionIDs, aspectRatioIDs, numPositions, m.VisionModelOptions)
	hiddenState = m.PreLayerNorm.Forward(ctx, hiddenState, m.eps)

	numPaddingPatches := 8 - (hiddenState.Dim(1)%8)%8
	hiddenState = hiddenState.Pad(ctx, 0, numPaddingPatches, 0, 0)

	hiddenState = hiddenState.Reshape(ctx, batchSize, hiddenState.Dim(1)*hiddenState.Dim(0), hiddenState.Dim(2))
	hiddenState, intermediateHiddenStates := m.Transformer.Forward(ctx, hiddenState, m.intermediateLayersIndices, m.VisionModelOptions)

	hiddenState = m.PostLayerNorm.Forward(ctx, hiddenState, m.eps)

	hiddenState = hiddenState.Reshape(ctx, batchSize, m.numTiles, numPositions+numPaddingPatches, m.hiddenSize)
	hiddenState = m.PostTilePositionEmbedding.Forward(ctx, hiddenState, aspectRatioIDs, m.VisionModelOptions)

	hiddenState = hiddenState.Reshape(ctx, batchSize, m.numTiles*(numPositions+numPaddingPatches), m.hiddenSize)
	hiddenState, _ = m.GlobalTransformer.Forward(ctx, hiddenState, nil, m.VisionModelOptions)

	hiddenStates := intermediateHiddenStates[0].Stack(ctx, 0, intermediateHiddenStates[1:]...)
	hiddenStates = hiddenStates.Reshape(ctx, batchSize, m.numTiles, numPositions+numPaddingPatches, len(intermediateHiddenStates)*m.hiddenSize)
	hiddenStates = hiddenStates.Unpad(ctx, 0, 0, numPaddingPatches, 0)

	hiddenState = hiddenState.Reshape(ctx, batchSize, m.numTiles, numPositions+numPaddingPatches, m.hiddenSize)
	hiddenState = hiddenState.Unpad(ctx, 0, 0, numPaddingPatches, 0)
	return hiddenState.Concat(ctx, hiddenStates, 0)
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Transformer:       &VisionEncoder{Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count"))},
		GlobalTransformer: &VisionEncoder{Layers: make([]VisionEncoderLayer, c.Uint("vision.global.block_count"))},

		VisionModelOptions: &VisionModelOptions{
			hiddenSize: int(c.Uint("vision.embedding_length")),
			numHeads:   int(c.Uint("vision.attention.head_count")),
			numTiles:   int(c.Uint("vision.max_num_tiles")),

			imageSize: int(c.Uint("vision.image_size")),
			patchSize: int(c.Uint("vision.patch_size")),

			eps: c.Float("vision.attention.layer_norm_epsilon"),

			intermediateLayersIndices: c.Uints("vision.intermediate_layers_indices"),
		},
	}
}
