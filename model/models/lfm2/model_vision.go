package lfm2

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

const lfm2VisionBatchSize = 1

type visionPatchGrid struct {
	Width  int
	Height int
}

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output,alt:attn_out"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	key := sa.Key.Forward(ctx, hiddenState)
	value := sa.Value.Forward(ctx, hiddenState)

	query = query.Reshape(ctx, headDim, opts.numHeads, query.Dim(1), lfm2VisionBatchSize)
	key = key.Reshape(ctx, headDim, opts.numHeads, key.Dim(1), lfm2VisionBatchSize)
	value = value.Reshape(ctx, headDim, opts.numHeads, value.Dim(1), lfm2VisionBatchSize)

	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), lfm2VisionBatchSize)
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	return mlp.Down.Forward(ctx, mlp.Up.Forward(ctx, hiddenState).GELU(ctx))
}

type VisionEncoderLayer struct {
	LayerNorm1    *nn.LayerNorm `gguf:"ln1"`
	SelfAttention *VisionSelfAttention

	LayerNorm2 *nn.LayerNorm `gguf:"ln2"`
	MLP        *VisionMLP
}

func (l *VisionEncoderLayer) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = l.LayerNorm1.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, opts)
	hiddenState = hiddenState.Add(ctx, residual)

	residual = hiddenState
	hiddenState = l.LayerNorm2.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState)
	return hiddenState.Add(ctx, residual)
}

type VisionModelOptions struct {
	hiddenSize, numHeads int
	imageSize, patchSize int
	eps                  float32
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv2D    `gguf:"patch_embd"`
	PositionEmbedding *nn.Embedding `gguf:"position_embd"`
	PostLayerNorm     *nn.LayerNorm `gguf:"post_ln"`

	Layers []VisionEncoderLayer `gguf:"blk"`

	*VisionModelOptions
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, patches visionPatchGrid) ml.Tensor {
	numPatches := patches.Width * patches.Height

	hiddenState := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, numPatches, m.hiddenSize)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	if m.PositionEmbedding != nil {
		posTokens := m.PositionEmbedding.Weight.Dim(1)
		source := int(math.Sqrt(float64(posTokens)))

		var positionEmbeddings ml.Tensor
		if source > 0 && source*source == posTokens && (source != patches.Width || source != patches.Height) {
			// SigLIP2 NAFlex-style position interpolation for variable image sizes.
			positionIDs := ctx.Arange(0, float32(posTokens), 1, ml.DTypeI32)
			positionEmbeddings = m.PositionEmbedding.Forward(ctx, positionIDs)
			positionEmbeddings = positionEmbeddings.Reshape(ctx, -1, source, source)
			positionEmbeddings = positionEmbeddings.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
			positionEmbeddings = positionEmbeddings.Interpolate(ctx, [4]int{
				patches.Width,
				patches.Height,
				hiddenState.Dim(0),
				1,
			}, ml.SamplingModeBilinear)
			positionEmbeddings = positionEmbeddings.Permute(ctx, 1, 2, 0, 3)
			positionEmbeddings = positionEmbeddings.Contiguous(ctx, -1, patches.Width*patches.Height)
		} else {
			positionIDs := ctx.Arange(0, float32(numPatches), 1, ml.DTypeI32)
			positionEmbeddings = m.PositionEmbedding.Forward(ctx, positionIDs)
		}

		hiddenState = hiddenState.Add(ctx, positionEmbeddings)
	}

	for _, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, m.VisionModelOptions)
	}

	return m.PostLayerNorm.Forward(ctx, hiddenState, m.eps)
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count")),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize: int(c.Uint("vision.embedding_length", 1152)),
			numHeads:   int(c.Uint("vision.attention.head_count", 16)),
			imageSize:  int(c.Uint("vision.image_size", 256)),
			patchSize:  int(c.Uint("vision.patch_size", 16)),
			eps:        c.Float("vision.attention.layer_norm_epsilon", 1e-6),
		},
	}
}

type VisionProjector struct {
	LayerNorm *nn.LayerNorm `gguf:"layer_norm"`
	Linear1   *nn.Linear    `gguf:"1"`
	Linear2   *nn.Linear    `gguf:"2"`
}

type VisionProjectorOptions struct {
	scaleFactor  int
	useLayerNorm bool
}

func (p *VisionProjector) Forward(ctx ml.Context, visionOutputs ml.Tensor, patches visionPatchGrid, opts VisionProjectorOptions) ml.Tensor {
	hiddenSize := visionOutputs.Dim(0)
	featureMap := visionOutputs

	merge := max(opts.scaleFactor, 1)
	if merge > 1 {
		width := patches.Width
		height := patches.Height

		featureMap = featureMap.Reshape(ctx, hiddenSize, width, height)

		// Match llama.cpp patch merger: pad spatial dims to merge factor.
		padWidth := (merge - width%merge) % merge
		padHeight := (merge - height%merge) % merge
		if padWidth != 0 || padHeight != 0 {
			featureMap = featureMap.Pad(ctx, 0, padWidth, padHeight, 0)
			width += padWidth
			height += padHeight
		}

		featureMap = featureMap.Reshape(ctx, hiddenSize*merge, width/merge, height)
		featureMap = featureMap.Permute(ctx, 0, 2, 1).Contiguous(ctx, hiddenSize*merge*merge, height/merge, width/merge)
		featureMap = featureMap.Permute(ctx, 0, 2, 1).Contiguous(ctx)
		featureMap = featureMap.Contiguous(ctx, featureMap.Dim(0), featureMap.Dim(1)*featureMap.Dim(2))
	}

	if opts.useLayerNorm && p.LayerNorm != nil {
		featureMap = p.LayerNorm.Forward(ctx, featureMap, 1e-5)
	}

	featureMap = p.Linear1.Forward(ctx, featureMap).GELU(ctx)
	return p.Linear2.Forward(ctx, featureMap)
}
