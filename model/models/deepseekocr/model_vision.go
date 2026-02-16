package deepseekocr

import (
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type visionModel struct {
	PatchEmbedding    *nn.Conv2D    `gguf:"patch_embd"`
	ClassEmbedding    ml.Tensor     `gguf:"class_embd"`
	PositionEmbedding *nn.Embedding `gguf:"position_embd"`

	PreLayerNorm *nn.LayerNorm `gguf:"pre_layrnorm"`
	Blocks       []visionBlock `gguf:"blk"`

	Options visionOptions
}

func (m *visionModel) absolutePositionEmbedding(ctx ml.Context, embeds ml.Tensor) ml.Tensor {
	numPatches := m.Options.imageSize / m.Options.patchSize * m.Options.imageSize / m.Options.patchSize
	positions := ctx.Arange(0, float32(numPatches+1), 1, ml.DTypeI32)
	positionEmbeds := m.PositionEmbedding.Forward(ctx, positions)

	source := int(math.Sqrt(float64(positionEmbeds.Dim(1) - 1)))
	target := int(math.Sqrt(float64(embeds.Dim(1) - 1)))
	if source != target {
		newPositionEmbeds := positionEmbeds.Slice(ctx, 1, 1, positionEmbeds.Dim(1), 1)
		newPositionEmbeds = newPositionEmbeds.Reshape(ctx, -1, source, source)
		newPositionEmbeds = newPositionEmbeds.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
		newPositionEmbeds = newPositionEmbeds.Interpolate(ctx, [4]int{target, target, embeds.Dim(0), 1}, ml.SamplingModeBilinear)
		newPositionEmbeds = newPositionEmbeds.Permute(ctx, 1, 2, 0, 3)
		newPositionEmbeds = newPositionEmbeds.Contiguous(ctx, -1, target*target)

		positionEmbeds = positionEmbeds.Slice(ctx, 1, 0, 1, 1).Concat(ctx, newPositionEmbeds, 1)
	}

	return positionEmbeds
}

func (m *visionModel) Forward(ctx ml.Context, pixelValues, patchEmbeds ml.Tensor) ml.Tensor {
	if patchEmbeds == nil {
		patchEmbeds = m.PatchEmbedding.Forward(ctx, pixelValues, m.Options.patchSize, m.Options.patchSize, 0, 0, 1, 1)
	}

	patchEmbeds = patchEmbeds.Reshape(ctx, -1, patchEmbeds.Dim(2), patchEmbeds.Dim(3))
	patchEmbeds = patchEmbeds.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	classEmbeds := m.ClassEmbedding.Repeat(ctx, 2, patchEmbeds.Dim(2))
	embeds := classEmbeds.Concat(ctx, patchEmbeds, 1)
	embeds = embeds.Add(ctx, m.absolutePositionEmbedding(ctx, embeds))

	hiddenStates := m.PreLayerNorm.Forward(ctx, embeds, m.Options.eps)
	for _, block := range m.Blocks {
		hiddenStates = block.Forward(ctx, hiddenStates, m.Options)
	}

	return hiddenStates
}

type visionOptions struct {
	hiddenSize,
	numHeads int
	eps float32

	imageSize, patchSize int
}

func (o visionOptions) headDim() int {
	return o.hiddenSize / o.numHeads
}

type visionBlock struct {
	Norm1       *nn.LayerNorm    `gguf:"layer_norm1"`
	Attention   *visionAttention `gguf:"self_attn"`
	Norm2       *nn.LayerNorm    `gguf:"layer_norm2"`
	FeedForward *visionMLP       `gguf:"mlp"`
}

func (m *visionBlock) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts visionOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = m.Norm1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.Attention.Forward(ctx, hiddenStates, opts)
	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = m.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.FeedForward.Forward(ctx, hiddenStates)
	hiddenStates = hiddenStates.Add(ctx, residual)
	return hiddenStates
}

type visionAttention struct {
	QKV    *nn.Linear `gguf:"qkv_proj"`
	Output *nn.Linear `gguf:"out_proj"`
}

func (m *visionAttention) Forward(ctx ml.Context, t ml.Tensor, opts visionOptions) ml.Tensor {
	qkv := m.QKV.Forward(ctx, t)
	qkv = qkv.Reshape(ctx, opts.headDim(), -1, qkv.Dim(1), qkv.Dim(2))
	chunks := qkv.Chunk(ctx, 1, opts.numHeads)
	query, key, value := chunks[0], chunks[1], chunks[2]

	attention := nn.Attention(ctx, query, key, value, 1/math.Sqrt(float64(opts.headDim())), nil)
	attention = attention.Reshape(ctx, -1, attention.Dim(2), attention.Dim(3))
	return m.Output.Forward(ctx, attention)
}

type visionMLP struct {
	FC1 *nn.Linear `gguf:"fc1"`
	FC2 *nn.Linear `gguf:"fc2"`
}

func (m *visionMLP) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	return m.FC2.Forward(ctx, m.FC1.Forward(ctx, t).QuickGELU(ctx))
}
