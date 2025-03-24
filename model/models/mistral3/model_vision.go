package mistral3

import (
	"image"
	"math"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

type PatchMerger struct {
	MergingLayer *nn.Linear `gguf:"merging_layer"`
}

func (pm *PatchMerger) Forward(ctx ml.Context, visionOutputs ml.Tensor, size image.Point, spatialMergeSize int) ml.Tensor {
	d := visionOutputs.Dim(0)
	imageGrid := visionOutputs.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx).Reshape(ctx, size.X, size.Y, d)
	kernel := ctx.Input().Empty(ml.DTypeF32, spatialMergeSize, spatialMergeSize, d)
	patches := kernel.IM2Col(ctx, imageGrid, spatialMergeSize, spatialMergeSize, 0, 0, 1, 1)
	reshaped := patches.Reshape(ctx, d*spatialMergeSize*spatialMergeSize, patches.Dim(1)*patches.Dim(2))
	return pm.MergingLayer.Forward(ctx, reshaped)
}

type MultiModalProjector struct {
	Norm        *nn.RMSNorm  `gguf:"norm"`
	Linear1     *nn.Linear   `gguf:"linear_1"`
	Linear2     *nn.Linear   `gguf:"linear_2"`
	PatchMerger *PatchMerger `gguf:"patch_merger"`

	spatialMergeSize int
	eps              float32
	patchSize        int
}

func (p *MultiModalProjector) Forward(ctx ml.Context, visionOutputs ml.Tensor, size image.Point) (ml.Tensor, image.Point) {
	visionOutputs = p.Norm.Forward(ctx, visionOutputs, p.eps)
	patchSizes := image.Point{size.X / p.patchSize, size.Y / p.patchSize}
	visionOutputs = p.PatchMerger.Forward(ctx, visionOutputs, patchSizes, p.spatialMergeSize)
	visionOutputs = p.Linear1.Forward(ctx, visionOutputs)
	visionOutputs = visionOutputs.GELU(ctx)
	return p.Linear2.Forward(ctx, visionOutputs), image.Point{patchSizes.X / p.spatialMergeSize, patchSizes.Y / p.spatialMergeSize}
}

func newMultiModalProjector(c ml.Config) *MultiModalProjector {
	return &MultiModalProjector{
		spatialMergeSize: int(c.Uint("spatial_merge_size", 2)),
		eps:              c.Float("text_config.rms_norm_eps", 1e-5),
		patchSize:        int(c.Uint("vision.patch_size", 14)),
	}
}

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	q := sa.Query.Forward(ctx, hiddenState)
	k := sa.Key.Forward(ctx, hiddenState)
	v := sa.Value.Forward(ctx, hiddenState)

	q = q.Reshape(ctx, opts.headDim, opts.numHeads, q.Dim(1), batchSize)
	k = k.Reshape(ctx, opts.headDim, opts.numHeads, k.Dim(1), batchSize)
	v = v.Reshape(ctx, opts.headDim, opts.numHeads, v.Dim(1), batchSize)

	ropeType := uint32(24) // 2d vision rope
	q = q.RoPEMulti(ctx, positionIDs, nil, uint32(opts.headDim/2), [4]int{0, opts.headDim / 2, opts.headDim / 2, 0}, ropeType, opts.ropeBase, opts.ropeScale)
	k = k.RoPEMulti(ctx, positionIDs, nil, uint32(opts.headDim/2), [4]int{0, opts.headDim / 2, opts.headDim / 2, 0}, ropeType, opts.ropeBase, opts.ropeScale)

	attention := nn.Attention(ctx, q, k, v, 1.0/math.Sqrt(float64(opts.headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).GELU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type VisionEncoderLayer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *VisionSelfAttention
	FFNNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *VisionMLP
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenState
	hiddenState = e.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.SelfAttention.Forward(ctx, hiddenState, positionIDs, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = e.FFNNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
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
	ropeScale        float32
}

type VisionModel struct {
	PatchEmbedding *nn.Conv2D           `gguf:"patch_conv"`
	EncoderNorm    *nn.RMSNorm          `gguf:"encoder_norm"`
	Layers         []VisionEncoderLayer `gguf:"blk"`

	*VisionModelOptions
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor) ml.Tensor {
	numPatchesW := pixelValues.Dim(0) / m.patchSize
	numPatchesH := pixelValues.Dim(1) / m.patchSize
	numPatches := numPatchesW * numPatchesH
	hiddenState := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, numPatches, m.hiddenSize)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	hiddenState = m.EncoderNorm.Forward(ctx, hiddenState, m.VisionModelOptions.eps)

	// Prepare position IDs for 2D rope
	positions := make([]int32, numPatches*4)
	for h := 0; h < numPatchesH; h++ {
		for w := 0; w < numPatchesW; w++ {
			idx := h*numPatchesW + w
			positions[idx] = 0                     // time (unused)
			positions[numPatches+idx] = int32(h)   // height
			positions[numPatches*2+idx] = int32(w) // width
			positions[numPatches*3+idx] = 0        // extra (unused)
		}
	}

	positionIDs, err := ctx.Input().FromIntSlice(positions, len(positions))
	if err != nil {
		panic(err)
	}

	for _, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, positionIDs, m.VisionModelOptions)
	}

	return hiddenState
}

func newVisionModel(c ml.Config) *VisionModel {
	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 24)),
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
			ropeScale:        c.Float("vision.rope.freq_scale", 1.0),
		},
	}
}
