package mistral3

import (
	"image"
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

var batchSize int = 1

type PatchMerger struct {
	MergingLayer *nn.Linear `gguf:"merging_layer"`
}

func rotateHalf(ctx ml.Context, t ml.Tensor) ml.Tensor {
	x1 := t.View(ctx, 0, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3))
	x2 := t.View(ctx, t.Stride(0)*t.Dim(0)/2, t.Dim(0)/2, t.Stride(1), t.Dim(1), t.Stride(2), t.Dim(2), t.Stride(3), t.Dim(3))
	return x2.Neg(ctx).Concat(ctx, x1, 0)
}

func applyRotaryPositionalEmbedding(ctx ml.Context, query, key, cos, sin ml.Tensor) (ml.Tensor, ml.Tensor) {
	query = query.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, query).Mul(ctx, sin))
	key = key.Mul(ctx, cos).Add(ctx, rotateHalf(ctx, key).Mul(ctx, sin))
	return query, key
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

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenStates, cos, sin ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, opts.headDim, opts.numHeads, query.Dim(1), batchSize)
	key = key.Reshape(ctx, opts.headDim, opts.numHeads, key.Dim(1), batchSize)
	value = value.Reshape(ctx, opts.headDim, opts.numHeads, value.Dim(1), batchSize)

	query = query.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	key = key.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	query, key = applyRotaryPositionalEmbedding(ctx, query, key, cos, sin)

	scores := key.Mulmat(ctx, query)
	scores = scores.Scale(ctx, 1.0/math.Sqrt(float64(opts.headDim)))
	scores = scores.Softmax(ctx)

	attention := value.Mulmat(ctx, scores).Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).GELU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenStates))
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

	h, err := ctx.Input().FromFloatSlice(frequenciesHeight, maxPatchesPerSide, frequencies/2)
	if err != nil {
		panic(err)
	}

	w, err := ctx.Input().FromFloatSlice(frequenciesWidth, maxPatchesPerSide, frequencies/2)
	if err != nil {
		panic(err)
	}

	h = h.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	w = w.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	h = h.Stack(ctx, 1, slices.Repeat([]ml.Tensor{h}, maxPatchesPerSide-1)...)
	h = h.Reshape(ctx, frequencies/2, maxPatchesPerSide, maxPatchesPerSide).Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	w = w.Stack(ctx, 2, slices.Repeat([]ml.Tensor{w}, maxPatchesPerSide-1)...)

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

	positionIDs, err := ctx.Input().FromIntSlice(positions, len(positions))
	if err != nil {
		panic(err)
	}

	positionEmbedding := m.positionalEmbedding(ctx, positionIDs)
	cos, sin := positionEmbedding.Cos(ctx), positionEmbedding.Sin(ctx)

	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(ctx, hiddenStates, cos, sin, m.VisionModelOptions)
	}

	return hiddenStates
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
		},
	}
}
