package nemotronh

import (
	"math"
	"sync"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

const nemotronVisionBatchSize = 1

type visionPatchGrid struct {
	Width  int
	Height int
}

type VisionPatchEmbedding struct {
	*nn.Linear
}

func packVisionPatchesCHW(values []float32, width, height, channels, patchSize int) []float32 {
	patchesX, patchesY := width/patchSize, height/patchSize
	patchDim := channels * patchSize * patchSize
	plane := width * height

	patches := make([]float32, patchDim*patchesX*patchesY)
	offset := 0
	for py := range patchesY {
		for px := range patchesX {
			for c := range channels {
				channelBase := c * plane
				for yy := range patchSize {
					rowBase := (py*patchSize + yy) * width
					for xx := range patchSize {
						patches[offset] = values[channelBase+rowBase+px*patchSize+xx]
						offset++
					}
				}
			}
		}
	}

	return patches
}

func (p *VisionPatchEmbedding) ForwardPacked(ctx ml.Context, patches []float32, patchDim, numPatches int) ml.Tensor {
	hiddenState := ctx.Input().FromFloats(patches, patchDim, numPatches)
	hiddenState = hiddenState.Duplicate(ctx)
	return p.Linear.Forward(ctx, hiddenState)
}

func (p *VisionPatchEmbedding) Forward(ctx ml.Context, pixelValues ml.Tensor, patchSize int) ml.Tensor {
	// Match the RADIO patch generator's exact flattening order: patches are laid
	// out token-major with each token packed as channel, then patch-row, then
	// patch-col. This is more explicit than the prior IM2Col path and likely
	// slower, but it avoids backend-specific packing differences that caused the
	// converted patch embedder to diverge badly from the reference model.
	width, height, channels := pixelValues.Dim(0), pixelValues.Dim(1), pixelValues.Dim(2)
	patchesX, patchesY := width/patchSize, height/patchSize
	patchDim := channels * patchSize * patchSize

	values := pixelValues.BackendGet()
	return p.ForwardPacked(ctx, packVisionPatchesCHW(values, width, height, channels, patchSize), patchDim, patchesX*patchesY)
}

type VisionSelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_out"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionOptions) ml.Tensor {
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	key := sa.Key.Forward(ctx, hiddenState)
	value := sa.Value.Forward(ctx, hiddenState)

	query = query.Reshape(ctx, headDim, opts.numHeads, query.Dim(1), nemotronVisionBatchSize)
	key = key.Reshape(ctx, headDim, opts.numHeads, key.Dim(1), nemotronVisionBatchSize)
	value = value.Reshape(ctx, headDim, opts.numHeads, value.Dim(1), nemotronVisionBatchSize)

	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(headDim)), nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), nemotronVisionBatchSize)
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

func (l *VisionEncoderLayer) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *VisionOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = l.LayerNorm1.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, opts)
	hiddenState = hiddenState.Add(ctx, residual)

	residual = hiddenState
	hiddenState = l.LayerNorm2.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState)
	return hiddenState.Add(ctx, residual)
}

type VisionOptions struct {
	hiddenSize int
	numHeads   int
	imageSize  int
	patchSize  int
	eps        float32
}

type VisionModel struct {
	PatchEmbedding    *VisionPatchEmbedding `gguf:"patch_embd"`
	PositionEmbedding ml.Tensor             `gguf:"position_embd"`
	ClassEmbedding    ml.Tensor             `gguf:"cls_embd"`

	Layers []VisionEncoderLayer `gguf:"blk"`

	*VisionOptions

	resizedPositionEmbeddingsMu sync.Mutex
	resizedPositionEmbeddings   map[visionPatchGrid][]float32
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, patches visionPatchGrid) ml.Tensor {
	numPatches := patches.Width * patches.Height

	hiddenState := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize)
	return m.forwardPatchEmbeddings(ctx, hiddenState, patches, numPatches)
}

func (m *VisionModel) ForwardPacked(ctx ml.Context, patchValues []float32, patches visionPatchGrid) ml.Tensor {
	numPatches := patches.Width * patches.Height
	patchDim := 0
	if numPatches > 0 {
		patchDim = len(patchValues) / numPatches
	}
	hiddenState := m.PatchEmbedding.ForwardPacked(ctx, patchValues, patchDim, numPatches)
	return m.forwardPatchEmbeddings(ctx, hiddenState, patches, numPatches)
}

func (m *VisionModel) forwardPatchEmbeddings(ctx ml.Context, hiddenState ml.Tensor, patches visionPatchGrid, numPatches int) ml.Tensor {
	if m.PositionEmbedding != nil {
		positionEmbeddings := m.positionEmbeddings(ctx, hiddenState, patches, numPatches)
		hiddenState = hiddenState.Add(ctx, positionEmbeddings)
	}

	if m.ClassEmbedding != nil {
		numPrefixTokens := m.ClassEmbedding.Dim(1)
		classEmbeddings := m.ClassEmbedding.Cast(ctx, hiddenState.DType())
		classEmbeddings = classEmbeddings.Reshape(ctx, classEmbeddings.Dim(0), numPrefixTokens, 1)
		hiddenState = classEmbeddings.Concat(ctx, hiddenState, 1)
	}

	for _, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, m.VisionOptions)
	}

	if m.ClassEmbedding != nil {
		hiddenState = hiddenState.Slice(ctx, 1, m.ClassEmbedding.Dim(1), hiddenState.Dim(1), 1)
	}

	return hiddenState.Reshape(ctx, hiddenState.Dim(0), hiddenState.Dim(1))
}

func (m *VisionModel) positionEmbeddings(ctx ml.Context, hiddenState ml.Tensor, patches visionPatchGrid, numPatches int) ml.Tensor {
	posTokens := m.PositionEmbedding.Dim(1)
	source := int(math.Sqrt(float64(posTokens)))

	positionEmbeddings := m.PositionEmbedding.Cast(ctx, hiddenState.DType())
	if !(source > 0 && source*source == posTokens && (source != patches.Width || source != patches.Height)) {
		if positionEmbeddings.Dim(1) > numPatches {
			positionEmbeddings = positionEmbeddings.Slice(ctx, 1, 0, numPatches, 1)
		}
		return positionEmbeddings
	}

	if cached, ok := m.cachePositionEmbeddings(ctx, hiddenState.Dim(0), patches); ok {
		return ctx.Input().FromFloats(cached, hiddenState.Dim(0), numPatches)
	}

	// Runner fit/reserve builds worst-case multimodal graphs before weights are
	// loaded, so the align-corners CPU cache path cannot materialize source
	// values there. Fall back to a graph-only bilinear resize for reservation;
	// the loaded inference path above still uses the cached align-corners data.
	positionEmbeddings = positionEmbeddings.Reshape(ctx, -1, source, source)
	positionEmbeddings = positionEmbeddings.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	positionEmbeddings = positionEmbeddings.Interpolate(ctx, [4]int{
		patches.Width,
		patches.Height,
		hiddenState.Dim(0),
		1,
	}, ml.SamplingModeBilinear)
	positionEmbeddings = positionEmbeddings.Permute(ctx, 1, 2, 0, 3)
	return positionEmbeddings.Contiguous(ctx, -1, patches.Width*patches.Height)
}

func (m *VisionModel) cachePositionEmbeddings(ctx ml.Context, hidden int, patches visionPatchGrid) ([]float32, bool) {
	m.resizedPositionEmbeddingsMu.Lock()
	cached := m.resizedPositionEmbeddings[patches]
	m.resizedPositionEmbeddingsMu.Unlock()
	if cached != nil {
		return cached, true
	}

	if len(m.PositionEmbedding.Bytes()) == 0 {
		return nil, false
	}

	posTokens := m.PositionEmbedding.Dim(1)
	source := int(math.Sqrt(float64(posTokens)))
	positionEmbeddingsF32 := m.PositionEmbedding.Cast(ctx, ml.DTypeF32)
	ctx.Forward(positionEmbeddingsF32).Compute(positionEmbeddingsF32)

	// RADIO eval-time CPE uses bilinear interpolation with align_corners=false.
	// Cache a CPU-resized token-major embedding here for correctness first. This
	// is likely slower than a native graph path and should be revisited if this
	// precision vs speed tradeoff is not worthwhile.
	cached = resizePositionEmbedding(positionEmbeddingsF32.Floats(), hidden, source, source, patches.Width, patches.Height)

	m.resizedPositionEmbeddingsMu.Lock()
	if m.resizedPositionEmbeddings == nil {
		m.resizedPositionEmbeddings = make(map[visionPatchGrid][]float32)
	}
	if existing := m.resizedPositionEmbeddings[patches]; existing != nil {
		cached = existing
	} else {
		m.resizedPositionEmbeddings[patches] = cached
	}
	m.resizedPositionEmbeddingsMu.Unlock()

	return cached, true
}

func resizePositionEmbedding(values []float32, hidden, sourceWidth, sourceHeight, targetWidth, targetHeight int) []float32 {
	out := make([]float32, hidden*targetWidth*targetHeight)

	scaleX := float64(sourceWidth) / float64(targetWidth)
	scaleY := float64(sourceHeight) / float64(targetHeight)

	for oy := range targetHeight {
		srcY := scaleY*(float64(oy)+0.5) - 0.5
		y0 := int(math.Floor(srcY))
		y1 := min(y0+1, sourceHeight-1)
		wy := float32(srcY - float64(y0))
		y0 = max(y0, 0)

		for ox := range targetWidth {
			srcX := scaleX*(float64(ox)+0.5) - 0.5
			x0 := int(math.Floor(srcX))
			x1 := min(x0+1, sourceWidth-1)
			wx := float32(srcX - float64(x0))
			x0 = max(x0, 0)

			t00 := (y0*sourceWidth + x0) * hidden
			t01 := (y0*sourceWidth + x1) * hidden
			t10 := (y1*sourceWidth + x0) * hidden
			t11 := (y1*sourceWidth + x1) * hidden
			dst := (oy*targetWidth + ox) * hidden
			for h := range hidden {
				v00 := values[t00+h]
				v01 := values[t01+h]
				v10 := values[t10+h]
				v11 := values[t11+h]
				top := v00 + (v01-v00)*wx
				bot := v10 + (v11-v10)*wx
				out[dst+h] = top + (bot-top)*wy
			}
		}
	}

	return out
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count", 32)),
		VisionOptions: &VisionOptions{
			hiddenSize: int(c.Uint("vision.embedding_length", 1280)),
			numHeads:   int(c.Uint("vision.attention.head_count", 16)),
			imageSize:  int(c.Uint("vision.image_size", 512)),
			patchSize:  int(c.Uint("vision.patch_size", 16)),
			eps:        c.Float("vision.attention.layer_norm_epsilon", 1e-6),
		},
	}
}

type MultiModalProjector struct {
	Norm    *nn.RMSNorm `gguf:"norm"`
	Linear1 *nn.Linear  `gguf:"1"`
	Linear2 *nn.Linear  `gguf:"2"`

	scaleFactor int
}

func (p *MultiModalProjector) Forward(ctx ml.Context, visionOutputs ml.Tensor, patches visionPatchGrid) ml.Tensor {
	scaleFactor := max(p.scaleFactor, 1)

	// The reference projector first pixel-shuffles the vision grid with
	// downsample_ratio=0.5 before applying the RMSNorm/MLP. Preserve that exact
	// v2 packing order here rather than flattening 2x2 neighborhoods via IM2Col.
	merged := pixelShuffleVisionOutputs(ctx, visionOutputs, patches, scaleFactor)

	merged = p.Norm.Forward(ctx, merged, 1e-5)
	merged = p.Linear1.Forward(ctx, merged)
	merged = merged.RELU(ctx)
	merged = merged.Mul(ctx, merged)
	return p.Linear2.Forward(ctx, merged)
}

func pixelShuffleVisionOutputs(ctx ml.Context, visionOutputs ml.Tensor, patches visionPatchGrid, scaleFactor int) ml.Tensor {
	hiddenSize := visionOutputs.Dim(0)
	scaleFactor = max(scaleFactor, 1)

	merged := visionOutputs.Reshape(ctx, hiddenSize, patches.Width, patches.Height, 1)

	width := patches.Width / scaleFactor
	height := patches.Height / scaleFactor
	channels := hiddenSize * scaleFactor

	merged = merged.Reshape(ctx, channels, width, patches.Height, 1)
	merged = merged.Reshape(ctx, channels, width, scaleFactor, height)
	merged = merged.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	return merged.Reshape(ctx, channels*scaleFactor, width*height, 1)
}

func newMultiModalProjector(c fs.Config) *MultiModalProjector {
	return &MultiModalProjector{
		scaleFactor: int(c.Uint("vision.projector.scale_factor", 2)),
	}
}
