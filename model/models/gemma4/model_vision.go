package gemma4

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
)

const batchSize = 1

// ClippableLinear is a linear layer with optional input/output clamping.
// Required by Gemma4 vision encoder for numerical stability with F16 weights.
type ClippableLinear struct {
	Weight ml.Tensor `gguf:"weight"`

	InputMin  ml.Tensor `gguf:"input_min"`
	InputMax  ml.Tensor `gguf:"input_max"`
	OutputMin ml.Tensor `gguf:"output_min"`
	OutputMax ml.Tensor `gguf:"output_max"`

	inMin, inMax, outMin, outMax float32
	hasClamp                     bool
	clampsLoaded                 bool
}

func scalarValue(t ml.Tensor) (float32, bool) {
	if t == nil {
		return 0, false
	}

	data := t.BackendGet()
	if len(data) == 0 {
		return 0, false
	}

	return data[0], true
}

func (l *ClippableLinear) loadClampFromScalars() {
	if l.clampsLoaded {
		return
	}
	l.clampsLoaded = true

	const (
		defaultMin = -math.MaxFloat32
		defaultMax = math.MaxFloat32
	)

	inMin, hasInMin := scalarValue(l.InputMin)
	inMax, hasInMax := scalarValue(l.InputMax)
	outMin, hasOutMin := scalarValue(l.OutputMin)
	outMax, hasOutMax := scalarValue(l.OutputMax)

	if !(hasInMin || hasInMax || hasOutMin || hasOutMax) {
		return
	}

	l.hasClamp = true
	l.inMin = defaultMin
	l.inMax = defaultMax
	l.outMin = defaultMin
	l.outMax = defaultMax

	if hasInMin {
		l.inMin = inMin
	}
	if hasInMax {
		l.inMax = inMax
	}
	if hasOutMin {
		l.outMin = outMin
	}
	if hasOutMax {
		l.outMax = outMax
	}
}

func (l *ClippableLinear) Forward(ctx ml.Context, x ml.Tensor) ml.Tensor {
	if l.hasClamp {
		x = x.Clamp(ctx, l.inMin, l.inMax)
	}
	out := l.Weight.Mulmat(ctx, x)
	if l.hasClamp {
		out = out.Clamp(ctx, l.outMin, l.outMax)
	}
	return out
}

// InitClamp distributes packed clamp values from v.clamp_data to ClippableLinear structs.
// If scalar clamp tensors (input_min/max, output_min/max) are present, they are used too.
// Layout: numLayers × 7 linears (q,k,v,out,gate,up,down) × 4 floats (inMin,inMax,outMin,outMax)
// then 4 floats for the projector.
func (m *VisionModel) InitClamp(proj *MultiModalProjector) {
	if m.clampInitDone {
		return
	}
	m.clampInitDone = true

	linears := func(l *VisionEncoderLayer) []*ClippableLinear {
		return []*ClippableLinear{
			l.SelfAttention.Query, l.SelfAttention.Key, l.SelfAttention.Value,
			l.SelfAttention.Output, l.MLP.Gate, l.MLP.Up, l.MLP.Down,
		}
	}

	for i := range m.Layers {
		for _, cl := range linears(&m.Layers[i]) {
			if cl != nil {
				cl.loadClampFromScalars()
			}
		}
	}
	if proj != nil && proj.Projection != nil {
		proj.Projection.loadClampFromScalars()
	}

	// Load packed clamp data when present (legacy Ollama format).
	if m.ClampData == nil {
		return
	}

	// Read all clamp values from packed F32 tensor
	data := m.ClampData.BackendGet()
	if len(data) == 0 {
		return
	}

	// Distribute to layer linears: 7 per layer × 4 values each
	for i := range m.Layers {
		for li, cl := range linears(&m.Layers[i]) {
			if cl == nil {
				continue
			}
			idx := (i*7 + li) * 4
			if idx+3 < len(data) {
				cl.inMin = data[idx]
				cl.inMax = data[idx+1]
				cl.outMin = data[idx+2]
				cl.outMax = data[idx+3]
				cl.hasClamp = true
			}
		}
	}

	// Projector clamp values (last 4 floats)
	if proj != nil && proj.Projection != nil {
		projIdx := len(m.Layers) * 7 * 4
		if projIdx+3 < len(data) {
			proj.Projection.inMin = data[projIdx]
			proj.Projection.inMax = data[projIdx+1]
			proj.Projection.outMin = data[projIdx+2]
			proj.Projection.outMax = data[projIdx+3]
			proj.Projection.hasClamp = true
		}
	}
}

type VisionSelfAttention struct {
	Query     *ClippableLinear `gguf:"attn_q"`
	Key       *ClippableLinear `gguf:"attn_k"`
	Value     *ClippableLinear `gguf:"attn_v"`
	QueryNorm *nn.RMSNorm      `gguf:"attn_q_norm"`
	KeyNorm   *nn.RMSNorm      `gguf:"attn_k_norm"`
	Output    *ClippableLinear `gguf:"attn_out"`
}

func (sa *VisionSelfAttention) Forward(ctx ml.Context, hiddenState, posX, posY, attnMask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	numPatches := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	query := sa.Query.Forward(ctx, hiddenState)
	key := sa.Key.Forward(ctx, hiddenState)
	value := sa.Value.Forward(ctx, hiddenState)

	query = query.Reshape(ctx, headDim, opts.numHeads, numPatches, batchSize)
	key = key.Reshape(ctx, headDim, opts.numHeads, numPatches, batchSize)
	value = value.Reshape(ctx, headDim, opts.numHeads, numPatches, batchSize)

	// Q/K norms (Gemma-style: x * (1 + weight) / rms(x))
	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	// V norm (RMSNorm without learned weights)
	value = value.RMSNorm(ctx, nil, opts.eps)

	// 2D RoPE: split head dim in half, apply NeoX RoPE with x positions to first half,
	// y positions to second half, then concatenate.
	halfDim := headDim / 2
	ropeOpts := rope.WithTypeNeoX()

	qFirst := query.View(ctx, 0, halfDim, query.Stride(1), opts.numHeads, query.Stride(2), numPatches)
	qFirst = nn.RoPE(ctx, qFirst, posX, halfDim, opts.ropeTheta, 1.0, ropeOpts)

	kFirst := key.View(ctx, 0, halfDim, key.Stride(1), opts.numHeads, key.Stride(2), numPatches)
	kFirst = nn.RoPE(ctx, kFirst, posX, halfDim, opts.ropeTheta, 1.0, ropeOpts)

	halfOffset := halfDim * query.Stride(0)
	qSecond := query.View(ctx, halfOffset, halfDim, query.Stride(1), opts.numHeads, query.Stride(2), numPatches)
	qSecond = nn.RoPE(ctx, qSecond, posY, halfDim, opts.ropeTheta, 1.0, ropeOpts)

	halfOffsetK := halfDim * key.Stride(0)
	kSecond := key.View(ctx, halfOffsetK, halfDim, key.Stride(1), opts.numHeads, key.Stride(2), numPatches)
	kSecond = nn.RoPE(ctx, kSecond, posY, halfDim, opts.ropeTheta, 1.0, ropeOpts)

	query = qFirst.Concat(ctx, qSecond, 0)
	key = kFirst.Concat(ctx, kSecond, 0)

	// Use flash attention for numerical stability (handles large attention scores
	// from unclamped RMSNorm weights, e.g. 26B has addOne weights up to 19.5)
	attention := nn.Attention(ctx, query, key, value, 1.0, nil)
	attention = attention.Reshape(ctx, opts.hiddenSize, attention.Dim(2), batchSize)

	return sa.Output.Forward(ctx, attention)
}

type VisionMLP struct {
	Gate *ClippableLinear `gguf:"ffn_gate"`
	Up   *ClippableLinear `gguf:"ffn_up"`
	Down *ClippableLinear `gguf:"ffn_down"`
}

func (mlp *VisionMLP) Forward(ctx ml.Context, hiddenState ml.Tensor) ml.Tensor {
	gate := mlp.Gate.Forward(ctx, hiddenState)
	up := mlp.Up.Forward(ctx, hiddenState)
	hiddenState = gate.QuickGELU(ctx, up)
	return mlp.Down.Forward(ctx, hiddenState)
}

type VisionEncoderLayer struct {
	AttentionNorm     *nn.RMSNorm `gguf:"ln1"`
	SelfAttention     *VisionSelfAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"attn_post_norm"`

	FFNNorm     *nn.RMSNorm `gguf:"ln2"`
	MLP         *VisionMLP
	PostFFNNorm *nn.RMSNorm `gguf:"ffn_post_norm"`

	LayerOutputScale ml.Tensor `gguf:"out_scale.weight"`
}

func (e *VisionEncoderLayer) Forward(ctx ml.Context, hiddenState, posX, posY, attnMask ml.Tensor, opts *VisionModelOptions) ml.Tensor {
	residual := hiddenState

	// Pre-attention norm -> self attention -> post-attention norm
	hiddenState = e.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.SelfAttention.Forward(ctx, hiddenState, posX, posY, attnMask, opts)
	hiddenState = e.PostAttentionNorm.Forward(ctx, hiddenState, opts.eps)

	// Residual connection
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	// Pre-FFN norm -> FFN -> post-FFN norm
	hiddenState = e.FFNNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = e.MLP.Forward(ctx, hiddenState)
	hiddenState = e.PostFFNNorm.Forward(ctx, hiddenState, opts.eps)

	// Residual connection
	hiddenState = hiddenState.Add(ctx, residual)

	// Per-layer output scale
	if e.LayerOutputScale != nil {
		hiddenState = hiddenState.Mul(ctx, e.LayerOutputScale)
	}

	return hiddenState
}

type VisionModelOptions struct {
	hiddenSize int
	numHeads   int
	patchSize  int
	nMerge     int
	eps        float32
	ropeTheta  float32
}

type VisionModel struct {
	PatchEmbedding    *nn.Conv2D `gguf:"patch_embd"`
	PositionEmbedding ml.Tensor  `gguf:"position_embd.weight"`
	ClampData         ml.Tensor  `gguf:"clamp_data"`
	StdBias           ml.Tensor  `gguf:"std_bias"`
	StdScale          ml.Tensor  `gguf:"std_scale"`

	Layers []VisionEncoderLayer `gguf:"blk"`

	*VisionModelOptions
	clampInitDone bool
}

func (m *VisionModel) Forward(ctx ml.Context, pixelValues ml.Tensor, numPatchesX, numPatchesY int) ml.Tensor {
	numPatches := numPatchesX * numPatchesY

	// Patch embedding via Conv2D
	hiddenState := m.PatchEmbedding.Forward(ctx, pixelValues, m.patchSize, m.patchSize, 0, 0, 1, 1)
	hiddenState = hiddenState.Reshape(ctx, numPatches, m.hiddenSize)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	// Conv2D with F16 weights produces F16 output via im2col; cast to F32 for encoder precision
	hiddenState = hiddenState.Cast(ctx, ml.DTypeF32)

	// 2D positional embeddings from 3D tensor [nEmbd, maxPos, 2]
	posSize := m.PositionEmbedding.Dim(1)
	nb1 := m.PositionEmbedding.Stride(1)
	tblX := m.PositionEmbedding.View(ctx, 0, m.hiddenSize, nb1, posSize)
	tblY := m.PositionEmbedding.View(ctx, posSize*nb1, m.hiddenSize, nb1, posSize)

	// Position indices for patches
	posXData := make([]int32, numPatches)
	posYData := make([]int32, numPatches)
	for i := range numPatches {
		posXData[i] = int32(i % numPatchesX)
		posYData[i] = int32(i / numPatchesX)
	}

	posXEmb := ctx.Input().FromInts(posXData, numPatches)
	posYEmb := ctx.Input().FromInts(posYData, numPatches)

	hiddenState = hiddenState.Add(ctx, tblX.Rows(ctx, posXEmb))
	hiddenState = hiddenState.Add(ctx, tblY.Rows(ctx, posYEmb))

	// No attention mask — all positions are real patches
	var attnMask ml.Tensor

	// RoPE positions
	posXRope := ctx.Input().FromInts(posXData, numPatches)
	posYRope := ctx.Input().FromInts(posYData, numPatches)

	// Vision transformer layers
	for i := range m.Layers {
		hiddenState = m.Layers[i].Forward(ctx, hiddenState, posXRope, posYRope, attnMask, m.VisionModelOptions)
	}

	return hiddenState
}

func newVisionModel(c fs.Config) *VisionModel {
	return &VisionModel{
		Layers: make([]VisionEncoderLayer, c.Uint("vision.block_count")),
		VisionModelOptions: &VisionModelOptions{
			hiddenSize: int(c.Uint("vision.embedding_length")),
			numHeads:   int(c.Uint("vision.attention.head_count")),
			patchSize:  int(c.Uint("vision.patch_size", 16)),
			nMerge:     int(c.Uint("vision.projector.scale_factor", 3)),
			eps:        c.Float("vision.attention.layer_norm_epsilon", 1e-6),
			ropeTheta:  100.0,
		},
	}
}

func visionPoolAndProject(ctx ml.Context, hiddenState ml.Tensor, numPatchesX, numPatchesY int, opts *VisionModelOptions, proj *MultiModalProjector, stdBias, stdScale ml.Tensor) ml.Tensor {
	hiddenSize := opts.hiddenSize

	// Reshape from [hiddenSize, numPatches] to spatial layout for pooling
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	hiddenState = hiddenState.Reshape(ctx, numPatchesX, numPatchesY, hiddenSize)

	// AvgPool2D with kernel=stride=nMerge
	hiddenState = hiddenState.AvgPool2D(ctx, opts.nMerge, opts.nMerge, 0)

	// Reshape back to [hiddenSize, numMergedPatches]
	mergedX := numPatchesX / opts.nMerge
	mergedY := numPatchesY / opts.nMerge
	hiddenState = hiddenState.Reshape(ctx, mergedX*mergedY, hiddenSize)
	hiddenState = hiddenState.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)

	hiddenState = hiddenState.Cast(ctx, ml.DTypeF32)
	hiddenState = hiddenState.Scale(ctx, math.Sqrt(float64(hiddenSize)))

	// Optional vision standardization before projection.
	if stdBias != nil && stdScale != nil {
		hiddenState = hiddenState.Sub(ctx, stdBias)
		hiddenState = hiddenState.Mul(ctx, stdScale)
	}

	// Project to text embedding dimension
	hiddenState = proj.Forward(ctx, hiddenState, opts.eps)

	return hiddenState
}
