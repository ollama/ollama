package deepseekocr

import (
	"math"
	"slices"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type samModel struct {
	PatchEmbedding    *nn.Conv2D `gguf:"patch_embd"`
	PositionEmbedding ml.Tensor  `gguf:"position_embd"`

	Blocks []samBlock `gguf:"blk"`

	Neck *samNeck   `gguf:"neck"`
	Net2 *nn.Conv2D `gguf:"net_2"`
	Net3 *nn.Conv2D `gguf:"net_3"`

	Options samOptions
}

func (m *samModel) absolutePositionEmbedding(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	source := m.PositionEmbedding.Dim(1)
	target := hiddenStates.Dim(2)
	if source != target {
		positionEmbed := m.PositionEmbedding.Permute(ctx, 2, 0, 1, 3)
		positionEmbed = positionEmbed.Interpolate(ctx, [4]int{target, target, hiddenStates.Dim(0), 1}, ml.SamplingModeBilinear)
		return positionEmbed.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
	}

	return m.PositionEmbedding
}

func (m *samModel) Forward(ctx ml.Context, t ml.Tensor) ml.Tensor {
	hiddenStates := m.PatchEmbedding.Forward(ctx, t, 16, 16, 0, 0, 1, 1)
	hiddenStates = hiddenStates.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	if m.PositionEmbedding != nil {
		hiddenStates = hiddenStates.Add(ctx, m.absolutePositionEmbedding(ctx, hiddenStates))
	}

	for i, block := range m.Blocks {
		var windowSize int
		if !slices.Contains(m.Options.globalAttentionLayers, int32(i)) {
			windowSize = 14
		}

		hiddenStates = block.Forward(ctx, hiddenStates, windowSize, m.Options)
	}

	hiddenStates = hiddenStates.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	hiddenStates = m.Neck.Forward(ctx, hiddenStates, m.Options)
	hiddenStates = m.Net2.Forward(ctx, hiddenStates, 2, 2, 1, 1, 1, 1)
	hiddenStates = m.Net3.Forward(ctx, hiddenStates, 2, 2, 1, 1, 1, 1)
	return hiddenStates
}

type samOptions struct {
	hiddenSize,
	numHeads int
	eps                   float32
	globalAttentionLayers []int32
}

func (o samOptions) headDim() int {
	return o.hiddenSize / o.numHeads
}

type samBlock struct {
	Norm1       *nn.LayerNorm `gguf:"norm1"`
	Attention   *samAttention `gguf:"attn"`
	Norm2       *nn.LayerNorm `gguf:"norm2"`
	FeedForward *samMLP       `gguf:"mlp"`
}

func (m *samBlock) Forward(ctx ml.Context, hiddenStates ml.Tensor, windowSize int, opts samOptions) ml.Tensor {
	c, w, h := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)

	residual := hiddenStates
	hiddenStates = m.Norm1.Forward(ctx, hiddenStates, opts.eps)

	var pw, ph int
	if windowSize > 0 {
		pw = (windowSize - hiddenStates.Dim(1)%windowSize) % windowSize
		ph = (windowSize - hiddenStates.Dim(2)%windowSize) % windowSize
		if pw > 0 || ph > 0 {
			hiddenStates = hiddenStates.Pad(ctx, 0, pw, ph, 0)
		}

		hiddenStates = hiddenStates.Reshape(ctx, c*windowSize, (w+pw)/windowSize, windowSize, -1)
		hiddenStates = hiddenStates.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, c, windowSize, windowSize, -1)
	}

	hiddenStates = m.Attention.Forward(ctx, hiddenStates, opts)

	if windowSize > 0 {
		hiddenStates = hiddenStates.Reshape(ctx, c*windowSize, windowSize, (w+pw)/windowSize, -1)
		hiddenStates = hiddenStates.Permute(ctx, 0, 2, 1, 3)
		hiddenStates = hiddenStates.Contiguous(ctx, c, w+pw, h+ph, -1)
		hiddenStates = hiddenStates.Pad(ctx, 0, -pw, -ph, 0)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)

	residual = hiddenStates
	hiddenStates = m.Norm2.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.FeedForward.Forward(ctx, hiddenStates, opts)
	return hiddenStates.Add(ctx, residual)
}

type samAttention struct {
	QKV    *nn.Linear `gguf:"qkv"`
	Output *nn.Linear `gguf:"proj"`

	RelativePosition *struct {
		Height ml.Tensor `gguf:"h"`
		Width  ml.Tensor `gguf:"w"`
	} `gguf:",pre:rel_pos_"`
}

func relativeCoordinates(ctx ml.Context, qn, kn int) ml.Tensor {
	s := make([]int32, qn*kn)
	for i := range qn {
		for j := range kn {
			q := i * max(kn/qn, 1)
			k := j * max(qn/kn, 1)
			s[i*kn+j] = int32(q - k + (kn-1)*max(qn/kn, 1))
		}
	}
	return ctx.Input().FromInts(s, qn*kn)
}

func relativePositions(ctx ml.Context, positions ml.Tensor, qn, kn int) ml.Tensor {
	maxRelativeDistance := 2*max(qn, kn) - 1
	if positions.Dim(1) != maxRelativeDistance {
		// linear interpolation kernel not available so approx. with bilinear interpolation
		positions = positions.Interpolate(ctx, [4]int{positions.Dim(0), maxRelativeDistance, 1, 1}, ml.SamplingModeBilinear)
	}

	rc := relativeCoordinates(ctx, qn, kn)
	return positions.Rows(ctx, rc).Reshape(ctx, positions.Dim(0), kn, qn)
}

func (m *samAttention) decomposedRelativePositions(ctx ml.Context, query ml.Tensor, qn, kn []int) (ml.Tensor, ml.Tensor) {
	qh, qw := qn[0], qn[1]
	kh, kw := kn[0], kn[1]

	rh := relativePositions(ctx, m.RelativePosition.Height, qh, kh)
	rw := relativePositions(ctx, m.RelativePosition.Width, qw, kw)

	query = query.Contiguous(ctx, query.Dim(0), qw, qh, -1)
	rh = rh.Mulmat(ctx, query).Reshape(ctx, 1, kh, qh*qw, -1)
	rw = rw.Mulmat(ctx, query.Permute(ctx, 0, 2, 1, 3)).Permute(ctx, 0, 2, 1, 3).Contiguous(ctx, kw, 1, qh*qw, -1)
	return rh, rw
}

func (m *samAttention) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts samOptions) ml.Tensor {
	w, h, b := hiddenStates.Dim(1), hiddenStates.Dim(2), hiddenStates.Dim(3)

	qkv := m.QKV.Forward(ctx, hiddenStates)
	qkv = qkv.Reshape(ctx, opts.headDim(), -1, w*h, b)
	chunks := qkv.Chunk(ctx, 1, opts.numHeads)
	query, key, value := chunks[0], chunks[1], chunks[2]

	ctx.Forward(query, key, value)

	query = query.Permute(ctx, 0, 2, 1, 3)
	rh, rw := m.decomposedRelativePositions(ctx, query, []int{h, w}, []int{h, w})
	mask := rh.Repeat(ctx, 0, rw.Dim(0)).Add(ctx, rw)
	mask = mask.Reshape(ctx, h*w, -1, opts.numHeads, b)

	key = key.Permute(ctx, 0, 2, 1, 3)
	scores := key.MulmatFullPrec(ctx, query)
	scores = scores.Scale(ctx, 1/math.Sqrt(float64(opts.headDim())))

	scores = scores.Add(ctx, mask)
	scores = scores.Softmax(ctx)

	value = value.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
	attention := value.Mulmat(ctx, scores)
	attention = attention.Permute(ctx, 0, 2, 1, 3)
	attention = attention.Contiguous(ctx, -1, w, h, b)
	return m.Output.Forward(ctx, attention)
}

type samMLP struct {
	Lin1 *nn.Linear `gguf:"lin1"`
	Lin2 *nn.Linear `gguf:"lin2"`
}

func (m *samMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts samOptions) ml.Tensor {
	return m.Lin2.Forward(ctx, m.Lin1.Forward(ctx, hiddenStates).GELU(ctx))
}

type LayerNorm2D struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (ln *LayerNorm2D) Forward(ctx ml.Context, x ml.Tensor, eps float32) ml.Tensor {
	x = x.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
	u := x.Mean(ctx)
	d := x.Sub(ctx, u)
	s := d.Sqr(ctx).Mean(ctx)
	x = d.Div(ctx, s.Add(ctx, ctx.Input().FromFloats([]float32{eps}, 1)).Sqrt(ctx))
	x = x.Mul(ctx, ln.Weight).Add(ctx, ln.Bias)
	return x.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
}

type samNeck struct {
	C1  *nn.Conv2D   `gguf:"0"`
	LN1 *LayerNorm2D `gguf:"1"`
	C2  *nn.Conv2D   `gguf:"2"`
	LN2 *LayerNorm2D `gguf:"3"`
}

func (m *samNeck) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts samOptions) ml.Tensor {
	hiddenStates = m.C1.Forward(ctx, hiddenStates, 1, 1, 0, 0, 1, 1)
	hiddenStates = m.LN1.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = m.C2.Forward(ctx, hiddenStates, 1, 1, 1, 1, 1, 1)
	hiddenStates = m.LN2.Forward(ctx, hiddenStates, opts.eps)
	return hiddenStates
}
