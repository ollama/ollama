package nemotronh

import (
	"math"
	"sync"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

type AudioOptions struct {
	hiddenSize        int
	numHeads          int
	headDim           int
	intermediateSize  int
	convKernelSize    int
	melBins           int
	sampleRate        int
	subsamplingKernel int
	subsamplingStride int
	scaleInput        bool
	eps               float32
}

type AudioFeatureExtractor struct {
	FB     ml.Tensor `gguf:"fb"`
	Window ml.Tensor `gguf:"window"`

	mu      sync.Mutex
	fb      []float32
	window  []float32
	fbShape [2]int
}

func (f *AudioFeatureExtractor) windowAndFilters(melBins, freqBins, sampleRate int) ([]float32, []float32) {
	if f == nil {
		return defaultParakeetWindow(), buildSlaneyMelFilterBank(freqBins, melBins, sampleRate)
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if f.window == nil {
		if f.Window != nil {
			if values := f.Window.BackendGet(); len(values) == parakeetWinLength {
				f.window = values
			}
		}
		if f.window == nil {
			f.window = defaultParakeetWindow()
		}
	}

	if f.fb == nil {
		if f.FB != nil {
			if values := f.FB.BackendGet(); len(values) == melBins*freqBins {
				f.fb = values
				f.fbShape = [2]int{melBins, freqBins}
			}
		}
		if f.fb == nil {
			f.fb = buildSlaneyMelFilterBank(freqBins, melBins, sampleRate)
			f.fbShape = [2]int{melBins, freqBins}
		}
	}

	return f.window, f.fb
}

type AudioSubsampling struct {
	Conv0 *nn.Conv2D            `gguf:"conv0"`
	DW1   *AudioDepthwiseConv2D `gguf:"dw1"`
	PW1   *nn.Conv2D            `gguf:"pw1"`
	DW2   *AudioDepthwiseConv2D `gguf:"dw2"`
	PW2   *nn.Conv2D            `gguf:"pw2"`

	Linear *nn.Linear `gguf:"linear"`
}

type AudioDepthwiseConv2D struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

type AudioFeedForward struct {
	Up   *nn.Linear `gguf:"up"`
	Down *nn.Linear `gguf:"down"`
}

type AudioSelfAttention struct {
	Query       *nn.Linear `gguf:"attn_q"`
	Key         *nn.Linear `gguf:"attn_k"`
	Value       *nn.Linear `gguf:"attn_v"`
	Output      *nn.Linear `gguf:"attn_out"`
	RelativeKey *nn.Linear `gguf:"attn_rel_k"`
	BiasU       ml.Tensor  `gguf:"attn_bias_u"`
	BiasV       ml.Tensor  `gguf:"attn_bias_v"`
}

type AudioConvolutionModule struct {
	Pointwise1 *nn.Linear        `gguf:"conv_pw1"`
	Depthwise  ml.Tensor         `gguf:"conv_dw.weight"`
	BatchNorm  *AudioBatchNorm1D `gguf:"conv_bn"`
	Pointwise2 *nn.Linear        `gguf:"conv_pw2"`
}

type AudioBatchNorm1D struct {
	Weight      ml.Tensor `gguf:"weight"`
	Bias        ml.Tensor `gguf:"bias"`
	RunningMean ml.Tensor `gguf:"running_mean"`
	RunningVar  ml.Tensor `gguf:"running_var"`
}

type AudioLayer struct {
	FFN1Norm *nn.LayerNorm `gguf:"ffn1_norm"`
	FFN1Up   *nn.Linear    `gguf:"ffn1_up"`
	FFN1Down *nn.Linear    `gguf:"ffn1_down"`

	AttentionNorm *nn.LayerNorm `gguf:"attn_norm"`
	Attention     *AudioSelfAttention

	ConvNorm *nn.LayerNorm `gguf:"conv_norm"`
	Conv     *AudioConvolutionModule

	FFN2Norm *nn.LayerNorm `gguf:"ffn2_norm"`
	FFN2Up   *nn.Linear    `gguf:"ffn2_up"`
	FFN2Down *nn.Linear    `gguf:"ffn2_down"`

	OutputNorm *nn.LayerNorm `gguf:"out_norm"`
}

type AudioModel struct {
	FeatureExtractor *AudioFeatureExtractor `gguf:"feature_extractor"`
	Subsampling      *AudioSubsampling      `gguf:"subsampling"`
	Layers           []AudioLayer           `gguf:"blk"`

	*AudioOptions
}

type AudioProjector struct {
	Norm    *nn.RMSNorm `gguf:"norm"`
	Linear1 *nn.Linear  `gguf:"1"`
	Linear2 *nn.Linear  `gguf:"2"`
}

func (p *AudioProjector) Forward(ctx ml.Context, x ml.Tensor, eps float32) ml.Tensor {
	x = p.Norm.Forward(ctx, x, eps)
	x = audioF32(ctx, p.Linear1.Forward(ctx, x))
	x = x.RELU(ctx)
	x = x.Mul(ctx, x)
	return audioF32(ctx, p.Linear2.Forward(ctx, x))
}

func (m *AudioModel) ForwardAudio(ctx ml.Context, melFeatures ml.Tensor, validFrames int, projector *AudioProjector) ml.Tensor {
	x := melFeatures.Reshape(ctx, melFeatures.Dim(0), melFeatures.Dim(1), 1, 1)
	validLen := validFrames

	x = forwardAudioConv2D(ctx, m.Subsampling.Conv0, x, m.subsamplingStride, m.subsamplingStride, audioConvPadding(m.subsamplingKernel), audioConvPadding(m.subsamplingKernel), 1, 1)
	x = x.RELU(ctx)
	validLen = convOutputLength(validLen, m.subsamplingKernel, m.subsamplingStride, audioConvPadding(m.subsamplingKernel))
	x = applyAudioTimeMask(ctx, x, validLen)

	x = forwardAudioDepthwiseConv2D(ctx, m.Subsampling.DW1, x, m.subsamplingStride, m.subsamplingStride, audioConvPadding(m.subsamplingKernel), audioConvPadding(m.subsamplingKernel), 1, 1)
	x = forwardAudioConv2D(ctx, m.Subsampling.PW1, x, 1, 1, 0, 0, 1, 1)
	x = x.RELU(ctx)
	validLen = convOutputLength(validLen, m.subsamplingKernel, m.subsamplingStride, audioConvPadding(m.subsamplingKernel))
	x = applyAudioTimeMask(ctx, x, validLen)

	x = forwardAudioDepthwiseConv2D(ctx, m.Subsampling.DW2, x, m.subsamplingStride, m.subsamplingStride, audioConvPadding(m.subsamplingKernel), audioConvPadding(m.subsamplingKernel), 1, 1)
	x = forwardAudioConv2D(ctx, m.Subsampling.PW2, x, 1, 1, 0, 0, 1, 1)
	x = x.RELU(ctx)
	validLen = convOutputLength(validLen, m.subsamplingKernel, m.subsamplingStride, audioConvPadding(m.subsamplingKernel))
	x = applyAudioTimeMask(ctx, x, validLen)

	x = flattenAudioSubsamplingOutput(ctx, x)
	x = m.Subsampling.Linear.Forward(ctx, x)

	if m.scaleInput {
		x = x.Scale(ctx, math.Sqrt(float64(m.hiddenSize)))
	}
	if validLen > 0 && validLen < x.Dim(1) {
		x = x.Slice(ctx, 1, 0, validLen, 1).Contiguous(ctx)
	}

	for i := range m.Layers {
		x = m.Layers[i].Forward(ctx, x, validLen, m.AudioOptions)
	}

	if projector != nil {
		x = projector.Forward(ctx, x, m.eps)
	}
	return x
}

func flattenAudioSubsamplingOutput(ctx ml.Context, x ml.Tensor) ml.Tensor {
	fOut, tOut, cOut := x.Dim(0), x.Dim(1), x.Dim(2)
	// PyTorch flattens the subsampling output after [B, C, T, F] ->
	// [B, T, C, F], so F must remain the fastest dimension inside each
	// channel block before the linear projection.
	x = x.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	return x.Reshape(ctx, cOut*fOut, tOut)
}

func (l *AudioLayer) Forward(ctx ml.Context, x ml.Tensor, validLen int, opts *AudioOptions) ml.Tensor {
	residual := x
	x = audioFeedForward(ctx, l.FFN1Up, l.FFN1Down, l.FFN1Norm.Forward(ctx, x, opts.eps)).Scale(ctx, 0.5)
	x = residual.Add(ctx, x)

	residual = x
	x = l.Attention.Forward(ctx, l.AttentionNorm.Forward(ctx, x, opts.eps), validLen, opts)
	x = residual.Add(ctx, x)

	residual = x
	x = l.Conv.Forward(ctx, l.ConvNorm.Forward(ctx, x, opts.eps), opts)
	x = residual.Add(ctx, x)

	residual = x
	x = audioFeedForward(ctx, l.FFN2Up, l.FFN2Down, l.FFN2Norm.Forward(ctx, x, opts.eps)).Scale(ctx, 0.5)
	x = residual.Add(ctx, x)

	return l.OutputNorm.Forward(ctx, x, opts.eps)
}

func audioFeedForward(ctx ml.Context, up, down *nn.Linear, x ml.Tensor) ml.Tensor {
	x = audioF32(ctx, up.Forward(ctx, x))
	x = x.SILU(ctx)
	return audioF32(ctx, down.Forward(ctx, x))
}

func (a *AudioSelfAttention) Forward(ctx ml.Context, x ml.Tensor, validLen int, opts *AudioOptions) ml.Tensor {
	seqLen := x.Dim(1)
	headDim := opts.headDim
	numHeads := opts.numHeads

	q := audioF32(ctx, a.Query.Forward(ctx, x)).Reshape(ctx, headDim, numHeads, seqLen)
	k := audioF32(ctx, a.Key.Forward(ctx, x)).Reshape(ctx, headDim, numHeads, seqLen)
	v := audioF32(ctx, a.Value.Forward(ctx, x)).Reshape(ctx, headDim, numHeads, seqLen)

	qU := q
	if a.BiasU != nil {
		qU = qU.Add(ctx, audioF32(ctx, a.BiasU).Reshape(ctx, headDim, numHeads, 1))
	}
	qV := q
	if a.BiasV != nil {
		qV = qV.Add(ctx, audioF32(ctx, a.BiasV).Reshape(ctx, headDim, numHeads, 1))
	}

	qP := qU.Permute(ctx, 0, 2, 1, 3)
	kP := k.Permute(ctx, 0, 2, 1, 3)
	logits := kP.MulmatFullPrec(ctx, qP)

	positionEmbeddings := parakeetPositionEmbeddings(ctx, seqLen, opts.hiddenSize)
	relKey := audioF32(ctx, a.RelativeKey.Forward(ctx, positionEmbeddings)).Reshape(ctx, headDim, numHeads, 2*seqLen-1)
	pP := relKey.Permute(ctx, 0, 2, 1, 3)
	qVP := qV.Permute(ctx, 0, 2, 1, 3)
	relLogits := pP.MulmatFullPrec(ctx, qVP)
	relLogits = relativeShiftParakeet(ctx, relLogits, seqLen, numHeads)

	logits = logits.Add(ctx, relLogits)
	logits = logits.Scale(ctx, math.Pow(float64(headDim), -0.5))
	if validLen > 0 && validLen < seqLen {
		logits = logits.Add(ctx, audioAttentionMask(ctx, seqLen, validLen))
	}
	logits = logits.Softmax(ctx)

	vP := v.Permute(ctx, 0, 2, 1, 3)
	vPT := vP.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	out := vPT.Mulmat(ctx, logits)
	out = out.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	out = out.Reshape(ctx, opts.hiddenSize, seqLen)
	return audioF32(ctx, a.Output.Forward(ctx, out))
}

func (c *AudioConvolutionModule) Forward(ctx ml.Context, x ml.Tensor, opts *AudioOptions) ml.Tensor {
	x = audioF32(ctx, c.Pointwise1.Forward(ctx, x))
	hidden := x.Dim(0) / 2
	value := x.Slice(ctx, 0, 0, hidden, 1).Contiguous(ctx)
	gate := x.Slice(ctx, 0, hidden, 2*hidden, 1).Contiguous(ctx).Sigmoid(ctx)
	x = value.Mul(ctx, gate)

	x = audioDepthwiseConv1DSame(ctx, x, c.Depthwise, audioConvPadding(opts.convKernelSize))
	x = c.BatchNorm.Forward(ctx, x, opts.eps)
	x = x.SILU(ctx)
	return audioF32(ctx, c.Pointwise2.Forward(ctx, x))
}

func audioF32(ctx ml.Context, x ml.Tensor) ml.Tensor {
	if x.DType() == ml.DTypeF32 {
		return x
	}

	// Metal binary kernels used by the audio graph require F32 operands here.
	// This likely slows audio and should be revisited once the precision vs.
	// speed tradeoff is validated against BF16-native elementwise paths.
	return x.Cast(ctx, ml.DTypeF32)
}

func (b *AudioBatchNorm1D) Forward(ctx ml.Context, x ml.Tensor, eps float32) ml.Tensor {
	if b == nil || b.RunningMean == nil || b.RunningVar == nil {
		return x
	}

	hidden := x.Dim(0)
	epsValues := make([]float32, hidden)
	for i := range epsValues {
		epsValues[i] = eps
	}

	variance := b.RunningVar.Add(ctx, ctx.Input().FromFloats(epsValues, hidden))
	x = x.Sub(ctx, b.RunningMean)
	x = x.Div(ctx, variance.Sqrt(ctx))
	if b.Weight != nil {
		x = x.Mul(ctx, b.Weight)
	}
	if b.Bias != nil {
		x = x.Add(ctx, b.Bias)
	}
	return x
}

func forwardAudioConv2D(ctx ml.Context, conv *nn.Conv2D, x ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	weight := conv.Weight.Contiguous(ctx)
	x = weight.Conv2D(ctx, x, s0, s1, p0, p1, d0, d1)
	if conv.Bias != nil {
		x = x.Add(ctx, conv.Bias.Reshape(ctx, 1, 1, -1))
	}
	return x
}

func forwardAudioDepthwiseConv2D(ctx ml.Context, conv *AudioDepthwiseConv2D, x ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	x = audioDepthwiseConv2D(ctx, x, conv.Weight, s0, s1, p0, p1, d0, d1)
	if conv.Bias != nil {
		x = x.Add(ctx, conv.Bias.Reshape(ctx, 1, 1, -1))
	}
	return x
}

func applyAudioTimeMask(ctx ml.Context, x ml.Tensor, validLen int) ml.Tensor {
	if validLen <= 0 || validLen >= x.Dim(1) {
		return x
	}
	mask := make([]float32, x.Dim(1))
	for i := range validLen {
		mask[i] = 1
	}
	return x.Mul(ctx, ctx.Input().FromFloats(mask, 1, x.Dim(1), 1, 1))
}

func audioDepthwiseConv1DSame(ctx ml.Context, x, kernel ml.Tensor, padding int) ml.Tensor {
	kernelSize := kernel.Dim(0)
	seqLen := x.Dim(1)

	kernelT := kernel.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx)
	var out ml.Tensor
	for k := range kernelSize {
		offset := k - padding
		shifted := x
		switch {
		case offset > 0:
			shifted = x.Slice(ctx, 1, offset, seqLen, 1).Contiguous(ctx)
			shifted = shifted.PadExt(ctx, 0, 0, 0, offset, 0, 0, 0, 0)
		case offset < 0:
			shift := -offset
			shifted = x.Slice(ctx, 1, 0, seqLen-shift, 1).Contiguous(ctx)
			shifted = shifted.PadExt(ctx, 0, 0, shift, 0, 0, 0, 0, 0)
		}

		wk := kernelT.Slice(ctx, 1, k, k+1, 1).Contiguous(ctx)
		term := shifted.Mul(ctx, wk)
		if out == nil {
			out = term
		} else {
			out = out.Add(ctx, term)
		}
	}
	return out
}

func audioDepthwiseConv2D(ctx ml.Context, x, kernel ml.Tensor, s0, s1, p0, p1, d0, d1 int) ml.Tensor {
	if d0 != 1 || d1 != 1 {
		panic("audio depthwise conv2d only supports dilation 1")
	}

	kernel = kernel.Contiguous(ctx)
	kernelW, kernelH := kernel.Dim(0), kernel.Dim(1)
	outW := convOutputLength(x.Dim(0), kernelW, s0, p0)
	outH := convOutputLength(x.Dim(1), kernelH, s1, p1)
	padded := x.PadExt(ctx, p0, p0, p1, p1, 0, 0, 0, 0)

	var out ml.Tensor
	for ky := range kernelH {
		for kx := range kernelW {
			patch := padded.Slice(ctx, 0, kx, kx+s0*(outW-1)+1, s0).Contiguous(ctx)
			patch = patch.Slice(ctx, 1, ky, ky+s1*(outH-1)+1, s1).Contiguous(ctx)

			wk := kernel.Slice(ctx, 0, kx, kx+1, 1).Slice(ctx, 1, ky, ky+1, 1).Contiguous(ctx)
			if wk.Dim(2) == 1 {
				wk = wk.Permute(ctx, 0, 1, 3, 2).Contiguous(ctx)
			} else {
				wk = wk.Reshape(ctx, 1, 1, wk.Dim(2), wk.Dim(3))
			}

			term := patch.Mul(ctx, wk)
			if out == nil {
				out = term
			} else {
				out = out.Add(ctx, term)
			}
		}
	}
	return out
}

func convOutputLength(inputLength, kernel, stride, padding int) int {
	if inputLength <= 0 {
		return 0
	}
	return (inputLength+2*padding-kernel)/stride + 1
}

func audioConvPadding(kernel int) int {
	return (kernel - 1) / 2
}

func parakeetPositionEmbeddings(ctx ml.Context, seqLen, hiddenSize int) ml.Tensor {
	half := hiddenSize / 2
	values := make([]float32, hiddenSize*(2*seqLen-1))
	for posIdx, pos := 0, seqLen-1; posIdx < 2*seqLen-1; posIdx, pos = posIdx+1, pos-1 {
		for i := range half {
			invFreq := math.Pow(10000, -float64(2*i)/float64(hiddenSize))
			angle := float64(pos) * invFreq
			values[posIdx*hiddenSize+2*i] = float32(math.Sin(angle))
			values[posIdx*hiddenSize+2*i+1] = float32(math.Cos(angle))
		}
	}
	return ctx.Input().FromFloats(values, hiddenSize, 2*seqLen-1)
}

func relativeShiftParakeet(ctx ml.Context, x ml.Tensor, seqLen, numHeads int) ml.Tensor {
	positionLen := 2*seqLen - 1
	x = x.PadExt(ctx, 1, 0, 0, 0, 0, 0, 0, 0)
	x = x.Reshape(ctx, seqLen, positionLen+1, numHeads)
	x = x.Slice(ctx, 1, 1, positionLen+1, 1).Contiguous(ctx)
	x = x.Reshape(ctx, positionLen, seqLen, numHeads)
	return x.Slice(ctx, 0, 0, seqLen, 1).Contiguous(ctx)
}

func audioAttentionMask(ctx ml.Context, seqLen, validLen int) ml.Tensor {
	values := make([]float32, seqLen*seqLen)
	for q := range seqLen {
		for k := range seqLen {
			if q >= validLen || k >= validLen {
				values[q*seqLen+k] = -1e9
			}
		}
	}
	return ctx.Input().FromFloats(values, seqLen, seqLen, 1)
}

func newAudioModel(c fs.Config) *AudioModel {
	numLayers := int(c.Uint("audio.block_count", 0))
	if numLayers == 0 {
		return nil
	}
	return &AudioModel{
		Layers:       make([]AudioLayer, numLayers),
		AudioOptions: newAudioOptions(c),
	}
}

func newAudioProjector(c fs.Config) *AudioProjector {
	if c.Uint("audio.block_count", 0) == 0 {
		return nil
	}
	return &AudioProjector{}
}

func newAudioOptions(c fs.Config) *AudioOptions {
	hiddenSize := int(c.Uint("audio.embedding_length", 1024))
	numHeads := int(c.Uint("audio.attention.head_count", 8))
	headDim := hiddenSize / max(1, numHeads)
	return &AudioOptions{
		hiddenSize:        hiddenSize,
		numHeads:          numHeads,
		headDim:           headDim,
		intermediateSize:  int(c.Uint("audio.feed_forward_length", uint32(hiddenSize*4))),
		convKernelSize:    int(c.Uint("audio.conv_kernel_size", 9)),
		melBins:           int(c.Uint("audio.num_mel_bins", 128)),
		sampleRate:        int(c.Uint("audio.sample_rate", 16000)),
		subsamplingKernel: int(c.Uint("audio.subsampling_conv_kernel_size", 3)),
		subsamplingStride: int(c.Uint("audio.subsampling_conv_stride", 2)),
		scaleInput:        c.Bool("audio.scale_input", false),
		eps:               c.Float("audio.attention.layer_norm_epsilon", 1e-5),
	}
}

func defaultAudioOptions() *AudioOptions {
	return &AudioOptions{
		hiddenSize:        1024,
		numHeads:          8,
		headDim:           128,
		intermediateSize:  4096,
		convKernelSize:    9,
		melBins:           128,
		sampleRate:        16000,
		subsamplingKernel: 3,
		subsamplingStride: 2,
		eps:               1e-5,
	}
}
