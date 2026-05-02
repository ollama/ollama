package gemma4

import (
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
)

// AudioModel holds the audio encoder and configuration.
type AudioModel struct {
	// SSCP: Sub-Sample Convolution Projection.
	SSCPConv0 *AudioConvBlock `gguf:"conv1d.0"`
	SSCPConv1 *AudioConvBlock `gguf:"conv1d.1"`

	// SSCP output projection (linear).
	SSCPInputProj *nn.Linear `gguf:"pre_encode.out"`

	// Conformer blocks.
	Layers []AudioConformerBlock `gguf:"blk"`

	// Output projection to embedder dimension.
	OutputProj *AudioOutputProj `gguf:"output_proj"`

	AudioModelOptions
}

type AudioOutputProj struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

// AudioModelOptions holds audio model hyperparameters.
type AudioModelOptions struct {
	hiddenSize     int
	numHeads       int
	headDim        int
	ffnSize        int
	numLayers      int
	melBins        int
	chunkSize      int
	maxPast        int
	maxFuture      int
	contextSize    int
	logitCap       float32
	residualWeight float32
	gradClip       float32
	convKernelSize int
	eps            float32
}

// AudioConvBlock is a single 2D convolution block for the SSCP.
type AudioConvBlock struct {
	Weight ml.Tensor     `gguf:"weight"`
	Norm   *nn.LayerNorm `gguf:"norm"`
}

// AudioConformerBlock is a single conformer layer.
// All tensors are flat at the block level (a.blk.N.<name>) using underscore naming.
type AudioConformerBlock struct {
	// Block-level norm
	Norm *nn.RMSNorm `gguf:"layer_pre_norm"`

	// FFW start
	FFWNorm     *nn.RMSNorm           `gguf:"ffn_norm"`
	FFWUp       *AudioClippableLinear `gguf:"ffn_up"`
	FFWDown     *AudioClippableLinear `gguf:"ffn_down"`
	FFWPostNorm *nn.RMSNorm           `gguf:"ffn_post_norm"`

	// FFW end
	FFWNorm1     *nn.RMSNorm           `gguf:"ffn_norm_1"`
	FFWUp1       *AudioClippableLinear `gguf:"ffn_up_1"`
	FFWDown1     *AudioClippableLinear `gguf:"ffn_down_1"`
	FFWPostNorm1 *nn.RMSNorm           `gguf:"ffn_post_norm_1"`

	// Attention
	AttnQ        *AudioClippableLinear `gguf:"attn_q"`
	AttnK        *AudioClippableLinear `gguf:"attn_k"`
	AttnV        *AudioClippableLinear `gguf:"attn_v"`
	AttnOut      *AudioClippableLinear `gguf:"attn_out"`
	AttnPreNorm  *nn.RMSNorm           `gguf:"ln1"`
	AttnPostNorm *nn.RMSNorm           `gguf:"ln2"`
	LinearPos    ml.Tensor             `gguf:"linear_pos.weight"`
	PerDimScale  ml.Tensor             `gguf:"per_dim_scale.weight"`

	// LightConv1d
	ConvPW1  *AudioClippableLinear `gguf:"conv_pw1"`
	ConvPW2  *AudioClippableLinear `gguf:"conv_pw2"`
	ConvDW   ml.Tensor             `gguf:"conv_dw.weight"`
	ConvNorm *nn.RMSNorm           `gguf:"conv_norm"`
	NormConv *nn.RMSNorm           `gguf:"norm_conv"`
}

// AudioClippableLinear is a linear layer with optional input/output clamping.
type AudioClippableLinear struct {
	Weight    ml.Tensor `gguf:"weight"`
	Bias      ml.Tensor `gguf:"bias"`
	InputMin  ml.Tensor `gguf:"input_min"`
	InputMax  ml.Tensor `gguf:"input_max"`
	OutputMin ml.Tensor `gguf:"output_min"`
	OutputMax ml.Tensor `gguf:"output_max"`

	// Cached scalar clamp values (populated on first forward).
	inMin, inMax, outMin, outMax float32
	clampsLoaded                 bool
}

func (l *AudioClippableLinear) loadClamps() {
	if l.clampsLoaded {
		return
	}
	l.clampsLoaded = true
	if l.InputMin != nil {
		vals := l.InputMin.BackendGet()
		if len(vals) > 0 {
			l.inMin = vals[0]
		}
	}
	if l.InputMax != nil {
		vals := l.InputMax.BackendGet()
		if len(vals) > 0 {
			l.inMax = vals[0]
		}
	}
	if l.OutputMin != nil {
		vals := l.OutputMin.BackendGet()
		if len(vals) > 0 {
			l.outMin = vals[0]
		}
	}
	if l.OutputMax != nil {
		vals := l.OutputMax.BackendGet()
		if len(vals) > 0 {
			l.outMax = vals[0]
		}
	}
}

func (l *AudioClippableLinear) Forward(ctx ml.Context, x ml.Tensor) ml.Tensor {
	l.loadClamps()
	if l.inMax != 0 {
		x = x.Clamp(ctx, l.inMin, l.inMax)
	}
	out := l.Weight.Mulmat(ctx, x)
	if l.Bias != nil {
		out = out.Add(ctx, l.Bias)
	}
	if l.outMax != 0 {
		out = out.Clamp(ctx, l.outMin, l.outMax)
	}
	return out
}

// AudioMultimodalProjector is the audio-to-text embedding projector.
type AudioMultimodalProjector struct {
	Projection *AudioClippableLinear `gguf:"input_projection"`
	FC         *AudioFC              `gguf:"fc"`
}

type AudioFC struct {
	Weight ml.Tensor `gguf:"weight"`
	Bias   ml.Tensor `gguf:"bias"`
}

func (p *AudioMultimodalProjector) Forward(ctx ml.Context, x ml.Tensor, eps float32) ml.Tensor {
	// FC: output projection from conformer to embedder dimension.
	x = p.FC.Weight.Mulmat(ctx, x)
	if p.FC.Bias != nil {
		x = x.Add(ctx, p.FC.Bias)
	}
	// Pre-projection RMSNorm (without learned weight) — matches Python's embedding_pre_projection_norm.
	x = x.RMSNorm(ctx, nil, eps)
	// Embedding projection to text hidden size.
	x = p.Projection.Forward(ctx, x)
	return x
}

// ForwardAudio encodes mel spectrogram features into soft tokens.
// melFeatures: float32 tensor with ne[0]=melBins, ne[1]=numFrames.
// Returns: [hiddenSize, numTokens] tensor.
func (m *AudioModel) ForwardAudio(ctx ml.Context, melFeatures ml.Tensor, proj *AudioMultimodalProjector, opts *AudioModelOptions) ml.Tensor {
	// SSCP Conv2D input: ne[0]=F (freq/width), ne[1]=T (time/height), ne[2]=C_in, ne[3]=B
	// melFeatures is [melBins, numFrames], add channel and batch dims.
	x := melFeatures.Reshape(ctx, melFeatures.Dim(0), melFeatures.Dim(1), 1, 1)

	// SSCP Conv block 0: [F, T, 1, 1] → [F', T', C0, 1]
	x = forwardConvBlock(ctx, m.SSCPConv0, x, opts)

	// SSCP Conv block 1: [F', T', C0, 1] → [F'', T'', C1, 1]
	x = forwardConvBlock(ctx, m.SSCPConv1, x, opts)

	// After conv blocks, layout is [F'', T'', C_out, B].
	// Permute to [C_out*F'', T'', B] for linear projection (channels+freq in ne[0]).
	fOut := x.Dim(0)
	tOut := x.Dim(1)
	cOut := x.Dim(2)
	// Permute [F'', T'', C, B] → [C, F'', T'', B]
	// (1,2,0,3): old[0]→pos1, old[1]→pos2, old[2]→pos0
	x = x.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
	x = x.Reshape(ctx, cOut*fOut, tOut)

	// Linear projection to hidden size.
	x = m.SSCPInputProj.Forward(ctx, x)

	// Build causal-valid mask for conformer attention.
	causalMask := buildCausalValidMaskF32(opts.chunkSize, opts.maxPast, opts.maxFuture)

	// Run conformer blocks.
	for i := range m.Layers {
		x = m.Layers[i].Forward(ctx, x, causalMask, opts, i)
	}

	// Output projection.
	if m.OutputProj != nil {
		x = m.OutputProj.Weight.Mulmat(ctx, x)
		if m.OutputProj.Bias != nil {
			x = x.Add(ctx, m.OutputProj.Bias)
		}
	}

	// Audio embedder: project to text embedding space.
	if proj != nil {
		x = proj.Forward(ctx, x, opts.eps)
	}

	return x
}

// forwardConvBlock runs a single SSCP Conv2D block.
// Conv2D receiver is the kernel, argument is the input data.
// Input: [F, T, C_in, B]. Output: [F', T', C_out, B].
func forwardConvBlock(ctx ml.Context, block *AudioConvBlock, x ml.Tensor, opts *AudioModelOptions) ml.Tensor {
	// Conv2D: kernel.Conv2D(ctx, input, s0, s1, p0, p1, d0, d1)
	// Kernel is 3x3, stride 2x2, padding 1x1 (matching SSCP config).
	// Output layout: [F', T', C_out, B]
	// Make weight contiguous — the shape reversal in the converter creates
	// a tensor where the physical data order doesn't match ne[]/stride[].
	weight := block.Weight.Contiguous(ctx)
	x = weight.Conv2D(ctx, x, 2, 2, 1, 1, 1, 1)

	// LayerNorm needs channels in ne[0]. Permute [F', T', C_out, B] → [C_out, F', T', B],
	// norm, then permute back.
	// GGML permute: axis i says where old axis i goes.
	// (1,2,0,3): old[0]→pos1, old[1]→pos2, old[2]→pos0 → [C_out, F', T', B]
	x = x.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)
	x = block.Norm.Forward(ctx, x, opts.eps)
	// (2,0,1,3): old[0]→pos2, old[1]→pos0, old[2]→pos1 → [F', T', C_out, B]
	x = x.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)

	x = x.RELU(ctx)
	return x
}

// Forward runs a single conformer block.
func (cb *AudioConformerBlock) Forward(ctx ml.Context, x ml.Tensor, causalMask []float32, opts *AudioModelOptions, blockIdx int) ml.Tensor {
	// FFW start (half-residual).
	x = cb.forwardFFW(ctx, cb.FFWNorm, cb.FFWUp, cb.FFWDown, cb.FFWPostNorm, x, opts)

	// Self-attention.
	x = cb.forwardAttention(ctx, x, causalMask, opts, blockIdx)

	// Lightweight Conv1d.
	x = cb.forwardLightConv(ctx, x, opts, blockIdx)

	// FFW end (half-residual).
	x = cb.forwardFFW(ctx, cb.FFWNorm1, cb.FFWUp1, cb.FFWDown1, cb.FFWPostNorm1, x, opts)

	// Gradient clipping + final norm.
	x = x.Clamp(ctx, -opts.gradClip, opts.gradClip)
	x = cb.Norm.Forward(ctx, x, opts.eps)

	return x
}

// forwardFFW runs a feedforward module with half-residual connection.
func (cb *AudioConformerBlock) forwardFFW(ctx ml.Context, preNorm *nn.RMSNorm, up, down *AudioClippableLinear, postNorm *nn.RMSNorm, x ml.Tensor, opts *AudioModelOptions) ml.Tensor {
	residual := x
	x = x.Clamp(ctx, -opts.gradClip, opts.gradClip)
	x = preNorm.Forward(ctx, x, opts.eps)
	x = up.Forward(ctx, x)
	x = x.SILU(ctx)
	x = down.Forward(ctx, x)
	x = x.Clamp(ctx, -opts.gradClip, opts.gradClip)
	x = postNorm.Forward(ctx, x, opts.eps)
	x = x.Scale(ctx, float64(opts.residualWeight))
	return residual.Add(ctx, x)
}

// forwardAttention runs the conformer block-local attention with relative position embeddings.
func (cb *AudioConformerBlock) forwardAttention(ctx ml.Context, x ml.Tensor, causalMask []float32, opts *AudioModelOptions, blockIdx int) ml.Tensor {
	residual := x
	x = x.Clamp(ctx, -opts.gradClip, opts.gradClip)
	x = cb.AttnPreNorm.Forward(ctx, x, opts.eps)

	hiddenSize := x.Dim(0)
	seqLen := x.Dim(1)

	// QKV projections: [hiddenSize, seqLen] → [headDim, numHeads, seqLen]
	q := cb.AttnQ.Forward(ctx, x).Reshape(ctx, opts.headDim, opts.numHeads, seqLen)
	k := cb.AttnK.Forward(ctx, x).Reshape(ctx, opts.headDim, opts.numHeads, seqLen)
	v := cb.AttnV.Forward(ctx, x).Reshape(ctx, opts.headDim, opts.numHeads, seqLen)

	// Per-dim scaling for queries: (headDim^-0.5 / log(2)) * softplus(per_dim_scale)
	// per_dim_scale is already softplus'd from the converter.
	qScale := float64(math.Pow(float64(opts.headDim), -0.5)) / math.Log(2)
	q = q.Scale(ctx, qScale)
	if cb.PerDimScale != nil {
		q = q.Mul(ctx, cb.PerDimScale)
	}

	// Key scaling: softplus(1) / log(2) — matches the query base scaling convention.
	kScale := math.Log(1+math.E) / math.Log(2)
	k = k.Scale(ctx, kScale)

	// Build sinusoidal position embeddings for the block-local context.
	maxSpan := opts.maxPast + opts.maxFuture + 1 // 13 unique relative positions
	posEmb := cb.buildPositionEmbeddings(ctx, maxSpan, opts)
	// posEmb: [headDim, numHeads, maxSpan]

	// Block-local attention: process chunks of size chunkSize.
	chunkSize := opts.chunkSize
	numChunks := (seqLen + chunkSize - 1) / chunkSize
	contextSize := opts.contextSize

	// Pad q/k/v to multiple of chunkSize on the time dimension (dim 2).
	padT := numChunks*chunkSize - seqLen
	if padT > 0 {
		q = q.Pad(ctx, 0, 0, padT, 0)
		k = k.Pad(ctx, 0, 0, padT, 0)
		v = v.Pad(ctx, 0, 0, padT, 0)
	}
	paddedLen := numChunks * chunkSize

	// Pad k/v for context extraction: add maxPast on left, (maxFuture+chunkSize-1) on right.
	// Use Pad (right) + PadExt (left) workaround since PadExt+Slice has issues.
	// Actually use Concat with zero tensors for reliable left-padding.
	padLeft := opts.maxPast
	padRight := opts.maxFuture + chunkSize - 1
	zeroLeft := ctx.Input().FromFloats(make([]float32, opts.headDim*opts.numHeads*padLeft), opts.headDim, opts.numHeads, padLeft)
	zeroRight := ctx.Input().FromFloats(make([]float32, opts.headDim*opts.numHeads*padRight), opts.headDim, opts.numHeads, padRight)
	kPadded := zeroLeft.Concat(ctx, k, 2).Concat(ctx, zeroRight, 2)
	vPadded := zeroLeft.Concat(ctx, v, 2).Concat(ctx, zeroRight, 2)

	// Reshape q into chunks: [headDim, numHeads, numChunks, chunkSize]
	qChunked := q.Reshape(ctx, opts.headDim, opts.numHeads, numChunks, chunkSize)

	// Process each chunk and collect results.
	chunkOutputs := make([]ml.Tensor, numChunks)
	for u := range numChunks {
		// Extract query block: [headDim, numHeads, 1, chunkSize] → [headDim, numHeads, chunkSize]
		qBlock := qChunked.Slice(ctx, 2, u, u+1, 1).Reshape(ctx, opts.headDim, opts.numHeads, chunkSize)

		// Extract key/value context: [headDim, numHeads, contextSize]
		cStart := u * chunkSize // offset in kPadded (padLeft already accounts for left context)
		kCtx := kPadded.Slice(ctx, 2, cStart, cStart+contextSize, 1).Contiguous(ctx)
		vCtx := vPadded.Slice(ctx, 2, cStart, cStart+contextSize, 1).Contiguous(ctx)

		// Content-content logits: qBlock^T @ kCtx → [chunkSize, contextSize] per head.
		// Mulmat(a, b) = a^T @ b. We want Q^T K, so: kCtx.Mulmat(qBlock) but that gives
		// [numHeads, chunkSize, contextSize] with wrong batching.
		// Instead: permute to [headDim, chunkSize, numHeads] and [headDim, contextSize, numHeads]
		// then Mulmat batches over numHeads.
		// GGML permute(0,2,1,3): old[0]→0, old[1]→2, old[2]→1
		qP := qBlock.Permute(ctx, 0, 2, 1, 3) // [headDim, chunkSize, numHeads]
		kP := kCtx.Permute(ctx, 0, 2, 1, 3)   // [headDim, contextSize, numHeads]

		termAC := kP.MulmatFullPrec(ctx, qP) // [contextSize, chunkSize, numHeads]

		// Content-position logits: qBlock^T @ posEmb → [chunkSize, maxSpan] per head.
		pP := posEmb.Permute(ctx, 0, 2, 1, 3)   // [headDim, maxSpan, numHeads]
		termBDRaw := pP.MulmatFullPrec(ctx, qP) // [maxSpan, chunkSize, numHeads]

		// Relative shift: [maxSpan, chunkSize, numHeads] → [contextSize, chunkSize, numHeads]
		termBD := cb.relativeShiftGGML(ctx, termBDRaw, maxSpan, chunkSize, contextSize, opts.numHeads)

		// Combined logits.
		logits := termAC.Add(ctx, termBD)

		// Logit softcap: tanh(logits / cap) * cap
		logits = logits.Scale(ctx, 1.0/float64(opts.logitCap))
		logits = logits.Tanh(ctx)
		logits = logits.Scale(ctx, float64(opts.logitCap))

		// Apply combined causal + validity mask.
		// causalMask [chunkSize * contextSize]: 1=causal-allowed, 0=masked.
		// Validity: context positions before the actual sequence start are invalid.
		// For chunk u, context position c corresponds to actual time: u*chunkSize + c - padLeft.
		// Valid if 0 <= actual_time < seqLen.
		// Mask tensor layout: [contextSize, chunkSize, 1] with ne[0]=contextSize contiguous.
		// Element at (context=j, chunk=i) is at flat index: i*contextSize + j.
		maskData := make([]float32, contextSize*chunkSize)
		for i := range chunkSize {
			for j := range contextSize {
				actualTime := u*chunkSize + j - padLeft
				causalOK := causalMask[i*contextSize+j] > 0
				validOK := actualTime >= 0 && actualTime < seqLen
				if causalOK && validOK {
					maskData[i*contextSize+j] = 0
				} else {
					maskData[i*contextSize+j] = -1e9
				}
			}
		}
		mask := ctx.Input().FromFloats(maskData, contextSize, chunkSize, 1) // 3D for broadcasting over numHeads
		logits = logits.Add(ctx, mask)

		// Softmax over context dimension (dim 0 = contextSize).
		logits = logits.Softmax(ctx) // softmax over ne[0]=contextSize

		// Weighted sum: logits^T @ vCtx.
		// logits: [contextSize, chunkSize, numHeads], vCtx: [headDim, numHeads, contextSize]
		// vCtx permuted: [headDim, contextSize, numHeads]
		vP := vCtx.Permute(ctx, 0, 2, 1, 3) // [headDim, contextSize, numHeads]
		// Weighted sum: for each head, value[headDim, contextSize] @ weights[contextSize, chunkSize]
		// = [headDim, chunkSize].
		// Mulmat(a, b) = a^T @ b. Need a=[contextSize, headDim, numHeads], b=[contextSize, chunkSize, numHeads].
		vPT := vP.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx) // [contextSize, headDim, numHeads]
		chunkOut := vPT.Mulmat(ctx, logits)                // [headDim, chunkSize, numHeads]

		// Permute back to [headDim, numHeads, chunkSize]
		chunkOut = chunkOut.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
		chunkOutputs[u] = chunkOut
	}

	// Concatenate chunk outputs along time dimension.
	var attnOut ml.Tensor
	if numChunks == 1 {
		attnOut = chunkOutputs[0]
	} else {
		attnOut = chunkOutputs[0]
		for _, co := range chunkOutputs[1:] {
			attnOut = attnOut.Concat(ctx, co, 2)
		}
	}

	// Trim to original sequence length if we padded.
	if paddedLen > seqLen {
		attnOut = attnOut.Slice(ctx, 2, 0, seqLen, 1).Contiguous(ctx)
	}

	// Reshape to [hiddenSize, seqLen] and project.
	attnOut = attnOut.Reshape(ctx, hiddenSize, seqLen)
	x = cb.AttnOut.Forward(ctx, attnOut)
	x = x.Clamp(ctx, -opts.gradClip, opts.gradClip)
	x = cb.AttnPostNorm.Forward(ctx, x, opts.eps)

	return residual.Add(ctx, x)
}

// buildPositionEmbeddings builds sinusoidal position embeddings and projects through linear_pos.
// Returns [headDim, numHeads, maxSpan] tensor.
func (cb *AudioConformerBlock) buildPositionEmbeddings(ctx ml.Context, maxSpan int, opts *AudioModelOptions) ml.Tensor {
	halfDim := opts.hiddenSize / 2
	hiddenSize := opts.hiddenSize

	// inv_timescales: exp(-i * log(10000) / max(D/2-1, 1))
	logInc := math.Log(10000.0) / math.Max(float64(halfDim-1), 1)

	// Sinusoidal embeddings for relative positions [maxPast, maxPast-1, ..., -maxFuture].
	posData := make([]float32, hiddenSize*maxSpan)
	for p := range maxSpan {
		relPos := float64(opts.maxPast - p)
		for d := range halfDim {
			angle := relPos * math.Exp(float64(-d)*logInc)
			posData[p*hiddenSize+d] = float32(math.Sin(angle))
			posData[p*hiddenSize+halfDim+d] = float32(math.Cos(angle))
		}
	}

	// Create [hiddenSize, maxSpan] input tensor.
	posEmb := ctx.Input().FromFloats(posData, hiddenSize, maxSpan)

	// Project through linear_pos: [hiddenSize, maxSpan] → Mulmat → [numHeads*headDim, maxSpan]
	projPos := cb.LinearPos.Mulmat(ctx, posEmb)

	// Reshape to [headDim, numHeads, maxSpan].
	return projPos.Reshape(ctx, opts.headDim, opts.numHeads, maxSpan)
}

// relativeShiftGGML performs the relative shift to extract correct position logits.
// Input: [maxSpan, chunkSize, numHeads]. Output: [contextSize, chunkSize, numHeads].
func (cb *AudioConformerBlock) relativeShiftGGML(ctx ml.Context, x ml.Tensor, maxSpan, chunkSize, contextSize, numHeads int) ml.Tensor {
	// The shift trick: pad ne[0] to contextSize+1, reshape to flatten first two dims,
	// skip first (contextSize+1-maxSpan) elements, take contextSize*chunkSize elements, reshape back.
	padAmt := contextSize + 1 - maxSpan
	if padAmt > 0 {
		x = x.Pad(ctx, padAmt, 0, 0, 0) // [maxSpan+padAmt, chunkSize, numHeads] = [contextSize+1, chunkSize, numHeads]
	}
	// Reshape to [(contextSize+1)*chunkSize, numHeads]
	x = x.Reshape(ctx, (contextSize+1)*chunkSize, numHeads)
	// Take the first contextSize*chunkSize elements (the standard relative shift trick).
	x = x.Slice(ctx, 0, 0, contextSize*chunkSize, 1).Contiguous(ctx)
	// Reshape to [contextSize, chunkSize, numHeads]
	return x.Reshape(ctx, contextSize, chunkSize, numHeads)
}

// forwardLightConv runs the lightweight depthwise convolution module.
func (cb *AudioConformerBlock) forwardLightConv(ctx ml.Context, x ml.Tensor, opts *AudioModelOptions, blockIdx int) ml.Tensor {
	residual := x

	x = cb.ConvNorm.Forward(ctx, x, opts.eps)
	x = cb.ConvPW1.Forward(ctx, x) // [2*D, T, B]

	// GLU: split in half along dim 0, sigmoid gate, multiply.
	d := x.Dim(0) / 2
	data := x.Slice(ctx, 0, 0, d, 1).Contiguous(ctx)
	gate := x.Slice(ctx, 0, d, d*2, 1).Contiguous(ctx).Sigmoid(ctx)
	x = data.Mul(ctx, gate) // [D, T, B]

	// Depthwise Conv1d: manual implementation using model weight tensor slices.
	// Kernel cb.ConvDW shape: [K=5, D=1024] (ne[0]=K, ne[1]=D) after shape reversal.
	// Actually in GGML, ne[0]=K=5 contiguous, ne[1]=D=1024.
	// We need per-tap weights [D] and shifted input copies.
	kernelSize := cb.ConvDW.Dim(0) // K=5
	seqLen := x.Dim(1)

	// Transpose kernel to [D, K] for per-tap slicing.
	// GGML permute(1,0,2,3): old[0]→pos1, old[1]→pos0 → swap ne[0] and ne[1]
	kernelT := cb.ConvDW.Permute(ctx, 1, 0, 2, 3).Contiguous(ctx) // [D, K]

	var convOut ml.Tensor
	for k := range kernelSize {
		shift := kernelSize - 1 - k
		var shifted ml.Tensor
		if shift == 0 {
			shifted = x
		} else {
			trimmed := x.Slice(ctx, 1, 0, seqLen-shift, 1).Contiguous(ctx)
			shifted = trimmed.PadExt(ctx, 0, 0, shift, 0, 0, 0, 0, 0)
		}

		wk := kernelT.Slice(ctx, 1, k, k+1, 1).Contiguous(ctx) // [D, 1]
		term := shifted.Mul(ctx, wk)
		if convOut == nil {
			convOut = term
		} else {
			convOut = convOut.Add(ctx, term)
		}
	}
	x = convOut

	x = x.Clamp(ctx, -opts.gradClip, opts.gradClip)
	x = cb.NormConv.Forward(ctx, x, opts.eps)
	x = x.SILU(ctx)
	x = cb.ConvPW2.Forward(ctx, x)

	return x.Add(ctx, residual)
}

func newAudioModel(c fs.Config) *AudioModel {
	numLayers := int(c.Uint("audio.block_count", 0))
	if numLayers == 0 {
		return nil
	}
	return &AudioModel{
		Layers: make([]AudioConformerBlock, numLayers),
	}
}

func newAudioModelOptions(c fs.Config) *AudioModelOptions {
	hiddenSize := int(c.Uint("audio.embedding_length", 0))
	if hiddenSize == 0 {
		return nil
	}
	numHeads := int(c.Uint("audio.attention.head_count", 8))
	headDim := hiddenSize / numHeads
	chunkSize := 12 // default conformer chunk size
	maxPast := 12   // conf_attention_context_left - 1
	maxFuture := 0  // conf_attention_context_right
	convKernel := int(c.Uint("audio.conv_kernel_size", 5))

	eps := c.Float("audio.attention.layer_norm_epsilon", 1e-6)

	return &AudioModelOptions{
		hiddenSize:     hiddenSize,
		numHeads:       numHeads,
		headDim:        headDim,
		ffnSize:        int(c.Uint("audio.feed_forward_length", uint32(hiddenSize*4))),
		numLayers:      int(c.Uint("audio.block_count", 12)),
		melBins:        int(c.Uint("audio.num_mel_bins", 128)),
		chunkSize:      chunkSize,
		maxPast:        maxPast,
		maxFuture:      maxFuture,
		contextSize:    chunkSize + maxPast + maxFuture,
		logitCap:       50.0,
		residualWeight: 0.5,
		gradClip:       1e10,
		convKernelSize: convKernel,
		eps:            float32(eps),
	}
}

// buildCausalValidMaskF32 creates the causal-valid mask for block-local attention.
// Returns flat [chunkSize * contextSize] float32 data (1.0 = allowed, 0.0 = masked).
func buildCausalValidMaskF32(chunkSize, maxPast, maxFuture int) []float32 {
	contextSize := chunkSize + maxPast + maxFuture
	upperDiag := maxPast + maxFuture

	result := make([]float32, chunkSize*contextSize)
	for r := range chunkSize {
		for c := range contextSize {
			lower := (r <= c)           // tril(contextSize, chunkSize) transposed
			upper := (c <= r+upperDiag) // tril(chunkSize, contextSize, diag=upperDiag)
			if lower && upper {
				result[r*contextSize+c] = 1.0
			}
		}
	}
	return result
}
