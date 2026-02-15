package gemma3n

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model/input"
)

type TextModel struct {
	TokenEmbedding *TextScaledWordEmbedding `gguf:"token_embd"`

	*PerLayerProjector

	AltupEmbd   *nn.Linear `gguf:"altup_proj"`
	AltupUnembd *nn.Linear `gguf:"altup_unembd_proj"`

	TextLayers []TextLayer `gguf:"blk"`
	OutputNorm *nn.RMSNorm `gguf:"output_norm"`
	Output     *nn.Linear  `gguf:"output,alt:token_embd"`

	TextOptions
}

func (m *TextModel) Forward(ctx ml.Context, batch input.Batch, cache kvcache.Cache) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))
	// Create a tensor of a single float32 value of 1.0 to use for altup correction
	one := ctx.Input().FromFloats([]float32{1.0}, 1)

	inputs := m.TokenEmbedding.Forward(ctx, batch.Inputs, math.Sqrt(float64(m.hiddenSize)))
	inputsPerLayer := m.PerLayerProjector.Forward(ctx, batch, inputs, &m.TextOptions)

	targetMagnitude := inputs.Sqr(ctx).Mean(ctx).Sqrt(ctx)
	targetMagnitude = targetMagnitude.Repeat(ctx, 2, m.altupInputs-1)

	hiddenState := inputs.Repeat(ctx, 2, m.altupInputs-1)
	altupProj := m.AltupEmbd.Forward(ctx, hiddenState)
	altupProj = altupProj.Mul(ctx, targetMagnitude.Div(ctx, altupProj.Sqr(ctx).Mean(ctx).Sqrt(ctx)))

	hiddenStates := inputs.Concat(ctx, altupProj, 2)

	firstSharedKeyValue := m.hiddenLayers - m.sharedKeyValueLayers
	for i, layer := range m.TextLayers {
		if i < firstSharedKeyValue {
			cache.SetLayer(i)
		} else if m.isLocal(i) {
			cache.SetLayer(firstSharedKeyValue - 2)
		} else {
			cache.SetLayer(firstSharedKeyValue - 1)
		}

		var layerType int
		ropeBase := m.ropeBase
		if m.isLocal(i) {
			layerType = 1
			ropeBase = m.ropeBaseLocal
		}

		cache.(*kvcache.WrapperCache).SetLayerType(layerType)

		// inputPerLayer = inputsPerLayer[:, i, :].squeeze(1)
		inputPerLayer := inputsPerLayer.View(ctx, i*inputsPerLayer.Stride(1), inputsPerLayer.Dim(0), inputsPerLayer.Stride(2), inputsPerLayer.Dim(2))
		hiddenStates = layer.Forward(ctx, hiddenStates, inputPerLayer, positions, one, cache, i >= firstSharedKeyValue, ropeBase, float64(m.activationSparsityScale[i]), &m.TextOptions)
	}

	// hiddenStates = hiddenStates[:, :, 0]
	hiddenStates0 := hiddenStates.Slice(ctx, 2, 0, 1, 1)
	targetMagnitude = hiddenStates0.Sqr(ctx).Mean(ctx).Sqrt(ctx)
	targetMagnitude = targetMagnitude.Repeat(ctx, 2, m.altupInputs-1)

	// hiddenState = hiddenStates[:, :, 1:]
	hiddenState = hiddenStates.Slice(ctx, 2, 1, hiddenStates.Dim(2), 1)
	altupUnembdProj := m.AltupUnembd.Forward(ctx, hiddenState)
	altupUnembdProj = altupUnembdProj.Mul(ctx, targetMagnitude.Div(ctx, altupUnembdProj.Sqr(ctx).Mean(ctx).Sqrt(ctx)))

	hiddenStates = hiddenStates0.Concat(ctx, altupUnembdProj, 2)

	hiddenStates = hiddenStates.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx).Mean(ctx)
	hiddenStates = hiddenStates.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	hiddenStates = hiddenStates.Rows(ctx, batch.Outputs)

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	ropeBase := m.ropeBase
	if m.isLocal(layer) {
		ropeBase = m.ropeBaseLocal
	}

	return m.applyRotaryPositionEmbeddings(ctx, key, shift, ropeBase), nil
}

type TextScaledWordEmbedding struct {
	*nn.Embedding
}

func (e TextScaledWordEmbedding) Forward(ctx ml.Context, inputIDs ml.Tensor, scale float64) ml.Tensor {
	return e.Embedding.Forward(ctx, inputIDs).Scale(ctx, scale)
}

type PerLayerProjector struct {
	TokenEmbedding *TextScaledWordEmbedding `gguf:"per_layer_token_embd"`
	Projector      *nn.Linear               `gguf:"per_layer_model_proj"`
	Norm           *nn.RMSNorm              `gguf:"per_layer_proj_norm"`
}

func (p PerLayerProjector) Forward(ctx ml.Context, batch input.Batch, inputs ml.Tensor, opts *TextOptions) ml.Tensor {
	inputsPerLayer := p.TokenEmbedding.Forward(ctx, batch.Inputs, math.Sqrt(float64(opts.hiddenSizePerLayerInput)))
	inputsPerLayer = inputsPerLayer.Reshape(ctx, opts.hiddenSizePerLayerInput, opts.hiddenLayers, batch.Inputs.Dim(0), batch.Inputs.Dim(1))

	perLayerProjection := p.Projector.Forward(ctx, inputs)
	perLayerProjection = perLayerProjection.Scale(ctx, math.Sqrt(float64(opts.hiddenSize)))
	perLayerProjection = perLayerProjection.Reshape(ctx, opts.hiddenSizePerLayerInput, opts.hiddenLayers, inputs.Dim(1))
	perLayerProjection = p.Norm.Forward(ctx, perLayerProjection, opts.eps)

	if inputsPerLayer != nil {
		perLayerProjection = perLayerProjection.Add(ctx, inputsPerLayer)
		perLayerProjection = perLayerProjection.Scale(ctx, 1/math.Sqrt(2))
	}

	return perLayerProjection
}

type TextLayer struct {
	*AltUp
	*Laurel

	AttentionNorm     *nn.RMSNorm `gguf:"attn_norm"`
	Attention         *TextAttention
	PostAttentionNorm *nn.RMSNorm `gguf:"post_attention_norm"`

	MLPNorm     *nn.RMSNorm `gguf:"ffn_norm"`
	MLP         *TextMLP
	PostMLPNorm *nn.RMSNorm `gguf:"post_ffw_norm"`

	PerLayerInputGate  *nn.Linear  `gguf:"inp_gate"`
	PerLayerProjection *nn.Linear  `gguf:"proj"`
	PostPerLayerNorm   *nn.RMSNorm `gguf:"post_norm"`
}

func (d TextLayer) Forward(ctx ml.Context, hiddenStates, perLayerInput, positions, one ml.Tensor, cache kvcache.Cache, sharedKV bool, ropeBase float32, activationSparsityScale float64, opts *TextOptions) ml.Tensor {
	predictions := d.Predict(ctx, hiddenStates, opts)
	active := opts.altupActive(ctx, predictions)

	attn := d.AttentionNorm.Forward(ctx, active, opts.eps)
	laurel := d.Laurel.Forward(ctx, attn, opts)

	attn = d.Attention.Forward(ctx, attn, positions, cache, sharedKV, ropeBase, opts)
	attn = d.PostAttentionNorm.Forward(ctx, attn, opts.eps)
	attn = active.Add(ctx, attn)
	attn = attn.Add(ctx, laurel).Scale(ctx, 1/math.Sqrt(2))

	mlp := d.MLPNorm.Forward(ctx, attn, opts.eps)
	mlp = d.MLP.Forward(ctx, mlp, activationSparsityScale)
	mlp = d.PostMLPNorm.Forward(ctx, mlp, opts.eps)
	mlp = attn.Add(ctx, mlp)

	predictions = d.Correct(ctx, predictions, mlp, one, opts)
	active = opts.altupActive(ctx, predictions)
	if opts.altupCorrectScale {
		active = d.ScaleCorrectedOutput(ctx, active)
	}

	active = d.PerLayerInputGate.Forward(ctx, active)
	active = active.GELU(ctx, perLayerInput)

	active = d.PerLayerProjection.Forward(ctx, active)
	active = d.PostPerLayerNorm.Forward(ctx, active, opts.eps)

	// inactive := predictions[:, :, 1:]
	inactive := predictions.Slice(ctx, 2, 1, predictions.Dim(2), 1)
	active = inactive.Add(ctx, active)

	predictions0 := predictions.Slice(ctx, 2, 0, 1, 1)
	return predictions0.Concat(ctx, active, 2)
}

type AltUp struct {
	CorrectionScale       ml.Tensor   `gguf:"altup_correct_scale.weight"`
	PredictionCoefficient *nn.Linear  `gguf:"altup_predict_coef"`
	CorrectionCoefficient *nn.Linear  `gguf:"altup_correct_coef"`
	Router                *nn.Linear  `gguf:"altup_router"`
	RouterNorm            *nn.RMSNorm `gguf:"altup_router_norm"`
}

func (a AltUp) computeRouterModalities(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	routerInputs := a.RouterNorm.Forward(ctx, hiddenStates, opts.eps).Scale(ctx, 1.0/float64(opts.hiddenSize))
	return a.Router.Forward(ctx, routerInputs).Tanh(ctx)
}

func (a AltUp) Predict(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	modalities := a.computeRouterModalities(ctx, opts.altupActive(ctx, hiddenStates), opts)

	coefficients := a.PredictionCoefficient.Forward(ctx, modalities)
	coefficients = coefficients.Reshape(ctx, opts.altupInputs, opts.altupInputs, coefficients.Dim(1), coefficients.Dim(2))

	predictions := coefficients.Mulmat(ctx, hiddenStates.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx))
	predictions = predictions.Permute(ctx, 2, 0, 1, 3).Contiguous(ctx)
	return predictions.Add(ctx, hiddenStates)
}

func (a AltUp) Correct(ctx ml.Context, predictions, activated, one ml.Tensor, opts *TextOptions) ml.Tensor {
	innovation := activated.Sub(ctx, opts.altupActive(ctx, predictions))
	innovation = innovation.Repeat(ctx, 2, opts.altupInputs)

	modalities := a.computeRouterModalities(ctx, activated, opts)
	coefficients := a.CorrectionCoefficient.Forward(ctx, modalities)
	coefficients = coefficients.Add(ctx, one)

	coefficients = coefficients.Reshape(ctx, 1, coefficients.Dim(0), coefficients.Dim(1))
	coefficients = coefficients.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)

	corrected := innovation.Mul(ctx, coefficients)
	corrected = corrected.Add(ctx, predictions)
	return corrected
}

func (a AltUp) ScaleCorrectedOutput(ctx ml.Context, predictions ml.Tensor) ml.Tensor {
	return predictions.Mul(ctx, a.CorrectionScale)
}

type Laurel struct {
	LinearLeft     *nn.Linear  `gguf:"laurel_l"`
	LinearRight    *nn.Linear  `gguf:"laurel_r"`
	PostLaurelNorm *nn.RMSNorm `gguf:"laurel_post_norm"`
}

func (l Laurel) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *TextOptions) ml.Tensor {
	residual := hiddenStates
	hiddenStates = l.LinearLeft.Forward(ctx, hiddenStates)
	hiddenStates = l.LinearRight.Forward(ctx, hiddenStates)
	hiddenStates = l.PostLaurelNorm.Forward(ctx, hiddenStates, opts.eps)
	return hiddenStates.Add(ctx, residual)
}

type TextAttention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output"`
}

func (attn TextAttention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, sharedKV bool, ropeBase float32, opts *TextOptions) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	query := attn.Query.Forward(ctx, hiddenStates)
	query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)
	query = attn.QueryNorm.Forward(ctx, query, opts.eps)
	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions, ropeBase)

	var key, value ml.Tensor
	if !sharedKV {
		key = attn.Key.Forward(ctx, hiddenStates)
		key = key.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)
		key = attn.KeyNorm.Forward(ctx, key, opts.eps)
		key = opts.applyRotaryPositionEmbeddings(ctx, key, positions, ropeBase)

		value = attn.Value.Forward(ctx, hiddenStates)
		value = value.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)
		value = value.RMSNorm(ctx, nil, opts.eps)
	}

	attention := nn.Attention(ctx, query, key, value, 1., cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return attn.Output.Forward(ctx, attention)
}

type TextMLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp TextMLP) Forward(ctx ml.Context, hiddenStates ml.Tensor, activationSparsityScale float64) ml.Tensor {
	upStates := mlp.Up.Forward(ctx, hiddenStates)
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates)
	if activationSparsityScale > 0 {
		mean := hiddenStates.Mean(ctx)
		std := hiddenStates.Stddev(ctx).Scale(ctx, activationSparsityScale)
		cutoff := mean.Add(ctx, std)
		hiddenStates = hiddenStates.Sub(ctx, cutoff).RELU(ctx)
	}

	hiddenStates = hiddenStates.GELU(ctx, upStates)
	hiddenStates = mlp.Down.Forward(ctx, hiddenStates)
	return hiddenStates
}

type TextOptions struct {
	hiddenLayers            int
	hiddenSize              int
	hiddenSizePerLayerInput int
	numHeads, numKVHeads    int
	keyLength, valueLength  int
	sharedKeyValueLayers    int

	altupActiveIndex  int
	altupInputs       int
	altupCorrectScale bool

	eps           float32
	ropeBase      float32
	ropeBaseLocal float32
	ropeScale     float32

	slidingWindowPattern    []bool
	activationSparsityScale []float32
}

func (o *TextOptions) altupActive(ctx ml.Context, t ml.Tensor) ml.Tensor {
	// t[:, :, o.altupActiveIndex]
	return t.Slice(ctx, 2, o.altupActiveIndex, o.altupActiveIndex+1, 1)
}

func (o *TextOptions) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

func (o *TextOptions) isLocal(i int) bool {
	return o.slidingWindowPattern[i]
}

func (o TextOptions) applyRotaryPositionEmbeddings(ctx ml.Context, t, p ml.Tensor, base float32) ml.Tensor {
	return nn.RoPE(ctx, t, p, o.headDim(), base, 1./o.ropeScale, rope.WithTypeNeoX())
}

func newTextModel(c fs.Config) *TextModel {
	return &TextModel{
		TextLayers: make([]TextLayer, c.Uint("block_count")),
		TextOptions: TextOptions{
			hiddenLayers:            int(c.Uint("block_count")),
			hiddenSize:              int(c.Uint("embedding_length")),
			hiddenSizePerLayerInput: int(c.Uint("embedding_length_per_layer_input")),
			numHeads:                int(c.Uint("attention.head_count")),
			numKVHeads:              int(c.Uint("attention.head_count_kv")),
			keyLength:               int(c.Uint("attention.key_length")),
			valueLength:             int(c.Uint("attention.value_length")),
			sharedKeyValueLayers:    int(c.Uint("attention.shared_kv_layers")),

			altupActiveIndex: int(c.Uint("altup.active_idx")),
			altupInputs:      int(c.Uint("altup.num_inputs")),

			eps:           c.Float("attention.layer_norm_rms_epsilon", 1e-06),
			ropeBase:      c.Float("rope.freq_base", 1_000_000),
			ropeBaseLocal: c.Float("rope.freq_base_local", 10_000),
			ropeScale:     c.Float("rope.scaling.factor", 1.0),

			slidingWindowPattern:    c.Bools("attention.sliding_window_pattern"),
			activationSparsityScale: c.Floats("activation_sparsity_scale"),
		},
	}
}
