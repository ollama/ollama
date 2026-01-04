package mistral3

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

type TextOptions struct {
	hiddenSize, numHeads, numKVHeads int
	headDim, ropeDim                 int
	eps, ropeBase, ropeScale         float32
	ropeOrigPosEmbeddings            int
	ropeScalingBeta                  float32
	ropeType                         string
	ropeExtrapolation                float32
	ropeBetaFast                     float32
	ropeBetaSlow                     float32
	ropeMscale                       float32
	ropeMscaleAllDim                 float32
}

func (o TextOptions) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	var ropeOpts []func(*rope.Options)
	if o.ropeType == "yarn" {
		if o.ropeMscale != 0 && o.ropeMscaleAllDim != 0 {
			ropeOpts = append(ropeOpts, rope.WithAttentionFactor(1.0/float32(0.1*math.Log(float64(o.ropeScale))+1.0)))
		}

		ropeOpts = append(ropeOpts,
			rope.WithOriginalContextLength(o.ropeOrigPosEmbeddings),
			rope.WithExtrapolationFactor(o.ropeExtrapolation),
			rope.WithBetaFast(o.ropeBetaFast),
			rope.WithBetaSlow(o.ropeBetaSlow),
		)
	}

	return nn.RoPE(ctx, states, positions, o.ropeDim, o.ropeBase, 1./o.ropeScale, ropeOpts...)
}

type TextModel struct {
	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*TextOptions
}

type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs, positionsScale ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := cmp.Or(opts.headDim, opts.hiddenSize/opts.numHeads)

	q := sa.Query.Forward(ctx, hiddenState)
	q = q.Reshape(ctx, headDim, opts.numHeads, batchSize)
	q = opts.applyRotaryPositionEmbeddings(ctx, q, positionIDs)

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	k = opts.applyRotaryPositionEmbeddings(ctx, k, positionIDs)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	if opts.ropeOrigPosEmbeddings > 0 {
		q = q.Mul(ctx, positionsScale)
	}

	kqv := nn.Attention(ctx, q, k, v, 1.0/math.Sqrt(float64(headDim)), cache)
	kqv = kqv.Reshape(ctx, headDim*opts.numHeads, batchSize)
	return sa.Output.Forward(ctx, kqv)
}

func (m *TextModel) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *TextOptions) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs, positionsScale, outputs ml.Tensor, cache kvcache.Cache, opts *TextOptions) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, positionsScale, cache, opts)

	// In the final layer (outputs != nil), optimize by pruning to just the token positions
	// we need logits for.
	if outputs != nil {
		hiddenState = hiddenState.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

func (m *TextModel) Forward(ctx ml.Context, inputs, positions, positionsScale, outputs ml.Tensor, batch input.Batch, cache kvcache.Cache) ml.Tensor {
	hiddenState := m.TokenEmbedding.Forward(ctx, inputs).Duplicate(ctx)

	// image embeddings
	for _, image := range batch.Multimodal {
		imageFeature := image.Multimodal[0].Tensor
		ctx.Forward(imageFeature.Copy(ctx, hiddenState.View(ctx, image.Index*hiddenState.Stride(1), imageFeature.Dim(0)*imageFeature.Dim(1))))
	}

	for i, layer := range m.Layers {
		cache.SetLayer(i)

		var lastLayerOutputs ml.Tensor
		if i == len(m.Layers)-1 {
			lastLayerOutputs = outputs
		}

		hiddenState = layer.Forward(ctx, hiddenState, positions, positionsScale, lastLayerOutputs, cache, m.TextOptions)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return m.Output.Forward(ctx, hiddenState)
}

func (m *TextModel) getScale(ctx ml.Context, positions []int32) ml.Tensor {
	posScale := make([]float32, len(positions))
	for n, pos := range positions {
		interval := math.Floor(float64(pos) / float64(m.ropeOrigPosEmbeddings))
		posScale[n] = float32(1.0 + float64(m.ropeScalingBeta)*math.Log(1.0+interval))
	}
	return ctx.Input().FromFloats(posScale, 1, 1, len(posScale))
}

func newTextModel(c fs.Config) *TextModel {
	return &TextModel{
		Layers: make([]Layer, c.Uint("block_count")),
		TextOptions: &TextOptions{
			hiddenSize:            int(c.Uint("embedding_length")),
			numHeads:              int(c.Uint("attention.head_count")),
			numKVHeads:            int(c.Uint("attention.head_count_kv")),
			headDim:               int(c.Uint("attention.key_length")),
			ropeDim:               int(c.Uint("rope.dimension_count")),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             c.Float("rope.scaling.factor", 1.0),
			ropeOrigPosEmbeddings: int(c.Uint("rope.scaling.original_context_length")),
			ropeScalingBeta:       c.Float("rope.scaling_beta", 0.1),
			ropeBetaFast:          c.Float("rope.scaling.beta_fast", 32.0),
			ropeBetaSlow:          c.Float("rope.scaling.beta_slow", 1.0),
			ropeType:              c.String("rope.scaling.type"),
			ropeMscale:            c.Float("rope.scaling.mscale"),
			ropeMscaleAllDim:      c.Float("rope.scaling.mscale_all_dim"),
			ropeExtrapolation:     c.Float("rope.scaling.extrapolation_factor", 1),
		},
	}
}
