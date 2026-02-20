package lfm2

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/tokenizer"
)

type Options struct {
	hiddenSize       int
	headDim, ropeDim int

	eps, ropeBase, ropeScale float32

	ropeType              string
	originalContextLength int

	// per-layer head counts (LFM2 alternates attention and recurrent layers)
	numHeadsByLayer   []int
	numKVHeadsByLayer []int
}

func (o Options) headDimValue() int {
	// Head dim is shared across layers; fall back to first attention layer head count.
	for _, h := range o.numHeadsByLayer {
		if h > 0 {
			return cmp.Or(o.headDim, o.hiddenSize/h)
		}
	}
	return cmp.Or(o.headDim, o.hiddenSize)
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	opts := []func(*rope.Options){rope.WithTypeNeoX()}
	if o.ropeType == "yarn" {
		attnFactor := float32(1.0 / (1.0 + 0.1*math.Log(float64(o.ropeScale))))
		opts = append(opts,
			rope.WithOriginalContextLength(o.originalContextLength),
			rope.WithExtrapolationFactor(1.),
			rope.WithAttentionFactor(attnFactor),
		)
	}

	headCount := 1
	for _, h := range o.numHeadsByLayer {
		if h > 0 {
			headCount = h
			break
		}
	}
	return nn.RoPE(ctx, states, positions, cmp.Or(o.ropeDim, o.headDim, o.hiddenSize/headCount), o.ropeBase, 1./o.ropeScale, opts...)
}

type Model struct {
	model.Base
	tokenizer.Tokenizer

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm,alt:token_embd_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Options
}

func New(c fs.Config) (model.Model, error) {
	if c.Uint("expert_count") > 0 {
		return nil, model.ErrUnsupportedModel
	}

	if c.String("tokenizer.ggml.model") != "gpt2" {
		return nil, model.ErrUnsupportedTokenizer
	}

	vocabulary := tokenizer.Vocabulary{
		Values: c.Strings("tokenizer.ggml.tokens"),
		Scores: c.Floats("tokenizer.ggml.scores"),
		Types:  c.Ints("tokenizer.ggml.token_type"),
		Merges: c.Strings("tokenizer.ggml.merges"),
		AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
		BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
		AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
		EOS: append(
			[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
			c.Ints("tokenizer.ggml.eos_token_ids")...,
		),
	}

	var pretokenizers []string
	switch c.String("tokenizer.ggml.pre") {
	case "default":
		// use default BPE pretokenizer
	default:
		// llama-bpe style (default for LFM2)
		pretokenizers = []string{
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		}
	}

	m := Model{
		Tokenizer: tokenizer.NewBytePairEncoding(&vocabulary, pretokenizers...),
		Layers:    make([]Layer, c.Uint("block_count")),
		Options: Options{
			hiddenSize:            int(c.Uint("embedding_length")),
			headDim:               int(c.Uint("attention.key_length")),
			ropeDim:               int(c.Uint("rope.dimension_count")),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeType:              c.String("rope.scaling.type"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             c.Float("rope.scaling.factor", 1),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
		},
	}

	type headCounts interface {
		HeadCount() []uint64
		HeadCountKV() []uint64
	}
	hc, ok := c.(headCounts)
	if !ok {
		return nil, model.ErrUnsupportedModel
	}

	headCount := hc.HeadCount()
	headCountKV := hc.HeadCountKV()

	m.numHeadsByLayer = make([]int, len(m.Layers))
	m.numKVHeadsByLayer = make([]int, len(m.Layers))
	for i := range m.Layers {
		m.numHeadsByLayer[i] = int(headCount[i])
		m.numKVHeadsByLayer[i] = int(headCountKV[i])

		if m.numKVHeadsByLayer[i] == 0 {
			m.Layers[i].Operator = &ShortConv{}
		} else {
			m.Layers[i].Operator = &Attention{}
		}
	}

	lCache := int(c.Uint("shortconv.l_cache"))
	dConv := max(0, lCache-1)
	m.Cache = NewHybridCache(m.Shift, m.hiddenSize, dConv)
	return &m, nil
}

type Operator interface {
	Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, layer int, opts *Options) ml.Tensor
}

type Attention struct {
	Query     *nn.Linear  `gguf:"attn_q"`
	QueryNorm *nn.RMSNorm `gguf:"attn_q_norm"`
	Key       *nn.Linear  `gguf:"attn_k"`
	KeyNorm   *nn.RMSNorm `gguf:"attn_k_norm"`
	Value     *nn.Linear  `gguf:"attn_v"`
	Output    *nn.Linear  `gguf:"attn_output,alt:attn_out"`
}

func (sa *Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache *HybridCache, layer int, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)
	headDim := opts.headDimValue()
	numHeads := opts.numHeadsByLayer[layer]
	numKVHeads := opts.numKVHeadsByLayer[layer]

	query := sa.Query.Forward(ctx, hiddenStates)
	key := sa.Key.Forward(ctx, hiddenStates)
	value := sa.Value.Forward(ctx, hiddenStates)

	query = query.Reshape(ctx, headDim, numHeads, batchSize)
	key = key.Reshape(ctx, headDim, numKVHeads, batchSize)
	value = value.Reshape(ctx, headDim, numKVHeads, batchSize)

	query = sa.QueryNorm.Forward(ctx, query, opts.eps)
	key = sa.KeyNorm.Forward(ctx, key, opts.eps)

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1./math.Sqrt(float64(headDim)), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return sa.Output.Forward(ctx, attention)
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Operator      Operator
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, layer int, hiddenState, positions, outputs ml.Tensor, cache *HybridCache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.Operator.Forward(ctx, hiddenState, positions, cache, layer, opts)

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

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

func (m *Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenState := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenState = layer.Forward(ctx, i, hiddenState, positions, outputs, m.Cache.(*HybridCache), &m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	return m.Output.Forward(ctx, hiddenState), nil
}

func init() {
	model.Register("lfm2", New)
}
