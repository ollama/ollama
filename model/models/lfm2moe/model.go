package lfm2moe

import (
	"cmp"
	"fmt"
	"math"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
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

	numExperts, numExpertsUsed int
	numDenseLayers             int
	routedScalingFactor        float32
	normTopKProb              bool
	useExpertBias             bool
}

func (o Options) headDimValue() int {
	// Head dim is shared across layers; fall back to first attention layer head count.
	if len(o.numHeadsByLayer) > 0 && o.numHeadsByLayer[0] > 0 {
		return cmp.Or(o.headDim, o.hiddenSize/o.numHeadsByLayer[0])
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
	model.TextProcessor

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm,alt:token_embd_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	Options
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

type shortConvKernel struct {
	Weight ml.Tensor `gguf:"weight"`
}

// ShortConv implements the LFM2 short-convolution block (GGML_OP_SSM_CONV) with a recurrent
// state stored in the HybridCache.
type ShortConv struct {
	Conv    *shortConvKernel `gguf:"shortconv.conv"`
	InProj  *nn.Linear       `gguf:"shortconv.in_proj"`
	OutProj *nn.Linear       `gguf:"shortconv.out_proj"`
}

func (sc *ShortConv) Forward(ctx ml.Context, hiddenStates ml.Tensor, _ ml.Tensor, cache *HybridCache, layer int, opts *Options) ml.Tensor {
	nSeqs := cache.numSeqs()
	seqTokens := cache.seqTokens()
	hiddenSize := hiddenStates.Dim(0)
	if nSeqs <= 0 || seqTokens <= 0 || hiddenStates.Dim(1) != nSeqs*seqTokens {
		panic("lfm2moe: unsupported batch layout for shortconv")
	}

	bcx := sc.InProj.Forward(ctx, hiddenStates).Reshape(ctx, 3*hiddenSize, seqTokens, nSeqs)

	elementSize := bcx.Stride(0)
	b := bcx.View(ctx, 0*hiddenSize*elementSize, hiddenSize, bcx.Stride(1), seqTokens, bcx.Stride(2), nSeqs)
	c := bcx.View(ctx, 1*hiddenSize*elementSize, hiddenSize, bcx.Stride(1), seqTokens, bcx.Stride(2), nSeqs)
	x := bcx.View(ctx, 2*hiddenSize*elementSize, hiddenSize, bcx.Stride(1), seqTokens, bcx.Stride(2), nSeqs)

	bx := b.Mul(ctx, x).Permute(ctx, 1, 0, 2, 3)

	state := cache.ConvState(ctx, layer)
	sx := state.Concat(ctx, bx, 0)

	convOut := sx.SSMConv(ctx, sc.Conv.Weight)
	y := c.Mul(ctx, convOut)

	dConv := sx.Dim(0) - seqTokens
	cache.UpdateConvState(ctx, layer, sx.Slice(ctx, 0, sx.Dim(0)-dConv, sx.Dim(0), 1))

	return sc.OutProj.Forward(ctx, y.Reshape(ctx, hiddenSize, seqTokens*nSeqs))
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

type SparseMoeBlock struct {
	Router       *nn.Linear      `gguf:"ffn_gate_inp,alt:gate_inp"`
	Gate         *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up           *nn.LinearBatch `gguf:"ffn_up_exps"`
	Down         *nn.LinearBatch `gguf:"ffn_down_exps"`
	ExpProbsBias ml.Tensor       `gguf:"exp_probs_b.bias,alt:exp_probs_b"`
}

func (moe *SparseMoeBlock) routeTokensToExperts(ctx ml.Context, routerLogits ml.Tensor, opts *Options) (ml.Tensor, ml.Tensor) {
	routingWeights := routerLogits.Sigmoid(ctx)

	scoresForRouting := routingWeights
	if moe.ExpProbsBias != nil && opts.useExpertBias {
		scoresForRouting = routingWeights.Add(ctx, moe.ExpProbsBias)
	}

	selectedExperts := scoresForRouting.TopK(ctx, opts.numExpertsUsed)
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, routerLogits.Dim(1)).Rows(ctx, selectedExperts)

	if opts.normTopKProb {
		routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, routerLogits.Dim(1))
		sumRows := routingWeights.SumRows(ctx)
		epsilon := ctx.Zeros(ml.DTypeF32, 1, routerLogits.Dim(1)).Scale(ctx, 1e-6)
		routingWeights = routingWeights.Div(ctx, sumRows.Add(ctx, epsilon))
		routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, routerLogits.Dim(1))
	}

	routingWeights = routingWeights.Scale(ctx, float64(opts.routedScalingFactor))
	return selectedExperts, routingWeights
}

func (moe *SparseMoeBlock) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim := hiddenState.Dim(0)
	batchSize := hiddenState.Dim(1)

	routerLogits := moe.Router.Forward(ctx, hiddenState)
	selectedExperts, routingWeights := moe.routeTokensToExperts(ctx, routerLogits, opts)

	hiddenState = hiddenState.Reshape(ctx, hiddenDim, 1, batchSize)
	hiddenState = hiddenState.Repeat(ctx, 1, opts.numExpertsUsed)

	gate := moe.Gate.Forward(ctx, hiddenState, selectedExperts)
	up := moe.Up.Forward(ctx, hiddenState, selectedExperts)
	experts := moe.Down.Forward(ctx, gate.SILU(ctx, up), selectedExperts)
	experts = experts.Mul(ctx, routingWeights)

	hiddenState = experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		hiddenState = hiddenState.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return hiddenState
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Operator      Operator
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
	MoE           *SparseMoeBlock
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
	if l.MLP != nil {
		hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	} else {
		hiddenState = l.MoE.Forward(ctx, hiddenState, opts)
	}
	return hiddenState.Add(ctx, residual)
}

func New(c fs.Config) (model.Model, error) {
	expertCount := c.Uint("expert_count")
	if expertCount == 0 {
		return nil, model.ErrUnsupportedModel
	}

	// Tokenizer
	vocabulary := model.Vocabulary{
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

	var processor model.TextProcessor
	switch c.String("tokenizer.ggml.model") {
	case "gpt2":
		// LFM2 uses a llama3-style BPE pretokenizer.
		var pretokenizers []string
		switch c.String("tokenizer.ggml.pre") {
		case "lfm2", "llama3", "llama-v3", "llama-bpe":
			pretokenizers = []string{
				"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			}
		case "qwen2":
			pretokenizers = []string{
				"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			}
		case "refact":
			pretokenizers = []string{
				`\p{N}`,
				`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
			}
		case "tekken":
			pretokenizers = []string{
				"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			}
		case "default":
			// no-op use the default bpe pretokenizer
		default:
			// use a llama-style pretokenizer
			pretokenizers = []string{
				"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
			}
		}
		processor = model.NewBytePairEncoding(&vocabulary, pretokenizers...)
	case "llama":
		return nil, fmt.Errorf("unsupported tokenizer: llama")
	default:
		return nil, model.ErrUnsupportedTokenizer
	}

	if strings.HasPrefix(c.String("general.name"), "Qwen2-beta") {
		return nil, fmt.Errorf("unsupported model: %s", c.String("general.name"))
	}

	numDenseLayers := int(c.Uint("leading_dense_block_count", 0))
	numExpertsPerTok := int(c.Uint("expert_used_count", 1))
	routedScalingFactor := c.Float("expert.routed_scaling_factor", 1.0)
	normTopKProb := c.Bool("expert.norm_topk_prob", false)
	useExpertBias := c.Bool("expert.use_expert_bias", false)

	m := Model{
		TextProcessor: processor,
		Layers:        make([]Layer, c.Uint("block_count")),
		Options: Options{
			hiddenSize:            int(c.Uint("embedding_length")),
			headDim:               int(c.Uint("attention.key_length")),
			ropeDim:               int(c.Uint("rope.dimension_count")),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeType:              c.String("rope.scaling.type"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             c.Float("rope.scaling.factor", 1),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
			numExperts:            int(expertCount),
			numExpertsUsed:        numExpertsPerTok,
			numDenseLayers:        numDenseLayers,
			routedScalingFactor:   routedScalingFactor,
			normTopKProb:          normTopKProb,
			useExpertBias:         useExpertBias,
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

		if i < numDenseLayers {
			m.Layers[i].MLP = &MLP{}
		} else {
			m.Layers[i].MoE = &SparseMoeBlock{}
		}
	}

	lCache := int(c.Uint("shortconv.l_cache"))
	dConv := max(0, lCache-1)
	m.Cache = NewHybridCache(m.Shift, m.hiddenSize, dConv)
	return &m, nil
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
	model.Register("lfm2moe", New)
}
