package gptoss

import (
	"cmp"
	"math"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/ml/nn/rope"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
)

type Transformer struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding    *nn.Embedding      `gguf:"token_embd"`
	TransformerBlocks []TransformerBlock `gguf:"blk"`
	OutputNorm        *nn.RMSNorm        `gguf:"output_norm"`
	Output            *nn.Linear         `gguf:"output,alt:token_embd"`

	Options
}

// Forward implements model.Model.
func (m *Transformer) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	for i, block := range m.TransformerBlocks {
		m.Cache.SetLayer(i)
		if c, ok := m.Cache.(*kvcache.WrapperCache); ok {
			// Even layers are sliding window attention.
			c.SetLayerType(i % 2)
		}

		var outputs ml.Tensor
		if i == len(m.TransformerBlocks)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = block.Forward(ctx, hiddenStates, positions, outputs, m.Cache, &m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	return m.Output.Forward(ctx, hiddenStates), nil
}

func (m *Transformer) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

type Options struct {
	hiddenSize,
	numHeads,
	numKVHeads,
	keyLength,
	valueLength,
	numExperts,
	numExpertsUsed,
	originalContextLength int

	eps,
	ropeBase,
	ropeScale float32
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, states, positions, o.headDim(), o.ropeBase, 1./o.ropeScale,
		rope.WithTypeNeoX(),
		rope.WithOriginalContextLength(o.originalContextLength),
		rope.WithExtrapolationFactor(1.),
	// NOTE: ggml sets this implicitly so there's no need to set it here
	// rope.WithAttentionFactor(0.1*float32(math.Log(float64(o.ropeScale))) + 1.0),
	)
}

func (o Options) headDim() int {
	return cmp.Or(o.keyLength, o.valueLength, o.hiddenSize/o.numHeads)
}

type TransformerBlock struct {
	Attention *AttentionBlock
	MLP       *MLPBlock
}

func (d *TransformerBlock) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	hiddenStates = d.Attention.Forward(ctx, hiddenStates, positions, cache, opts)
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
	}

	hiddenStates = d.MLP.Forward(ctx, hiddenStates, opts)
	return hiddenStates
}

type AttentionBlock struct {
	Norm *nn.RMSNorm `gguf:"attn_norm"`

	QKV *nn.Linear `gguf:"attn_qkv"`

	Query *nn.Linear `gguf:"attn_q"`
	Key   *nn.Linear `gguf:"attn_k"`
	Value *nn.Linear `gguf:"attn_v"`

	Output *nn.Linear `gguf:"attn_out,alt:attn_output"`
	Sinks  ml.Tensor  `gguf:"attn_sinks,alt:attn_sinks.weight"`
}

func (attn *AttentionBlock) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)

	residual := hiddenStates
	hiddenStates = attn.Norm.Forward(ctx, hiddenStates, opts.eps)

	var query, key, value ml.Tensor
	if attn.QKV != nil {
		qkv := attn.QKV.Forward(ctx, hiddenStates)
		qkv = qkv.Reshape(ctx, opts.headDim(), -1, batchSize)
		chunks := qkv.ChunkSections(ctx, 1, opts.numHeads, opts.numKVHeads, opts.numKVHeads)
		query, key, value = chunks[0], chunks[1], chunks[2]
	} else {
		query = attn.Query.Forward(ctx, hiddenStates)
		query = query.Reshape(ctx, opts.headDim(), opts.numHeads, batchSize)

		key = attn.Key.Forward(ctx, hiddenStates)
		key = key.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)

		value = attn.Value.Forward(ctx, hiddenStates)
		value = value.Reshape(ctx, opts.headDim(), opts.numKVHeads, batchSize)
	}

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.AttentionWithSinks(ctx, query, key, value, attn.Sinks, 1/math.Sqrt(float64(opts.headDim())), cache)
	attention = attention.Reshape(ctx, attention.Dim(0)*attention.Dim(1), batchSize)
	return attn.Output.Forward(ctx, attention).Add(ctx, residual)
}

type MLPBlock struct {
	Norm   *nn.RMSNorm `gguf:"ffn_norm,alt:post_attention_norm"`
	Router *nn.Linear  `gguf:"ffn_gate_inp"`

	GateUp *nn.LinearBatch `gguf:"ffn_gate_up_exps"`

	Gate *nn.LinearBatch `gguf:"ffn_gate_exps"`
	Up   *nn.LinearBatch `gguf:"ffn_up_exps"`

	Down *nn.LinearBatch `gguf:"ffn_down_exps"`
}

func (mlp *MLPBlock) Forward(ctx ml.Context, hiddenStates ml.Tensor, opts *Options) ml.Tensor {
	hiddenDim, sequenceLength, batchSize := hiddenStates.Dim(0), hiddenStates.Dim(1), hiddenStates.Dim(2)

	residual := hiddenStates
	hiddenStates = mlp.Norm.Forward(ctx, hiddenStates, opts.eps)

	hiddenStates = hiddenStates.Reshape(ctx, hiddenDim, sequenceLength*batchSize)
	routingWeights := mlp.Router.Forward(ctx, hiddenStates)

	selectedExperts := routingWeights.TopK(ctx, opts.numExpertsUsed)
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExperts, sequenceLength*batchSize).Rows(ctx, selectedExperts)
	routingWeights = routingWeights.Reshape(ctx, opts.numExpertsUsed, sequenceLength*batchSize).Softmax(ctx)
	routingWeights = routingWeights.Reshape(ctx, 1, opts.numExpertsUsed, sequenceLength*batchSize)

	hiddenStates = hiddenStates.Reshape(ctx, hiddenStates.Dim(0), 1, hiddenStates.Dim(1))

	var gate, up ml.Tensor
	if mlp.GateUp != nil {
		hiddenStates = mlp.GateUp.Forward(ctx, hiddenStates, selectedExperts)
		gate = hiddenStates.Slice(ctx, 0, 0, hiddenStates.Dim(0), 2)
		up = hiddenStates.Slice(ctx, 0, 1, hiddenStates.Dim(0), 2)
	} else {
		gate = mlp.Gate.Forward(ctx, hiddenStates, selectedExperts)
		up = mlp.Up.Forward(ctx, hiddenStates, selectedExperts)
	}

	hiddenStates = gate.SILUAlphaLimit(ctx, up, 1.702, 7)

	experts := mlp.Down.Forward(ctx, hiddenStates, selectedExperts)
	experts = experts.Mul(ctx, routingWeights)

	nextStates := experts.View(ctx, 0, experts.Dim(0), experts.Stride(2), experts.Dim(2))
	for i := 1; i < opts.numExpertsUsed; i++ {
		nextStates = nextStates.Add(ctx, experts.View(ctx, i*experts.Stride(1), experts.Dim(0), experts.Stride(2), experts.Dim(2)))
	}

	return nextStates.Add(ctx, residual)
}

func New(c fs.Config) (model.Model, error) {
	m := Transformer{
		TransformerBlocks: make([]TransformerBlock, c.Uint("block_count")),
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", false),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			strings.Join([]string{
				`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
				`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
				`\p{N}{1,3}`,
				` ?[^\s\p{L}\p{N}]+[\r\n/]*`,
				`\s*[\r\n]+`,
				`\s+(?!\S)`,
				`\s+`,
			}, "|"),
		),
		Options: Options{
			hiddenSize:            int(c.Uint("embedding_length")),
			numHeads:              int(c.Uint("attention.head_count")),
			numKVHeads:            int(c.Uint("attention.head_count_kv")),
			keyLength:             int(c.Uint("attention.key_length")),
			valueLength:           int(c.Uint("attention.value_length")),
			numExperts:            int(c.Uint("expert_count")),
			numExpertsUsed:        int(c.Uint("expert_used_count")),
			eps:                   c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:              c.Float("rope.freq_base"),
			ropeScale:             c.Float("rope.scaling.factor", 1.),
			originalContextLength: int(c.Uint("rope.scaling.original_context_length")),
		},
	}

	m.Cache = kvcache.NewWrapperCache(
		kvcache.NewSWAMemCache(int32(c.Uint("attention.sliding_window")), 4096, m.Shift),
		kvcache.NewCausalCache(m.Shift),
	)
	return &m, nil
}

func init() {
	model.Register("gptoss", New)
	model.Register("gpt-oss", New)
}
