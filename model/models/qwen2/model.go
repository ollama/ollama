package qwen2

import (
	"cmp"
	"fmt"
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

type Options struct {
	hiddenSize, numHeads, numKVHeads int
	headDim, ropeDim                 int
	eps, ropeBase, ropeScale         float32
}

func (o Options) applyRotaryPositionEmbeddings(ctx ml.Context, states, positions ml.Tensor) ml.Tensor {
	return nn.RoPE(ctx, states, positions, cmp.Or(o.ropeDim, o.headDim, o.hiddenSize/o.numHeads), o.ropeBase, 1./o.ropeScale, rope.WithTypeNeoX())
}

type Attention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (attn Attention) Forward(ctx ml.Context, hiddenStates, positions ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenStates.Dim(1)
	headDim := cmp.Or(opts.headDim, opts.hiddenSize/opts.numHeads)

	query := attn.Query.Forward(ctx, hiddenStates)
	query = query.Reshape(ctx, headDim, opts.numHeads, batchSize)

	key := attn.Key.Forward(ctx, hiddenStates)
	key = key.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	value := attn.Value.Forward(ctx, hiddenStates)
	value = value.Reshape(ctx, headDim, opts.numKVHeads, batchSize)

	query = opts.applyRotaryPositionEmbeddings(ctx, query, positions)
	key = opts.applyRotaryPositionEmbeddings(ctx, key, positions)

	attention := nn.Attention(ctx, query, key, value, 1.0/math.Sqrt(float64(headDim)), cache)
	attention = attention.Reshape(ctx, headDim*opts.numHeads, batchSize)

	return attn.Output.Forward(ctx, attention)
}

type MLP struct {
	Gate *nn.Linear `gguf:"ffn_gate"`
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
}

func (mlp MLP) Forward(ctx ml.Context, hiddenStates ml.Tensor) ml.Tensor {
	hiddenStates = mlp.Gate.Forward(ctx, hiddenStates).SILU(ctx, mlp.Up.Forward(ctx, hiddenStates))
	return mlp.Down.Forward(ctx, hiddenStates)
}

type DecoderLayer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	Attention     *Attention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (d DecoderLayer) Forward(ctx ml.Context, hiddenStates, positions, outputs ml.Tensor, cache kvcache.Cache, opts *Options) ml.Tensor {
	residual := hiddenStates

	hiddenStates = d.AttentionNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.Attention.Forward(ctx, hiddenStates, positions, cache, opts)
	if outputs != nil {
		hiddenStates = hiddenStates.Rows(ctx, outputs)
		residual = residual.Rows(ctx, outputs)
	}

	hiddenStates = hiddenStates.Add(ctx, residual)
	residual = hiddenStates

	hiddenStates = d.MLPNorm.Forward(ctx, hiddenStates, opts.eps)
	hiddenStates = d.MLP.Forward(ctx, hiddenStates)
	return hiddenStates.Add(ctx, residual)
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding  `gguf:"token_embd"`
	Layers         []DecoderLayer `gguf:"blk"`
	OutputNorm     *nn.RMSNorm    `gguf:"output_norm"`
	Output         *nn.Linear     `gguf:"output,alt:token_embd"`

	Options
}

// Forward implements model.Model.
func (m Model) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	positions := ctx.Input().FromInts(batch.Positions, len(batch.Positions))

	hiddenStates := m.TokenEmbedding.Forward(ctx, batch.Inputs)

	for i, layer := range m.Layers {
		m.Cache.SetLayer(i)

		var outputs ml.Tensor
		if i == len(m.Layers)-1 {
			outputs = batch.Outputs
		}

		hiddenStates = layer.Forward(ctx, hiddenStates, positions, outputs, m.Cache, &m.Options)
	}

	hiddenStates = m.OutputNorm.Forward(ctx, hiddenStates, m.eps)
	hiddenStates = m.Output.Forward(ctx, hiddenStates)
	return hiddenStates, nil
}

func (m Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return m.applyRotaryPositionEmbeddings(ctx, key, shift), nil
}

func New(c fs.Config) (model.Model, error) {
	// This model currently only supports the gpt2 tokenizer
	if c.String("tokenizer.ggml.model") == "llama" {
		return nil, fmt.Errorf("unsupported tokenizer: llama")
	}
	// detect library/qwen model(s) which are incompatible
	if strings.HasPrefix(c.String("general.name"), "Qwen2-beta") {
		return nil, fmt.Errorf("unsupported model: %s", c.String("general.name"))
	}
	m := Model{
		Layers: make([]DecoderLayer, c.Uint("block_count")),
		BytePairEncoding: model.NewBytePairEncoding(
			&model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Ints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				AddBOS: c.Bool("tokenizer.ggml.add_bos_token", true),
				BOS:    []int32{int32(c.Uint("tokenizer.ggml.bos_token_id"))},
				AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false),
				EOS: append(
					[]int32{int32(c.Uint("tokenizer.ggml.eos_token_id"))},
					c.Ints("tokenizer.ggml.eos_token_ids")...,
				),
			},
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		),
		Options: Options{
			hiddenSize: int(c.Uint("embedding_length")),
			numHeads:   int(c.Uint("attention.head_count")),
			numKVHeads: int(c.Uint("attention.head_count_kv")),
			headDim:    int(c.Uint("attention.key_length")),
			ropeDim:    int(c.Uint("rope.dimension_count")),
			ropeBase:   c.Float("rope.freq_base"),
			ropeScale:  c.Float("rope.scaling.factor", 1),
			eps:        c.Float("attention.layer_norm_rms_epsilon"),
		},
	}

	m.Cache = kvcache.NewCausalCache(m.Shift)
	return &m, nil
}

func init() {
	model.Register("qwen2", New)
}
