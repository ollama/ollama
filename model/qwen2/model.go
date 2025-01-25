package qwen2

import (
	"fmt"
	"log/slog"
	"math"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Options struct {
	hiddenSize, numHeads, numKVHeads int64
	eps, ropeBase, ropeScale         float32
	ropeDim                          uint32
}

type Model struct {
	model.Base
	model.BytePairEncoding

	TokenEmbedding *nn.Embedding `gguf:"token_embd"`
	Layers         []Layer       `gguf:"blk"`
	OutputNorm     *nn.RMSNorm   `gguf:"output_norm"`
	Output         *nn.Linear    `gguf:"output,alt:token_embd"`

	*Options
}

func New(c ml.Config) (model.Model, error) {
	m := &Model{
		BytePairEncoding: model.BytePairEncoding{
			Pretokenizer: c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
			Vocabulary: &model.Vocabulary{
				Values: c.Strings("tokenizer.ggml.tokens"),
				Types:  c.Uints("tokenizer.ggml.token_type"),
				Merges: c.Strings("tokenizer.ggml.merges"),
				BOS:    c.Uint("tokenizer.ggml.bos_token_id"),
				EOS:    c.Uint("tokenizer.ggml.eos_token_id"),
			},
		},
		Layers: make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize: int64(c.Uint("embedding_length")),
			numHeads:   int64(c.Uint("attention.head_count")),
			numKVHeads: int64(c.Uint("attention.head_count_kv")),
			eps:        c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:   c.Float("rope.freq_base"),
			ropeScale:  c.Float("rope.freq_scale", 1),
			ropeDim:    c.Uint("rope.dimension_count", 64),
		},
	}

	slog.Debug("model configuration",
		"arch", "qwen2",
		"vocab_size", len(c.Strings("tokenizer.ggml.tokens")),
		"n_merges", len(c.Strings("tokenizer.ggml.merges")),
		"n_ctx_train", c.Uint("context_length"),
		"n_embd", m.hiddenSize,
		"n_layer", len(m.Layers),
		"n_head", m.numHeads,
		"n_head_kv", m.numKVHeads,
		"n_rot", m.ropeDim,
		"f_norm_rms_eps", m.eps,
		"rope_freq_base", m.ropeBase,
		"rope_freq_scale", m.ropeScale,
		"bos_token_id", c.Uint("tokenizer.ggml.bos_token_id"),
		"eos_token_id", c.Uint("tokenizer.ggml.eos_token_id"),
	)

	return m, nil
}

type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, inputPositions ml.Tensor, layerIdx int, cache cache.Cache, opts *Options) ml.Tensor {
	batchSize := hiddenState.Dim(1)
	headDim := opts.hiddenSize / opts.numHeads

	q := sa.Query.Forward(ctx, hiddenState)
	ctx.Trace(fmt.Sprintf("model.layers.%d.self_attn.q_proj", layerIdx), q)

	q = q.Reshape(ctx, headDim, opts.numHeads, batchSize)
	q = q.RoPE(ctx, inputPositions, nil, opts.ropeDim, opts.ropeBase, opts.ropeScale)
	ctx.Trace(fmt.Sprintf("model.layers.%d.self_attn.q_proj.rope", layerIdx), q)

	k := sa.Key.Forward(ctx, hiddenState)
	k = k.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	k = k.RoPE(ctx, inputPositions, nil, opts.ropeDim, opts.ropeBase, opts.ropeScale)
	ctx.Trace(fmt.Sprintf("model.layers.%d.self_attn.k_proj.rope", layerIdx), k)

	v := sa.Value.Forward(ctx, hiddenState)
	v = v.Reshape(ctx, headDim, opts.numKVHeads, batchSize)
	ctx.Trace(fmt.Sprintf("model.layers.%d.self_attn.v_proj", layerIdx), v)

	k, v, mask := cache.Put(ctx, k, v)

	q = q.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	k = k.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	v = v.Permute(ctx, 1, 2, 0, 3).Contiguous(ctx)

	kq := k.Mulmat(ctx, q)
	kq = kq.Scale(ctx, 1.0/math.Sqrt(float64(headDim)))
	kq = kq.Add(ctx, mask)
	kq = kq.Softmax(ctx)

	kqv := v.Mulmat(ctx, kq)
	kqv = kqv.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	kqv = kqv.Reshape(ctx, opts.hiddenSize, batchSize)

	output := sa.Output.Forward(ctx, kqv)
	return output
}

type MLP struct {
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	hiddenState = mlp.Gate.Forward(ctx, hiddenState).SILU(ctx).Mul(ctx, mlp.Up.Forward(ctx, hiddenState))
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, layerIdx int, cache cache.Cache, opts *Options) ml.Tensor {
	ctx.Trace(fmt.Sprintf("model.layers.%d.input", layerIdx), hiddenState)
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	ctx.Trace(fmt.Sprintf("model.layers.%d.input_layernorm", layerIdx), hiddenState)

	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, layerIdx, cache, opts)
	ctx.Trace(fmt.Sprintf("model.layers.%d.self_attn.output", layerIdx), hiddenState)

	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState
	ctx.Trace(fmt.Sprintf("model.layers.%d.self_attn.residual", layerIdx), hiddenState)

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	ctx.Trace(fmt.Sprintf("model.layers.%d.post_attention_layernorm", layerIdx), hiddenState)

	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	ctx.Trace(fmt.Sprintf("model.layers.%d.mlp", layerIdx), hiddenState)

	output := hiddenState.Add(ctx, residual)
	ctx.Trace(fmt.Sprintf("model.layers.%d.output", layerIdx), output)

	return output
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	slog.Debug("input tokens", "input_ids", opts.Inputs())
	inputs, err := ctx.FromIntSlice(opts.Inputs(), len(opts.Inputs()))
	if err != nil {
		return nil, err
	}

	positions, err := ctx.FromIntSlice(opts.Positions(), len(opts.Positions()))
	if err != nil {
		return nil, err
	}

	hiddenState := m.TokenEmbedding.Forward(ctx, inputs)
	ctx.Trace("model.embed_tokens", hiddenState)

	for i, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, positions, i, opts.Cache.Sub(i), m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	ctx.Trace("model.norm", hiddenState)

	hiddenState = m.Output.Forward(ctx, hiddenState)
	ctx.Trace("model.output", hiddenState)

	outputs, err := ctx.FromIntSlice(opts.Outputs(), len(opts.Outputs()))
	if err != nil {
		return nil, err
	}

	return hiddenState.Rows(ctx, outputs), nil
}

func init() {
	model.Register("qwen2", New)
}
