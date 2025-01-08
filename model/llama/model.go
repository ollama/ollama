package llama

import (
	"math"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Options struct {
	RopeFactors                      ml.Tensor `ggml:"rope_freqs.weight"`
	hiddenSize, numHeads, numKVHeads int64
	eps, ropeBase, ropeScale         float32
	ropeDim                          uint32
}

type Model struct {
	model.Base

	TextProcessor

	TokenEmbedding *nn.Embedding `ggml:"token_embd"`
	Layers         []Layer       `ggml:"blk"`
	OutputNorm     *nn.RMSNorm   `ggml:"output_norm"`
	Output         *nn.Linear    `ggml:"output"`

	*Options
}

func New(c ml.Config) (model.Model, error) {
	r := &Model{
		TextProcessor: newTextProcessor(c),
		Layers:        make([]Layer, c.Uint("block_count")),
		Options: &Options{
			hiddenSize: int64(c.Uint("embedding_length")),
			numHeads:   int64(c.Uint("attention.head_count")),
			numKVHeads: int64(c.Uint("attention.head_count_kv")),
			eps:        c.Float("attention.layer_norm_rms_epsilon"),
			ropeBase:   c.Float("rope.freq_base"),
			ropeScale:  c.Float("rope.freq_scale", 1),
			ropeDim:    c.Uint("rope.dimension_count"),
		},
	}
	return r, nil
}

type SelfAttention struct {
	Query  *nn.Linear `ggml:"attn_q"`
	Key    *nn.Linear `ggml:"attn_k"`
	Value  *nn.Linear `ggml:"attn_v"`
	Output *nn.Linear `ggml:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState ml.Tensor, offset int32, cache model.Cache, opts *Options) ml.Tensor {
	// Ref: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py
	shape := hiddenState.Shape()
	B := shape[0]
	L := shape[1]
	n_heads := opts.numHeads
	n_kv_heads := opts.numKVHeads

	head_dim := opts.hiddenSize / opts.numHeads
	scale := math.Pow(float64(head_dim), -0.5)

	queries := sa.Query.Forward(ctx, hiddenState)
	keys := sa.Key.Forward(ctx, hiddenState)
	values := sa.Value.Forward(ctx, hiddenState)

	queries = queries.Reshape(ctx, B, L, n_heads, -1).Permute(ctx, 0, 2, 1, 3)
	keys = keys.Reshape(ctx, B, L, n_kv_heads, -1).Permute(ctx, 0, 2, 1, 3)
	values = values.Reshape(ctx, B, L, n_kv_heads, -1).Permute(ctx, 0, 2, 1, 3)

	queries = sa.Rope(ctx, queries, offset, opts)
	keys = sa.Rope(ctx, keys, offset, opts)

	// TODO - when this comes back, the input should be truncated to just the latest token
	// keys, values = cache.Put(ctx, keys, values, cache.Options)

	// TODO - some sort of discovery mechanism to know if the backend supports the fast routine
	var output ml.Tensor
	if false {
		// Begin scaled dot product attention
		// Ref: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L196
		n_rep := int(n_heads / n_kv_heads)
		keys = keys.Permute(ctx, 0, 2, 1, 3)
		values = values.Permute(ctx, 0, 2, 1, 3)
		queries = queries.Permute(ctx, 0, 2, 1, 3)
		// repeat k/v heads if n_kv_heads < n_heads
		// equiv to torch.repeat_interleave(x, dim=2, repeats=n_rep)
		keys = keys.Repeat(ctx, n_rep, 2).Contiguous(ctx)
		values = values.Repeat(ctx, n_rep, 2).Contiguous(ctx)

		queries = queries.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx) // (bs, n_heads, L, head_dim)
		keys = keys.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)       // (bs, n_heads, cache_len + L, head_dim)
		values = values.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)   // (bs, n_heads, cache_len + L, head_dim)

		kp := keys.Permute(ctx, 0, 1, 3, 2)
		scores := kp.Mulmat(ctx, queries).Scale(ctx, 1.0/math.Sqrt(float64(head_dim)))
		// TODO mask here

		scores = scores.Softmax(ctx) // Without axis=-1 this starts to drift
		output = values.Mulmat(ctx, scores)
	} else {
		output = ctx.FastScaledDotProductAttention(queries, keys, values, float32(scale), nil)
	}

	output = output.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	output = output.Reshape(ctx, B, L, -1)
	output = sa.Output.Forward(ctx, output)
	return output
}

func (sa *SelfAttention) Rope(ctx ml.Context, x ml.Tensor, offset int32, opts *Options) ml.Tensor {
	var once sync.Once
	var _freqs ml.Tensor
	dims := opts.ropeDim
	onceBody := func() {
		// Reference: https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/rope_utils.py#L9

		base := opts.ropeBase // aka rope_scale
		if base == 0 {
			base = 10000.0
		}
		low_freq_factor := opts.ropeScale // ???
		high_freq_factor := float32(4.0)  // TODO should attempt to get from metadata
		factor := float32(8.0)            // metadata?
		old_context_len := float32(8192)  // metadata?  (aka original_max_position_embeddings)

		// Calcs...
		low_freq_wavelen := float32(old_context_len) / low_freq_factor
		high_freq_wavelen := float32(old_context_len) / high_freq_factor

		// freqs = base ** (mx.arange(0, dims, 2) / dims)
		freqs := arange(0, float32(dims), 2)
		for i := range freqs {
			freqs[i] = (float32)(math.Pow(float64(base), float64(freqs[i])/float64(dims)))
		}
		// wavelens = 2 * mx.pi * freqs
		wavelens := make([]float32, len(freqs))
		for i := range wavelens {
			wavelens[i] = freqs[i] * 2 * float32(math.Pi)
		}
		// freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
		for i := range freqs {
			if wavelens[i] > low_freq_wavelen {
				freqs[i] = freqs[i] * factor
			}
		}
		// is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
		is_medium_freq := make([]bool, len(freqs))
		for i := range freqs {
			is_medium_freq[i] = (wavelens[i] > high_freq_wavelen) && (wavelens[i] < low_freq_wavelen)
		}
		// smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
		smooth_factors := make([]float32, len(freqs))
		for i := range freqs {
			smooth_factors[i] = ((old_context_len)/wavelens[i] - (low_freq_factor)) / ((high_freq_factor) - (low_freq_factor))
		}
		// smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
		smooth_freqs := make([]float32, len(freqs))
		for i := range freqs {
			smooth_freqs[i] = freqs[i] / ((1-smooth_factors[i])/factor + (smooth_factors[i]))
		}
		// _freqs = mx.where(is_medium_freq, smooth_freqs, freqs)
		for i := range freqs {
			if is_medium_freq[i] {
				freqs[i] = float32(smooth_freqs[i])
			}
		}
		_freqs, _ = ctx.FromFloatSlice(freqs, len(freqs))
	}
	once.Do(onceBody)

	return x.Rope(
		ctx,
		offset,
		_freqs,
		dims,
		0,   // base unused
		1.0, // scale
	)
}

func arange(start, end, step float32) []float32 {
	if step == 0 || start >= end {
		return nil
	}
	var res []float32
	for i := float32(start); i < end; i += step {
		res = append(res, i)
	}
	return res
}

type MLP struct {
	Up   *nn.Linear `ggml:"ffn_up"`
	Down *nn.Linear `ggml:"ffn_down"`
	Gate *nn.Linear `ggml:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	g := mlp.Gate.Forward(ctx, hiddenState)
	x := mlp.Up.Forward(ctx, hiddenState)
	hiddenState = g.SILU(ctx).Mul(ctx, x)
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `ggml:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `ggml:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState ml.Tensor, offset int32, cache model.Cache, opts *Options) ml.Tensor {
	residual := hiddenState
	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, offset, cache, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	out := hiddenState.Add(ctx, residual)
	return out
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	inputs, err := ctx.FromIntSlice(opts.Inputs(), len(opts.Inputs()))
	if err != nil {
		return nil, err
	}
	offset := int32(0)
	hiddenState := m.TokenEmbedding.Forward(ctx, inputs).Reshape(ctx, 1, -1, 4096)

	for i, layer := range m.Layers {
		hiddenState = layer.Forward(ctx, hiddenState, offset, opts.Cache.Sub(i), m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)

	// TODO this isn't the right solution, but we need to do this only once (there's a bug here someplace...)
	s := m.Output.Weight.Shape()
	if s[0] != hiddenState.Shape()[2] {
		m.Output.Weight = m.Output.Weight.Permute(ctx, 1, 0)
	}

	outputs, err := ctx.FromIntSlice([]int32{-1}, 1, 1)
	if err != nil {
		return nil, err
	}
	t := hiddenState.Rows(ctx, outputs).Reshape(ctx, 1, -1)

	hiddenState = m.Output.Forward(ctx, t)
	return hiddenState, nil
}

func init() {
	model.Register("llama", New)
}
