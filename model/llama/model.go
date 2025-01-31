package llama

import (
	"math"
	"sync"

	"github.com/ollama/ollama/cache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn"
	"github.com/ollama/ollama/model"
)

type Options struct {
	RopeFactors                      ml.Tensor `gguf:"rope_freqs.weight"`
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
	return &Model{
		BytePairEncoding: model.BytePairEncoding{
			Pretokenizer: c.String("tokenizer.ggml.pretokenizer", `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
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
			ropeDim:    c.Uint("rope.dimension_count"),
		},
	}, nil
}

type SelfAttention struct {
	Query  *nn.Linear `gguf:"attn_q"`
	Key    *nn.Linear `gguf:"attn_k"`
	Value  *nn.Linear `gguf:"attn_v"`
	Output *nn.Linear `gguf:"attn_output"`
}

func (sa *SelfAttention) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache cache.Cache, opts *Options) ml.Tensor {
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

	queries = RoPE(ctx, queries, positionIDs, opts)
	keys = RoPE(ctx, keys, positionIDs, opts)

	cache.Put(ctx, keys, values, 2)
	keys, values, mask := cache.Get(ctx)
	var output ml.Tensor
	if sdpa, ok := ctx.(ml.FastScaledDotProductAttention); ok {
		output = sdpa.FastScaledDotProductAttention(queries, keys, values, float32(scale), mask)
	} else {
		// Note: this "non-fast" flow can work on backends besides GGML, but requires some adjustments called out below
		// TODO make this a common utility models

		// Begin scaled dot product attention
		// Ref: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L196
		// Note: The mlx python implementation does not have this, since it relies on FastScaledDotProductAttention
		keys = keys.Permute(ctx, 0, 2, 1, 3)
		values = values.Permute(ctx, 0, 2, 1, 3)
		queries = queries.Permute(ctx, 0, 2, 1, 3)

		// repeat k/v heads if n_kv_heads < n_heads
		// equiv to torch.repeat_interleave(x, dim=2, repeats=int(n_heads / n_kv_heads))
		// keys = keys.Repeat(ctx, int(n_heads/n_kv_heads), 2).Contiguous(ctx)     // Note: enable this line on !GGML backend
		// values = values.Repeat(ctx, int(n_heads/n_kv_heads), 2).Contiguous(ctx) // Note: enable this line on !GGML backend

		queries = queries.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx) // (bs, n_heads, L, head_dim)
		keys = keys.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)       // (bs, n_heads, cache_len + L, head_dim)

		// Note: toggle these on !GGML backend
		// values = values.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx) // !GGML logic (bs, n_heads, cache_len + L, head_dim)
		values = values.Permute(ctx, 0, 3, 1, 2).Contiguous(ctx) // Note: GGML requires this to work

		// keys = keys.Permute(ctx, 0, 1, 3, 2).Contiguous(ctx) // Note: enable this line on !GGML backend

		scores := keys.Mulmat(ctx, queries)
		scores = scores.Scale(ctx, 1.0/math.Sqrt(float64(head_dim)))

		scores = scores.Add(ctx, mask)
		scores = scores.Softmax(ctx) // Without axis=-1 this starts to drift
		output = values.Mulmat(ctx, scores)
	}

	output = output.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
	output = output.Reshape(ctx, B, L, -1)
	output = sa.Output.Forward(ctx, output)
	return output
}

// TODO make this a common utility for LLama models
func RoPE(ctx ml.Context, x, positionIDs ml.Tensor, opts *Options) ml.Tensor {
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

	return x.RoPE(
		ctx,
		positionIDs,
		opts.RopeFactors,
		_freqs,
		dims,
		500000, // base
		1.0,    // scale
	)
}

// TODO make this a common util
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
	Up   *nn.Linear `gguf:"ffn_up"`
	Down *nn.Linear `gguf:"ffn_down"`
	Gate *nn.Linear `gguf:"ffn_gate"`
}

func (mlp *MLP) Forward(ctx ml.Context, hiddenState ml.Tensor, opts *Options) ml.Tensor {
	g := mlp.Gate.Forward(ctx, hiddenState)
	x := mlp.Up.Forward(ctx, hiddenState)
	hiddenState = g.SILU(ctx).Mul(ctx, x)
	return mlp.Down.Forward(ctx, hiddenState)
}

type Layer struct {
	AttentionNorm *nn.RMSNorm `gguf:"attn_norm"`
	SelfAttention *SelfAttention
	MLPNorm       *nn.RMSNorm `gguf:"ffn_norm"`
	MLP           *MLP
}

func (l *Layer) Forward(ctx ml.Context, hiddenState, positionIDs ml.Tensor, cache cache.Cache, opts *Options) ml.Tensor {
	residual := hiddenState

	hiddenState = l.AttentionNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.SelfAttention.Forward(ctx, hiddenState, positionIDs, cache, opts)
	hiddenState = hiddenState.Add(ctx, residual)
	residual = hiddenState

	hiddenState = l.MLPNorm.Forward(ctx, hiddenState, opts.eps)
	hiddenState = l.MLP.Forward(ctx, hiddenState, opts)
	return hiddenState.Add(ctx, residual)
}

func (m *Model) Forward(ctx ml.Context, opts model.Options) (ml.Tensor, error) {
	inputs, err := ctx.FromIntSlice(opts.Inputs(), len(opts.Inputs()))
	if err != nil {
		return nil, err
	}
	positions, err := ctx.FromIntSlice(opts.Positions(), len(opts.Positions()))
	if err != nil {
		return nil, err
	}

	hiddenState := m.TokenEmbedding.Forward(ctx, inputs)
	hiddenState = hiddenState.Reshape(ctx, 1, -1, 4096)

	for i, layer := range m.Layers {
		opts.Cache.SetLayer(i)
		hiddenState = layer.Forward(ctx, hiddenState, positions, opts.Cache, m.Options)
	}

	hiddenState = m.OutputNorm.Forward(ctx, hiddenState, m.eps)
	hiddenState = m.Output.Forward(ctx, hiddenState)

	// outputs, err := ctx.FromIntSlice(opts.Outputs(), len(opts.Outputs()))
	outputs, err := ctx.FromIntSlice([]int32{int32(hiddenState.Dim(1)) - 1}, 1, 1)
	if err != nil {
		return nil, err
	}

	return hiddenState.Rows(ctx, outputs).Reshape(ctx, 1, -1), nil
}

func (m *Model) Shift(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
	return RoPE(ctx, key, shift, m.Options), nil
}

func init() {
	model.Register("llama", New)
}
