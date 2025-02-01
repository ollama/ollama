package llama

import (
	"math"
	"sync"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

func LlamaRoPE(ctx ml.Context, x, positionIDs ml.Tensor, opts *Options) ml.Tensor {
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

		// freqs = base ** (mx.model.ArangeF32(0, dims, 2) / dims)
		freqs := model.ArangeF32(0, float32(dims), 2)
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

	/* MLX breadcrumb for Fast RoPE
	a (array) – Input array.
	dims (int) – The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged.
	traditional (bool) – If set to True choose the traditional implementation which rotates consecutive dimensions.
	base (float, optional) – The base used to compute angular frequency for each dimension in the positional encodings. Exactly one of base and freqs must be None.
	scale (float) – The scale used to scale the positions.
	offset (int or array) – The position offset to start at.
	freqs (array, optional) – Optional frequencies to use with RoPE. If set, the base parameter must be None. Default: None.
	*/

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

/*
B: The batch size.
N_q: The number of query heads.
N_kv: The number of key and value heads.
T_q: The number of queries per example.
T_kv: The number of keys and values per example.
D: The per-head dimension.

q (array) – Queries with shape [B, N_q, T_q, D].
k (array) – Keys with shape [B, N_kv, T_kv, D].
v (array) – Values with shape [B, N_kv, T_kv, D].
scale (float) – Scale for queries (typically 1.0 / sqrt(q.shape(-1))
mask (array, optional) – A boolean or additive mask to apply to the query-key scores.

	The mask can have at most 4 dimensions and must be broadcast-compatible with the shape [B, N, T_q, T_kv].
	If an additive mask is given its type must promote to the promoted type of q, k, and v.
*/
func ScaledDotProductAttention(ctx ml.Context, queries, keys, values, mask ml.Tensor, scale float32) ml.Tensor {
	if sdpa, ok := ctx.(ml.FastScaledDotProductAttention); ok {
		// MLX support
		return sdpa.FastScaledDotProductAttention(queries, keys, values, scale, mask)
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
		scores = scores.Scale(ctx, float64(scale))

		scores = scores.Add(ctx, mask)
		scores = scores.Softmax(ctx) // Without axis=-1 this starts to drift
		return values.Mulmat(ctx, scores)
	}
}
