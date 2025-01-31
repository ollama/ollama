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

func ScaledDotProductAttention(ctx ml.Context, q, k, v, mask ml.Tensor, scale float32) ml.Tensor {
	if sdpa, ok := ctx.(ml.FastScaledDotProductAttention); ok {
		// MLX support
		return sdpa.FastScaledDotProductAttention(q, k, v, scale, mask)
	} else {
		// GGML support
		kq := k.Mulmat(ctx, q)
		kq = kq.Scale(ctx, float64(scale))
		kq = kq.Add(ctx, mask)
		kq = kq.Softmax(ctx)
		v = v.Permute(ctx, 0, 2, 1, 3).Contiguous(ctx)
		return v.Mulmat(ctx, kq)
	}
}
