package diffusiongemma

import (
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// decodeCanvasSample runs one bidirectional denoising forward, unembeds, and
// computes the per-position reductions (argmax, entropy, self-cond top-k, and
// the multinomial draw) ON-DEVICE, returning only the small host arrays: the
// [canvasLen*vocab] logits never cross to the host and the per-vocab math runs on
// the GPU instead of a single-threaded Go loop over a host-materialized copy.
// key seeds the multinomial draw.
func (m *Model) decodeCanvasSample(canvas []int32, nPast int32, selfCond *SelfCond, caches []cache.Cache, temp float32, scK int, key *mlx.Array) canvasSample {
	hidden := m.forward(&batch.Batch{
		InputIDs:     mlx.FromValues(canvas, 1, len(canvas)),
		SeqOffsets:   []int32{nPast},
		SeqQueryLens: []int32{int32(len(canvas))},
	}, caches, &forwardOpts{
		decoderPhase: true,
		canvasStart:  nPast,
		canvasLen:    int32(len(canvas)),
		selfCond:     selfCond,
	})

	vocab := int(m.VocabSize)
	logits := mlx.Reshape(m.Unembed(hidden).AsType(mlx.DTypeFloat32), int32(len(canvas)), int32(vocab))
	return sampleCanvasDevice(logits, len(canvas), vocab, temp, scK, key)
}

// sampleCanvasDevice computes the per-position step results from on-device,
// already-softcapped logits ([canvasLen, vocab]). It mirrors sampleCanvas (the
// host reference) exactly for the deterministic outputs (argmax, entropy,
// self-cond top-k); the multinomial draw uses MLX's key-based RNG, so it differs
// from the host inverse-CDF draw but samples from the same temperature softmax.
func sampleCanvasDevice(logits *mlx.Array, canvasLen, vocab int, temp float32, scK int, key *mlx.Array) canvasSample {
	if temp <= 0 {
		temp = 1
	}
	if scK < 1 {
		scK = 1
	}
	if scK > vocab {
		scK = vocab
	}

	scaled := logits
	if temp != 1 {
		scaled = mlx.DivScalar(logits, temp)
	}

	// Entropy H = -Σ p·ln p with p = softmax(scaled). Using the log-sum-exp,
	// ln p = scaled - lse, so H = lse - Σ p·scaled (a numerically stable form).
	lse := scaled.LogsumexpAxis(-1, true)            // [L,1]
	p := mlx.Exp(mlx.Sub(scaled, lse))               // [L,V] softmax
	pScaled := mlx.Sum(mlx.Mul(p, scaled), -1, true) // [L,1]
	entropy := mlx.Reshape(mlx.Sub(lse, pScaled), int32(canvasLen))

	// Argmax (committed token) and the unordered top-k self-cond ids, both via
	// argpartition on the negated logits (smallest-of-negated == largest).
	neg := scaled.Negative()
	argmax := neg.ArgpartitionAxis(0, -1).Slice(mlx.Slice(), mlx.Slice(0, 1)).AsType(mlx.DTypeInt32) // [L,1]
	topkU := neg.ArgpartitionAxis(scK-1, -1).Slice(mlx.Slice(), mlx.Slice(0, scK))                   // [L,scK] (U32)
	topkProbs := p.TakeAlongAxis(topkU, -1)                                                          // [L,scK]
	topkIDs := topkU.AsType(mlx.DTypeInt32)

	// Multinomial draw per position (used to renoise accepted positions).
	sampled := scaled.CategoricalWithKey(-1, key).AsType(mlx.DTypeInt32) // [L]

	mlx.Eval(entropy, argmax, topkIDs, topkProbs, sampled)

	ent := entropy.Floats()
	for i, e := range ent {
		if e < 0 {
			ent[i] = 0 // guard tiny negative from fp error, matching the host path
		}
	}
	return canvasSample{
		entropy: ent,
		argmax:  intsToInt32(argmax.Ints()),
		sampled: intsToInt32(sampled.Ints()),
		scIDs:   intsToInt32(topkIDs.Ints()),
		scProbs: topkProbs.Floats(),
	}
}

func intsToInt32(in []int) []int32 {
	out := make([]int32, len(in))
	for i, v := range in {
		out[i] = int32(v)
	}
	return out
}
