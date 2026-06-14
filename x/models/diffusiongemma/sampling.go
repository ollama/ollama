package diffusiongemma

import (
	"math"
	"math/rand"
	"sort"
)

// This file implements the host-side block-diffusion sampling kernels, ported
// from the llama.cpp reference (PR #24427, diffusion-gemma-cli.cpp full-softmax
// host path). They operate on plain Go slices of host-materialized logits so the
// algorithm — the tricky part — is unit-testable without a Metal device. The MLX
// forward passes that produce the logits live in the model forward.
//
// Key facts that drive this design (verified against the reference):
//   - There is NO mask token. A canvas is initialized to uniformly random token
//     ids; "remasking" means renoising rejected positions with fresh random ids.
//   - Position acceptance is ENTROPY-BOUND prefix selection (sort positions by
//     entropy ascending, accept a prefix while the running entropy sum stays
//     <= entropy_bound), NOT confidence/top-k unmasking.
//   - The committed/emitted tokens are the per-position ARGMAX of the final
//     settled step, not the multinomially sampled canvas.
//   - Logits handed in here are already final-softcapped by the model's Unembed.

// tempAt returns the linear temperature for a denoising step. curStep counts
// DOWN from nSteps to 1, so step nSteps yields tMax and step 1 yields
// tMin + (tMax-tMin)/nSteps (the tMin floor is the asymptote, never reached —
// matching the reference exactly).
func tempAt(curStep, nSteps int, tMin, tMax float32) float32 {
	if nSteps <= 0 {
		return tMax
	}
	return tMin + (tMax-tMin)*float32(curStep)/float32(nSteps)
}

// numCanvases returns how many autoregressive canvases (blocks) to generate for
// the requested number of predicted tokens. Defaults to 1 when nPredict <= 0.
func numCanvases(nPredict, canvasLen int) int {
	if canvasLen <= 0 {
		return 1
	}
	if nPredict <= 0 {
		return 1
	}
	return max((nPredict+canvasLen-1)/canvasLen, 1)
}

// randomCanvas fills a fresh canvas with uniformly random token ids in
// [0, vocab).
func randomCanvas(rng *rand.Rand, canvasLen, vocab int) []int32 {
	c := make([]int32, canvasLen)
	for i := range c {
		c[i] = int32(rng.Intn(vocab))
	}
	return c
}

// canvasSample holds the per-position results of one denoising step.
type canvasSample struct {
	entropy []float32 // per-position Shannon entropy of the temperature softmax
	sampled []int32   // per-position multinomial draw
	argmax  []int32   // per-position argmax (the committed/emitted candidate)
	scIDs   []int32   // self-cond top-k token ids, laid out [pos*k + j]
	scProbs []float32 // self-cond top-k probs, same layout
}

// sampleCanvas computes, per canvas position, the temperature-softmax entropy, a
// multinomial sample, the argmax, and the top-k (id, prob) self-conditioning
// payload. logits is row-major [canvasLen*vocab] and already softcapped. This is
// the full-softmax host path; it is O(canvasLen*vocab) and intentionally simple
// (perf — moving this on-device — is a later concern).
func sampleCanvas(logits []float32, canvasLen, vocab int, temp float32, scK int, rng *rand.Rand) canvasSample {
	if temp <= 0 {
		temp = 1
	}
	if scK < 1 {
		scK = 1
	}
	if scK > vocab {
		scK = vocab
	}
	out := canvasSample{
		entropy: make([]float32, canvasLen),
		sampled: make([]int32, canvasLen),
		argmax:  make([]int32, canvasLen),
		scIDs:   make([]int32, canvasLen*scK),
		scProbs: make([]float32, canvasLen*scK),
	}

	for j := range canvasLen {
		row := logits[j*vocab : (j+1)*vocab]

		// max(logit/temp) for numerical stability, and the argmax token.
		amax := 0
		maxScaled := float32(math.Inf(-1))
		for v, lg := range row {
			s := lg / temp
			if s > maxScaled {
				maxScaled = s
				amax = v
			}
		}

		// Unnormalized softmax p[v] = exp(logit/temp - maxScaled), and its sum.
		// Accumulate entropy via sum(p) and sum(p*log p), normalized at the end:
		//   H = ln(sum) - (1/sum) * sum(p * ln p_unnorm)   ... computed directly below.
		var sum float64
		var plogp float64 // sum p*ln(p) over unnormalized p
		for _, lg := range row {
			p := math.Exp(float64(lg/temp - maxScaled))
			sum += p
			if p > 0 {
				plogp += p * math.Log(p)
			}
		}
		// Normalized entropy: H = -sum (p/Z) ln(p/Z) = ln Z - (1/Z) sum p ln p.
		entropy := math.Log(sum) - plogp/sum
		if entropy < 0 {
			entropy = 0 // guard tiny negative from fp error
		}
		out.entropy[j] = float32(entropy)
		out.argmax[j] = int32(amax)

		// Multinomial draw via inverse CDF over unnormalized p; fall back to amax.
		r := rng.Float64() * sum
		acc := 0.0
		sampled := amax
		for v, lg := range row {
			acc += math.Exp(float64(lg/temp - maxScaled))
			if acc >= r {
				sampled = v
				break
			}
		}
		out.sampled[j] = int32(sampled)

		// Top-k by scaled logit, with full-normalized softmax probs, for self-cond.
		fillTopK(row, temp, maxScaled, float32(sum), scK, out.scIDs[j*scK:(j+1)*scK], out.scProbs[j*scK:(j+1)*scK])
	}
	return out
}

// fillTopK writes the top-k token ids (by logit) and their normalized softmax
// probabilities into ids/probs. Naive (sorts all vocab indices); fine for tests
// and the reference host path, optimized later.
func fillTopK(row []float32, temp, maxScaled, sum float32, k int, ids []int32, probs []float32) {
	idx := make([]int, len(row))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return row[idx[a]] > row[idx[b]] })
	for t := range k {
		v := idx[t]
		ids[t] = int32(v)
		probs[t] = float32(math.Exp(float64(row[v]/temp-maxScaled))) / sum
	}
}

// entropyBoundAccept selects which positions to commit this step: positions
// sorted by entropy ascending are accepted while the running entropy sum stays
// <= bound. The bound is tested BEFORE adding the current position's entropy
// (matching the reference's off-by-one), so the last accepted position may push
// the sum past the bound.
func entropyBoundAccept(entropy []float32, bound float32) []bool {
	n := len(entropy)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return entropy[order[a]] < entropy[order[b]] })

	accept := make([]bool, n)
	var prefix float32
	for _, idx := range order {
		if prefix <= bound {
			accept[idx] = true
			prefix += entropy[idx]
		} else {
			break
		}
	}
	return accept
}

// stableAndConfident is the per-canvas early-stop test. stable means the
// full-canvas argmax is unchanged from the previous step (for stabThresh==1; a
// stabThresh of 0 means always-stable). confident means the mean per-position
// entropy is below confThresh. prevArgmax is nil on the first step (never stable).
func stableAndConfident(argmax, prevArgmax []int32, entropy []float32, confThresh float32, stabThresh int) bool {
	stable := stabThresh == 0 || int32sEqual(argmax, prevArgmax)
	if !stable {
		return false
	}
	var sum float32
	for _, e := range entropy {
		sum += e
	}
	mean := sum / float32(max(len(entropy), 1))
	return mean < confThresh
}

// renoise produces the next-step canvas: accepted positions keep their sampled
// token, the rest are overwritten with fresh uniformly random tokens.
func renoise(sampled []int32, accept []bool, rng *rand.Rand, vocab int) []int32 {
	next := make([]int32, len(sampled))
	for i := range sampled {
		if accept[i] {
			next[i] = sampled[i]
		} else {
			next[i] = int32(rng.Intn(vocab))
		}
	}
	return next
}

func int32sEqual(a, b []int32) bool {
	if len(a) != len(b) || b == nil {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
