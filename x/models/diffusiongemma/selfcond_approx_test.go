package diffusiongemma

import (
	"math"
	"math/rand"
	"testing"
)

// fullSoftmaxSelfCondSoft is the REFERENCE self-conditioning "soft" vector that
// the top-k path in computeSelfCond approximates. It matches the llama.cpp
// reference (PR #24427): soft[j] = Σ_v softmax(logits[j]/temp)[v] · embed[v],
// summed over the WHOLE vocabulary. Returned row-major [canvasLen*dim]. The
// EmbedScale and the self-cond MLP that computeSelfCond applies afterwards are
// identical for both paths, so they are omitted here — only `soft` differs.
func fullSoftmaxSelfCondSoft(logits, embed []float32, canvasLen, vocab, dim int, temp float32) []float32 {
	soft := make([]float32, canvasLen*dim)
	for j := range canvasLen {
		row := logits[j*vocab : (j+1)*vocab]
		maxScaled := float32(math.Inf(-1))
		for _, lg := range row {
			if s := lg / temp; s > maxScaled {
				maxScaled = s
			}
		}
		var sum float64
		for _, lg := range row {
			sum += math.Exp(float64(lg/temp - maxScaled))
		}
		for v, lg := range row {
			p := float32(math.Exp(float64(lg/temp-maxScaled)) / sum)
			for d := range dim {
				soft[j*dim+d] += p * embed[v*dim+d]
			}
		}
	}
	return soft
}

// topKSelfCondSoft is what computeSelfCond actually sums: only the top-k terms,
// with the SAME normalized softmax probs (fillTopK / sample_device both divide by
// the full softmax sum). It is therefore the exact truncation of
// fullSoftmaxSelfCondSoft. Also returns the per-position captured probability mass.
func topKSelfCondSoft(scIDs []int32, scProbs, embed []float32, canvasLen, scK, dim int) (soft, captured []float32) {
	soft = make([]float32, canvasLen*dim)
	captured = make([]float32, canvasLen)
	for j := range canvasLen {
		for t := range scK {
			id := int(scIDs[j*scK+t])
			pr := scProbs[j*scK+t]
			captured[j] += pr
			for d := range dim {
				soft[j*dim+d] += pr * embed[id*dim+d]
			}
		}
	}
	return soft, captured
}

// TestSelfCondTopKApproximatesFullSoftmax documents and bounds the deliberate
// top-k self-conditioning approximation in computeSelfCond. Because the top-k
// probs are the normalized full-softmax probs (not renormalized), soft_topk is
// the EXACT truncation of soft_full, so the error is exactly the dropped tail:
//
//	||soft_full - soft_topk|| = ||Σ_tail p·embed|| <= (1 - capturedMass) · maxEmbedNorm
//
// This bound holds for ANY distribution — proving the approximation is intentional
// and bounded — and is TIGHT for the peaked distributions that converging
// denoising settles into, where top-k (SelfCondK=32) captures ~all the mass.
func TestSelfCondTopKApproximatesFullSoftmax(t *testing.T) {
	const (
		L     = 4
		vocab = 512
		dim   = 16
		scK   = 32
		temp  = float32(0.8)
	)
	rng := rand.New(rand.NewSource(7))

	// Synthetic embedding table [vocab, dim] and its max row L2 norm.
	embed := make([]float32, vocab*dim)
	for i := range embed {
		embed[i] = float32(rng.NormFloat64())
	}
	maxNorm := 0.0
	for v := range vocab {
		var n float64
		for d := range dim {
			e := float64(embed[v*dim+d])
			n += e * e
		}
		maxNorm = math.Max(maxNorm, math.Sqrt(n))
	}

	mkLogits := func(peaked bool) []float32 {
		lg := make([]float32, L*vocab)
		for i := range lg {
			lg[i] = float32(rng.NormFloat64())
		}
		if peaked {
			for j := range L {
				for range 3 { // a few sharply dominant tokens per position
					lg[j*vocab+rng.Intn(vocab)] += 12
				}
			}
		}
		return lg
	}

	cases := []struct {
		name        string
		peaked      bool
		minCaptured float32
	}{
		{"peaked (converging-denoising regime)", true, 0.99},
		{"flat (worst case — only the general bound must hold)", false, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			logits := mkLogits(tc.peaked)
			host := sampleCanvas(logits, L, vocab, temp, scK, rng)

			full := fullSoftmaxSelfCondSoft(logits, embed, L, vocab, dim, temp)
			topk, captured := topKSelfCondSoft(host.scIDs, host.scProbs, embed, L, scK, dim)

			for j := range L {
				var errNorm float64
				for d := range dim {
					e := float64(full[j*dim+d] - topk[j*dim+d])
					errNorm += e * e
				}
				errNorm = math.Sqrt(errNorm)

				// General bound: the truncation error never exceeds the dropped
				// tail mass times the largest embedding norm.
				bound := float64(1-captured[j])*maxNorm + 1e-3 // fp slack
				if errNorm > bound {
					t.Errorf("pos %d: error %.6g exceeds tail-mass bound %.6g (captured=%.4f)",
						j, errNorm, bound, captured[j])
				}
				// Tightness: in the peaked regime top-k captures ~all the mass.
				if captured[j] < tc.minCaptured {
					t.Errorf("pos %d: top-%d captured only %.4f of mass (want >= %.2f)",
						j, scK, captured[j], tc.minCaptured)
				}
			}
		})
	}
}
