package cache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// TestRotatingKVCacheDecodeParity drives a rotating cache past its
// wrap point with single-token writes, runs an L=1 decode through
// SDPA, and compares against a reference computed from the same per-
// position K/V in logical-position order with the same caller mask.
//
// Attention is permutation-invariant when K, V, and the mask are
// permuted together, so the cache's storage-order output (with the
// applier's gather composing the caller's logical mask back into
// storage order) must equal the logical-order reference.
func TestRotatingKVCacheDecodeParity(t *testing.T) {
	skipIfNoMLX(t)
	const H, D = 1, 4
	const window = 4
	const totalWrites = 7 // past wrap (window=4); last write is the L=1 decode
	const scale = 1.0

	// Per-position k, v values. Use distinct seeds so the per-position
	// values are clearly distinguishable.
	perPosKV := func(pos int) (k, v *mlx.Array) {
		kVals := make([]float32, H*D)
		vVals := make([]float32, H*D)
		for i := range kVals {
			kVals[i] = 0.1*float32(pos+1) + 0.01*float32(i)
			vVals[i] = -0.1*float32(pos+1) + 0.01*float32(i)
		}
		k = mlx.FromValues(kVals, 1, H, 1, D)
		v = mlx.FromValues(vVals, 1, H, 1, D)
		return
	}

	q := mlx.FromValues([]float32{0.7, -0.4, 0.2, 0.9}, 1, H, 1, D)
	mlx.Eval(q)

	// Drive the cache: write positions 0..totalWrites-2 as a "history",
	// then position totalWrites-1 is the actual L=1 decode under test.
	c := NewRotatingKVCache(window)
	for pos := range totalWrites - 1 {
		k, v := perPosKV(pos)
		c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
	}

	finalPos := totalWrites - 1
	kFinal, vFinal := perPosKV(finalPos)
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 1),
		SeqOffsets:   []int32{int32(finalPos)},
		SeqQueryLens: []int32{1},
	}
	history := c.Update(b, kFinal, vFinal)

	// Reference: the in-window logical-position-ordered K and V are
	// the last `window` per-position values (positions
	// [finalPos-window+1, finalPos]). Build them in that order.
	startPos := max(finalPos-window+1, 0)
	logicalKs := make([]*mlx.Array, 0, window)
	logicalVs := make([]*mlx.Array, 0, window)
	for pos := startPos; pos <= finalPos; pos++ {
		kp, vp := perPosKV(pos)
		logicalKs = append(logicalKs, kp)
		logicalVs = append(logicalVs, vp)
	}
	kLogical := mlx.Concatenate(logicalKs, 2)
	vLogical := mlx.Concatenate(logicalVs, 2)

	// A logical-order ArrayMask with distinct, non-trivial values per
	// key column. Picked so each column's contribution to softmax is
	// distinct — the test fails if the cache's gather permutes the
	// columns wrong before the kernel sees them.
	maskVals := []float32{0.1, -0.3, 0.7, -0.2}
	logicalMask := mlx.FromValues(maskVals, 1, 1, 1, window)

	cases := []struct {
		name  string
		model nn.AttentionMask
		// reference mask uses the same coordinates the model mask
		// represents; for ArrayMask it's the same tensor (since the
		// reference K/V is in logical order).
		refMode string
		refMask *mlx.Array
	}{
		{"zero", nn.AttentionMask{}, "", nil},
		{"causal-at-L1", nn.CausalMask(), "", nil},
		{"array", nn.ArrayMask(logicalMask), "array", logicalMask},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := nn.ScaledDotProductAttention(b, q, scale,
				nn.WithKVHistory(history),
				nn.WithMask(tc.model))

			want := mlx.FastScaledDotProductAttention(q, kLogical, vLogical, scale,
				tc.refMode, tc.refMask)

			mlx.Eval(got, want)
			gs, ws := got.Floats(), want.Floats()
			for i := range ws {
				if math.Abs(float64(gs[i]-ws[i])) > 1e-5 {
					t.Fatalf("index %d: got %v, want %v", i, gs[i], ws[i])
				}
			}
		})
	}
}

// TestRotatingKVCachePrefillParity drives an L>1 prefill into a
// rotating cache and verifies SDPA output through WithKVHistory
// matches a reference computed from the same K/V with the model mask
// and window restriction composed manually.
func TestRotatingKVCachePrefillParity(t *testing.T) {
	skipIfNoMLX(t)
	const H, L, D = 1, 6, 4
	const window = 4
	const scale = 1.0

	qVals := make([]float32, 1*H*L*D)
	kVals := make([]float32, 1*H*L*D)
	vVals := make([]float32, 1*H*L*D)
	for i := range qVals {
		qVals[i] = 0.5 + 0.05*float32(i)
		kVals[i] = -0.3 + 0.07*float32(i)
		vVals[i] = 0.3 + 0.03*float32(i)
	}
	q := mlx.FromValues(qVals, 1, H, L, D)
	k := mlx.FromValues(kVals, 1, H, L, D)
	v := mlx.FromValues(vVals, 1, H, L, D)
	b := newKVBatch(0, L)

	cases := []struct {
		name string
		mask nn.AttentionMask
		// rect arguments matching nn.AttentionMask.Relax (qLo, qHi, kLo, kHi)
		relax  [][4]int
		causal bool
	}{
		{"zero", nn.AttentionMask{}, nil, false},
		{"causal", nn.CausalMask(), nil, true},
		{"causal+relax", nn.CausalMask().Relax(0, 1, 4, 2, 5), [][4]int{{1, 4, 2, 5}}, true},
	}

	negInf := float32(math.Inf(-1))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := NewRotatingKVCache(window)
			history := c.Update(b, k, v)

			got := nn.ScaledDotProductAttention(b, q, scale,
				nn.WithKVHistory(history),
				nn.WithMask(tc.mask))

			// Reference mask: causal blocks k > absQ; relax rectangles
			// release causal-blocked cells; window blocks k < absQ - window + 1.
			refVals := make([]float32, L*L)
			for qi := range L {
				absQ := qi
				for ki := range L {
					blocked := false
					if tc.causal && ki > absQ {
						blocked = true
					}
					for _, r := range tc.relax {
						qLo, qHi, kLo, kHi := r[0], r[1], r[2], r[3]
						if absQ >= qLo && absQ < qHi && ki >= kLo && ki < kHi {
							blocked = false
						}
					}
					if window > 0 && ki < absQ-window+1 {
						blocked = true
					}
					if blocked {
						refVals[qi*L+ki] = negInf
					}
				}
			}
			refMask := mlx.FromValues(refVals, 1, 1, L, L)
			want := mlx.FastScaledDotProductAttention(q, k, v, scale, "array", refMask)

			mlx.Eval(got, want)
			gs, ws := got.Floats(), want.Floats()
			for i := range ws {
				if math.Abs(float64(gs[i]-ws[i])) > 1e-4 {
					t.Fatalf("index %d: got %v, want %v", i, gs[i], ws[i])
				}
			}
		})
	}
}

// TestRotatingKVCacheMLAParity drives a rotating cache with the MLA
// shape — K = [kvLatent, kPE] concatenated, V = zero-width — then
// uses WithMLAHistory to slice V from K and compares output against
// a manual reference. Pins the cache+MLA integration that
// glm4_moe_lite uses in production.
func TestRotatingKVCacheMLAParity(t *testing.T) {
	skipIfNoMLX(t)
	const H, L, D, valueDim = 1, 3, 6, 4
	const scale = 1.0
	const window = 8 // window >= L so no window restriction

	kVals := make([]float32, 1*H*L*D)
	for i := range kVals {
		kVals[i] = 0.1 * float32(i+1)
	}
	k := mlx.FromValues(kVals, 1, H, L, D)
	v := mlx.Zeros(mlx.DTypeFloat32, 1, H, L, 0)

	q := mlx.Zeros(mlx.DTypeFloat32, 1, H, L, D)
	b := newKVBatch(0, L)

	c := NewRotatingKVCache(window)
	history := c.Update(b, k, v)
	got := nn.ScaledDotProductAttention(b, q, scale,
		nn.WithMLAHistory(history, valueDim),
		nn.WithMask(nn.CausalMask()))

	vRef := k.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, valueDim))
	want := mlx.FastScaledDotProductAttention(q, k, vRef, scale, "causal", nil)

	mlx.Eval(got, want)
	gs, ws := got.Floats(), want.Floats()
	for i := range ws {
		if math.Abs(float64(gs[i]-ws[i])) > 1e-5 {
			t.Fatalf("index %d: got %v, want %v", i, gs[i], ws[i])
		}
	}
}
