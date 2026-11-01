package cache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/nn"
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
	const H, D = 1, 4
	const window = 4
	const totalWrites = 7 // past wrap (window=4); last write is the L=1 decode
	const scale = 1.0

	var q, kLogical, vLogical, logicalMask *mlx.Array
	var b *batch.Batch
	var history *nn.KVHistory
	mlxtest.Run(t, func(t *mlxtest.T) {
		q = mlx.FromValues([]float32{0.7, -0.4, 0.2, 0.9}, 1, H, 1, D)

		// Drive the cache: write positions 0..totalWrites-2 as a "history",
		// then position totalWrites-1 is the actual L=1 decode under test.
		c := NewRotatingKVCache(window)
		for pos := range totalWrites - 1 {
			k, v := perPosKV(pos)
			c.Update(newKVBatch(c.Offset(), k.Dim(2)), k, v)
		}

		finalPos := totalWrites - 1
		kFinal, vFinal := perPosKV(finalPos)
		b = &batch.Batch{
			InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 1),
			SeqOffsets:   []int32{int32(finalPos)},
			SeqQueryLens: []int32{1},
		}
		history = c.Update(b, kFinal, vFinal)

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
		kLogical = mlx.Concatenate(logicalKs, 2)
		vLogical = mlx.Concatenate(logicalVs, 2)

		// A logical-order ArrayMask with distinct, non-trivial values per
		// key column. Picked so each column's contribution to softmax is
		// distinct — the test fails if the cache's gather permutes the
		// columns wrong before the kernel sees them.
		logicalMask = mlx.FromValues([]float32{0.1, -0.3, 0.7, -0.2}, 1, 1, 1, window)
	})

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
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			got := nn.ScaledDotProductAttention(b, q, scale,
				nn.WithKVHistory(history),
				nn.WithMask(tc.model))

			want := mlx.FastScaledDotProductAttention(q, kLogical, vLogical, scale,
				mlx.FastSDPANoWindow, tc.refMode, tc.refMask)

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

func TestAssistantSharedHistoryL1MasksMatchNoMask(t *testing.T) {
	const H, D = 1, 4
	const window = 4
	const total = 7
	const scale = 1.0

	var available bool
	var q *mlx.Array
	var b *batch.Batch
	var cases []struct {
		name string
		h    *nn.KVHistory
		mask nn.AttentionMask
	}
	mlxtest.Run(t, func(t *mlxtest.T) {
		available = mlx.MetalIsAvailable()
		if !available {
			return
		}

		q = mlx.FromValues([]float32{0.7, -0.4, 0.2, 0.9}, 1, H, 1, D)
		full := NewKVCache()
		sliding := NewRotatingKVCache(window)
		for pos := range total {
			kVals := make([]float32, H*D)
			vVals := make([]float32, H*D)
			for i := range kVals {
				kVals[i] = 0.1*float32(pos+1) + 0.01*float32(i)
				vVals[i] = -0.1*float32(pos+1) + 0.01*float32(i)
			}
			k := mlx.FromValues(kVals, 1, H, 1, D)
			v := mlx.FromValues(vVals, 1, H, 1, D)
			full.Update(newKVBatch(full.Offset(), 1), k, v)
			sliding.Update(newKVBatch(sliding.Offset(), 1), k, v)
		}

		b = newKVBatch(total-1, 1)
		cases = []struct {
			name string
			h    *nn.KVHistory
			mask nn.AttentionMask
		}{
			{name: "full", h: full.View(b), mask: nn.CausalMask()},
			{name: "sliding", h: sliding.View(b), mask: nn.CausalMask()},
		}
	})
	if !available {
		t.Skip("MLX Metal not available")
	}

	for _, tc := range cases {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			got := nn.ScaledDotProductAttention(b, q, scale, nn.WithKVHistory(tc.h), nn.WithMask(tc.mask))
			want := mlx.FastScaledDotProductAttention(q, tc.h.K(), tc.h.V(), scale, mlx.FastSDPANoWindow, "", nil)

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
	const H, L, D = 1, 6, 4
	const window = 4
	const scale = 1.0

	var q, k, v *mlx.Array
	var b *batch.Batch
	mlxtest.Run(t, func(t *mlxtest.T) {
		qVals := make([]float32, 1*H*L*D)
		kVals := make([]float32, 1*H*L*D)
		vVals := make([]float32, 1*H*L*D)
		for i := range qVals {
			qVals[i] = 0.5 + 0.05*float32(i)
			kVals[i] = -0.3 + 0.07*float32(i)
			vVals[i] = 0.3 + 0.03*float32(i)
		}
		q = mlx.FromValues(qVals, 1, H, L, D)
		k = mlx.FromValues(kVals, 1, H, L, D)
		v = mlx.FromValues(vVals, 1, H, L, D)
		b = newKVBatch(0, L)
	})

	cases := []struct {
		name string
		mask nn.AttentionMask
		// rect arguments matching nn.AttentionMask.Relax (qLo, qHi, kLo, kHi)
		relax  [][4]int
		causal bool
	}{
		{"zero", nn.AttentionMask{}, nil, false},
		{"causal", nn.CausalMask(), nil, true},
		{"device-causal", nn.DeviceCausalMask(), nil, true},
		{"causal+relax", nn.CausalMask().Relax(0, 1, 4, 2, 5), [][4]int{{1, 4, 2, 5}}, true},
	}

	negInf := float32(math.Inf(-1))
	for _, tc := range cases {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
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
			want := mlx.FastScaledDotProductAttention(q, k, v, scale, mlx.FastSDPANoWindow, "array", refMask)

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

// perPosKV builds clearly distinguishable per-position K/V rows for
// rotating-cache parity tests.
var perPosKV = func(pos int) (k, v *mlx.Array) {
	const H, D = 1, 4
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

// logicalWindowRef builds the post-Update logical (oldest-first) K/V over
// positions [start, end] and the additive causal ∩ sliding reference mask
// for a continuation query run starting at absolute offset off of length L.
func logicalWindowRef(off, L, window, start, end int) (kRef, vRef, mask *mlx.Array) {
	K := end - start + 1
	ks := make([]*mlx.Array, 0, K)
	vs := make([]*mlx.Array, 0, K)
	for pos := start; pos <= end; pos++ {
		kp, vp := perPosKV(pos)
		ks = append(ks, kp)
		vs = append(vs, vp)
	}
	kRef = mlx.Concatenate(ks, 2)
	vRef = mlx.Concatenate(vs, 2)

	negInf := float32(math.Inf(-1))
	vals := make([]float32, L*K)
	for r := range L {
		absQ := off + r
		for k := range K {
			absK := start + k
			if absK > absQ || absK < absQ-window+1 {
				vals[r*K+k] = negInf
			}
		}
	}
	mask = mlx.FromValues(vals, 1, 1, L, K)
	return
}

// TestRotatingKVCacheContinuationPrefillParity drives single-token writes
// past the wrap point, then an L>1 continuation chunk — the shape that
// linearizes the ring and dispatches natively (mode="causal" with a kernel
// window) instead of materializing a mask — and compares against a reference
// computed from per-position K/V in logical order with the causal ∩ window
// mask composed manually. A follow-on L=1 decode pins the mixed sequence.
func TestRotatingKVCacheContinuationPrefillParity(t *testing.T) {
	const H, D = 1, 4
	const window = 4
	const before = 5 // positions 0..4, past wrap (window=4)
	const L = 4      // continuation chunk: positions 5..8
	const scale = 1.0

	mlxtest.Run(t, func(t *mlxtest.T) {
		c := NewRotatingKVCache(window)
		for pos := range before {
			k, v := perPosKV(pos)
			c.Update(newKVBatch(c.Offset(), 1), k, v)
		}

		// Concat linearizes the wrapped buffer and keeps maxSize-1 old
		// tokens: positions 2..4 old + 5..8 new, K=7 in logical order.
		batchK, batchV := perPosKV(before)
		for i := 1; i < L; i++ {
			k, v := perPosKV(before + i)
			batchK = mlx.Concatenate([]*mlx.Array{batchK, k}, 2)
			batchV = mlx.Concatenate([]*mlx.Array{batchV, v}, 2)
		}
		qVals := make([]float32, H*L*D)
		for i := range qVals {
			qVals[i] = 0.5 + 0.05*float32(i)
		}
		q := mlx.FromValues(qVals, 1, H, L, D)
		b := newKVBatch(before, L)
		history := c.Update(b, batchK, batchV)

		got := nn.ScaledDotProductAttention(b, q, scale,
			nn.WithKVHistory(history),
			nn.WithMask(nn.CausalMask()))

		kRef, vRef, refMask := logicalWindowRef(before, L, window, 2, before+L-1)
		want := mlx.FastScaledDotProductAttention(q, kRef, vRef, scale,
			mlx.FastSDPANoWindow, "array", refMask)
		mlx.Eval(got, want)
		gs, ws := got.Floats(), want.Floats()
		for i := range ws {
			if math.Abs(float64(gs[i]-ws[i])) > 1e-4 {
				t.Fatalf("continuation index %d: got %v, want %v", i, gs[i], ws[i])
			}
		}

		// Follow-on decode at position 9: storage bounds the window, so
		// compare against the trailing positions 6..9 in logical order
		// with no mask (attention is permutation-invariant).
		kDec, vDec := perPosKV(9)
		bDec := newKVBatch(9, 1)
		gotDec := nn.ScaledDotProductAttention(bDec, mlx.SliceStartStop(q, []int32{0, 0, 0, 0}, []int32{1, H, 1, D}), scale,
			nn.WithKVHistory(c.Update(bDec, kDec, vDec)),
			nn.WithMask(nn.CausalMask()))
		kDecRef, vDecRef := perPosKV(6)
		for _, pos := range []int{7, 8, 9} {
			kp, vp := perPosKV(pos)
			kDecRef = mlx.Concatenate([]*mlx.Array{kDecRef, kp}, 2)
			vDecRef = mlx.Concatenate([]*mlx.Array{vDecRef, vp}, 2)
		}
		qDec := mlx.SliceStartStop(q, []int32{0, 0, 0, 0}, []int32{1, H, 1, D})
		wantDec := mlx.FastScaledDotProductAttention(qDec, kDecRef, vDecRef, scale,
			mlx.FastSDPANoWindow, "", nil)
		mlx.Eval(gotDec, wantDec)
		gs, ws = gotDec.Floats(), wantDec.Floats()
		for i := range ws {
			if math.Abs(float64(gs[i]-ws[i])) > 1e-5 {
				t.Fatalf("decode index %d: got %v, want %v", i, gs[i], ws[i])
			}
		}
	})
}

// TestRotatingKVCacheSinksWindowPrefillParity is the flint gate at cache
// level: an L>1 chunk into a rotating cache with WithSinks must match the
// materialized causal ∩ sliding reference with sinks, at a GQA shape like
// flint's sliding layers. Native dispatch composes sinks and window in one
// kernel; this fails if that composition is wrong.
func TestRotatingKVCacheSinksWindowPrefillParity(t *testing.T) {
	const Hq, Hkv, D = 2, 1, 4
	const window = 4
	const before = 3 // positions 0..2
	const L = 3      // chunk: positions 3..5
	const scale = 1.0

	mlxtest.Run(t, func(t *mlxtest.T) {
		// Per-position K/V single-head rows.
		write := func(c *RotatingKVCache, pos int) {
			k, v := perPosKV(pos)
			c.Update(newKVBatch(c.Offset(), 1), k, v)
		}
		chunkKV := func(start, n int) (k, v *mlx.Array) {
			k, v = perPosKV(start)
			for i := 1; i < n; i++ {
				kp, vp := perPosKV(start + i)
				k = mlx.Concatenate([]*mlx.Array{k, kp}, 2)
				v = mlx.Concatenate([]*mlx.Array{v, vp}, 2)
			}
			return
		}

		c := NewRotatingKVCache(window)
		for pos := range before {
			write(c, pos)
		}
		kChunk, vChunk := chunkKV(before, L)

		qVals := make([]float32, Hq*L*D)
		for i := range qVals {
			qVals[i] = 0.5 + 0.03*float32(i)
		}
		q := mlx.FromValues(qVals, 1, Hq, L, D)
		b := newKVBatch(before, L)
		history := c.Update(b, kChunk, vChunk)
		sinks := mlx.FromValues([]float32{0.3, -0.7}, Hq)

		got := nn.ScaledDotProductAttention(b, q, scale,
			nn.WithKVHistory(history),
			nn.WithMask(nn.CausalMask()),
			nn.WithSinks(sinks))

		// K=6 logical positions 0..5 (buffer never wrapped).
		kRef, vRef, refMask := logicalWindowRef(before, L, window, 0, before+L-1)
		want := mlx.FastScaledDotProductAttentionWithSinks(q, kRef, vRef, scale,
			mlx.FastSDPANoWindow, "array", refMask, sinks)
		mlx.Eval(got, want)
		gs, ws := got.Floats(), want.Floats()
		for i := range ws {
			if math.Abs(float64(gs[i]-ws[i])) > 1e-3 {
				t.Fatalf("index %d: got %v, want %v", i, gs[i], ws[i])
			}
		}
	})
}

// TestRotatingKVCacheScheduledSnapshotParity verifies that scheduling per-token
// snapshots over a batched write past the wrap point does not change the
// attention the write returns: the history must produce the same SDPA output as
// an identical write with no snapshots scheduled. Capture happens after the
// write via lazy snapshots, so it must not perturb the write itself.
func TestRotatingKVCacheScheduledSnapshotParity(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const H, D = 1, 4
		const window = 4
		const before = 5 // past wrap before the batched write
		const draft = 4
		const scale = 1.0

		// Build the batched K/V for offsets [before, before+draft).
		batchK := make([]*mlx.Array, draft)
		batchV := make([]*mlx.Array, draft)
		for i := range draft {
			batchK[i], batchV[i] = perPosKV(before + i)
		}
		kBatch := mlx.Concatenate(batchK, 2)
		vBatch := mlx.Concatenate(batchV, 2)

		qVals := make([]float32, H*draft*D)
		for i := range qVals {
			qVals[i] = 0.5 + 0.05*float32(i)
		}
		q := mlx.FromValues(qVals, 1, H, draft, D)
		b := newKVBatch(before, draft)

		// Run the same write twice: once with snapshots scheduled, once without.
		run := func(schedule bool) []float32 {
			c := NewRotatingKVCache(window)
			for pos := range before {
				k, v := perPosKV(pos)
				c.Update(newKVBatch(c.Offset(), 1), k, v)
			}
			if schedule {
				offsets := make([]int, draft)
				for i := range offsets {
					offsets[i] = before + i
				}
				c.PrepareSnapshots(offsets)
			}
			history := c.Update(b, kBatch, vBatch)
			out := nn.ScaledDotProductAttention(b, q, scale,
				nn.WithKVHistory(history),
				nn.WithMask(nn.CausalMask()))
			if schedule {
				for _, s := range c.TakeSnapshots() {
					if s != nil {
						s.Close()
					}
				}
			}
			mlx.Eval(out)
			return out.Floats()
		}

		withSnap := run(true)
		noSnap := run(false)
		if len(withSnap) != len(noSnap) {
			t.Fatalf("output length %d vs %d", len(withSnap), len(noSnap))
		}
		for i := range noSnap {
			if math.Abs(float64(withSnap[i]-noSnap[i])) > 1e-5 {
				t.Fatalf("index %d: scheduled=%v, unscheduled=%v", i, withSnap[i], noSnap[i])
			}
		}
	})
}

// TestRotatingKVCacheMLAParity drives a rotating cache with the MLA
// shape — K = [kvLatent, kPE] concatenated, V = zero-width — then
// uses WithMLAHistory to slice V from K and compares output against
// a manual reference. Pins the cache+MLA integration that
// glm4_moe_lite uses in production.
func TestRotatingKVCacheMLAParity(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
		want := mlx.FastScaledDotProductAttention(q, k, vRef, scale, mlx.FastSDPANoWindow, "causal", nil)

		mlx.Eval(got, want)
		gs, ws := got.Floats(), want.Floats()
		for i := range ws {
			if math.Abs(float64(gs[i]-ws[i])) > 1e-5 {
				t.Fatalf("index %d: got %v, want %v", i, gs[i], ws[i])
			}
		}
	})
}
