package cache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// TestRecurrentCacheRestoreExactOffset verifies that RecurrentCache restore
// only succeeds when target exactly matches the snapshot's offset. Recurrent
// state is cumulative, so it can't be rewound or fast-forwarded.
func TestRecurrentCacheRestoreExactOffset(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 12, 4, 8, 8)
	b1 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1)}
	c.Get(b1, mlx.DTypeFloat16) // lazy-init

	b10 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 10), SeqQueryLens: []int32{10}}
	c.Put(b10, nil, nil) // advance to 10

	snap := c.Snapshot(0) // snap.offset == 10

	b5 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 5), SeqQueryLens: []int32{5}}
	c.Put(b5, nil, nil) // cache now at 15

	// target < snap.offset: fails (can't rewind past snapshot)
	if c.Restore(snap, 5) {
		t.Fatal("Restore(snap, 5) should fail — target != snap.offset")
	}

	// target > snap.offset: fails (can't advance without feeding tokens)
	if c.Restore(snap, 15) {
		t.Fatal("Restore(snap, 15) should fail — target != snap.offset")
	}

	// target == snap.offset: succeeds
	if !c.Restore(snap, 10) {
		t.Fatal("Restore(snap, 10) should succeed — target == snap.offset")
	}
	if c.Offset() != 10 {
		t.Fatalf("offset = %d, want 10", c.Offset())
	}
}

func TestRecurrentCacheGetLazyInit(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 4, 2, 4, 4)
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 1),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
	}
	h := c.Get(b, mlx.DTypeBFloat16)
	if c.Offset() != 0 {
		t.Fatalf("Get should not advance; got offset %d", c.Offset())
	}
	if h.ConvState() == nil || h.DeltaState() == nil {
		t.Fatal("history should expose conv/delta tensors")
	}
	if got := h.ConvState().DType(); got != mlx.DTypeBFloat16 {
		t.Fatalf("conv state dtype = %v, want %v", got, mlx.DTypeBFloat16)
	}
	if got := h.DeltaState().DType(); got != mlx.DTypeFloat32 {
		t.Fatalf("delta state dtype = %v, want %v", got, mlx.DTypeFloat32)
	}
}

func TestSpeculativeRecurrentCacheUsesStagedState(t *testing.T) {
	skipIfNoMLX(t)
	target := NewRecurrentCache(2, 3, 1, 2, 3)
	caches, ok := BeginIsolatedSpeculation([]Cache{target})
	if !ok {
		t.Fatal("BeginIsolatedSpeculation failed")
	}
	c := caches[0].(*speculativeRecurrentCache)
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, 1),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
	}

	c.Get(b, mlx.DTypeFloat32)

	convVals := []float32{1, 2, 3, 4, 5, 6}
	deltaVals := []float32{7, 8, 9, 10, 11, 12}
	nextConv := mlx.FromValues(convVals, 1, 2, 3)
	nextDelta := mlx.FromValues(deltaVals, 1, 1, 2, 3)
	c.Put(b, nextConv, nextDelta)

	h := c.Get(b, mlx.DTypeFloat32)
	state := c.State()
	if len(state) != 2 {
		t.Fatalf("State() returned %d arrays, want 2", len(state))
	}

	assertArray := func(name string, got, want *mlx.Array) {
		t.Helper()
		if got != want {
			t.Fatalf("%s = %p, want %p", name, got, want)
		}
	}
	assertArray("history conv", h.ConvState(), nextConv)
	assertArray("history delta", h.DeltaState(), nextDelta)
	assertArray("state conv", state[0], nextConv)
	assertArray("state delta", state[1], nextDelta)

	if got := c.Offset(); got != 1 {
		t.Fatalf("speculative offset = %d, want 1", got)
	}
	if got := target.Offset(); got != 0 {
		t.Fatalf("target offset = %d, want 0", got)
	}
}

// TestRecurrentCachePaddedRoundTrip runs Get → CausalConv1D →
// GatedDelta → Put on a B=1 batch with qLen<L, then again on a
// fresh cache with an unpadded length-qLen batch using the same
// real prefix. After the call, Offset() must equal qLen (not L),
// and the resulting cache state must match the unpadded equivalent.
// Pins the recurrent contract: a forward with padding produces the
// same end-state as a forward with the real-prefix-only input.
func TestRecurrentCachePaddedRoundTrip(t *testing.T) {
	skipIfNoMLX(t)
	const convTail, convDim = 2, 6
	const numVHeads, headVDim, headKDim = 1, 4, 6
	const L = 4
	const qLen = 2

	// Use distinct values for the real prefix and the padded tail so
	// we can detect any leak from padded positions into the result.
	makeQKV := func(seed float32, T int) (q, k, v *mlx.Array) {
		mkLast := func(off float32, T, n, d int) *mlx.Array {
			vals := make([]float32, 1*T*n*d)
			for i := range vals {
				vals[i] = off + 0.05*float32(i)
			}
			return mlx.FromValues(vals, 1, T, n, d)
		}
		q = mkLast(seed, T, 1, headKDim)
		k = mkLast(seed+0.1, T, 1, headKDim)
		v = mkLast(seed+0.2, T, numVHeads, headVDim)
		return
	}
	makeGB := func(seed float32, T int) (g, beta *mlx.Array) {
		gVals := make([]float32, 1*T*numVHeads)
		bVals := make([]float32, 1*T*numVHeads)
		for i := range gVals {
			gVals[i] = seed + 0.01*float32(i)
			bVals[i] = seed - 0.02*float32(i)
		}
		g = mlx.FromValues(gVals, 1, T, numVHeads)
		beta = mlx.FromValues(bVals, 1, T, numVHeads)
		return
	}
	makeQKVPadded := func() (q, k, v *mlx.Array) {
		qReal, kReal, vReal := makeQKV(0.3, qLen)
		// Distinct, large junk values in the padded tail to surface
		// any leak (real outputs are O(1)).
		qPad := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L-qLen, 1, headKDim), 99)
		kPad := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L-qLen, 1, headKDim), 99)
		vPad := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L-qLen, numVHeads, headVDim), 99)
		q = mlx.Concatenate([]*mlx.Array{qReal, qPad}, 1)
		k = mlx.Concatenate([]*mlx.Array{kReal, kPad}, 1)
		v = mlx.Concatenate([]*mlx.Array{vReal, vPad}, 1)
		return
	}
	makeGBPadded := func() (g, beta *mlx.Array) {
		gReal, betaReal := makeGB(0.1, qLen)
		gPad := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L-qLen, numVHeads), 99)
		betaPad := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L-qLen, numVHeads), 99)
		g = mlx.Concatenate([]*mlx.Array{gReal, gPad}, 1)
		beta = mlx.Concatenate([]*mlx.Array{betaReal, betaPad}, 1)
		return
	}

	// The conv input dimension must match the cache's convDim.
	mkConvInput := func(seed float32, T int) *mlx.Array {
		vals := make([]float32, 1*T*convDim)
		for i := range vals {
			vals[i] = seed + 0.05*float32(i)
		}
		return mlx.FromValues(vals, 1, T, convDim)
	}
	mkWeight := func(seed float32) *mlx.Array {
		vals := make([]float32, convDim*(convTail+1))
		for i := range vals {
			vals[i] = seed + 0.1*float32(i)
		}
		return mlx.FromValues(vals, convDim, convTail+1)
	}
	weight := mkWeight(0.2)

	runForward := func(c *RecurrentCache, b *batch.Batch, T int) (*mlx.Array, *mlx.Array) {
		var convInput *mlx.Array
		if T == L {
			realPart := mkConvInput(0.4, qLen)
			padPart := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, T-qLen, convDim), 99)
			convInput = mlx.Concatenate([]*mlx.Array{realPart, padPart}, 1)
		} else {
			convInput = mkConvInput(0.4, T)
		}

		history := c.Get(b, mlx.DTypeFloat32)
		_, nextConv := nn.CausalConv1D(b, convInput, nil, weight, convTail,
			nn.WithRecurrentHistory(history))

		var q, k, v, g, beta *mlx.Array
		if T == L {
			q, k, v = makeQKVPadded()
			g, beta = makeGBPadded()
		} else {
			q, k, v = makeQKV(0.3, T)
			g, beta = makeGB(0.1, T)
		}
		_, newDelta := nn.GatedDelta(b, q, k, v, g, beta,
			nn.WithRecurrentHistory(history))

		c.Put(b, nextConv, newDelta)
		return nextConv, newDelta
	}

	// Padded forward.
	cPad := NewRecurrentCache(convTail, convDim, numVHeads, headVDim, headKDim)
	bPad := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, L),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{int32(qLen)},
	}
	nextConvPad, deltaPad := runForward(cPad, bPad, L)
	mlx.Eval(nextConvPad, deltaPad)
	if got := cPad.Offset(); got != qLen {
		t.Fatalf("padded forward: Offset() = %d, want %d (must advance by SeqQueryLens, not L)", got, qLen)
	}

	// Unpadded reference.
	cRef := NewRecurrentCache(convTail, convDim, numVHeads, headVDim, headKDim)
	bRef := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 1, qLen),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{int32(qLen)},
	}
	nextConvRef, deltaRef := runForward(cRef, bRef, qLen)
	mlx.Eval(nextConvRef, deltaRef)
	if got := cRef.Offset(); got != qLen {
		t.Fatalf("unpadded forward: Offset() = %d, want %d", got, qLen)
	}

	gp := nextConvPad.Floats()
	gr := nextConvRef.Floats()
	if len(gp) != len(gr) {
		t.Fatalf("nextConv shape mismatch: padded %d vs unpadded %d", len(gp), len(gr))
	}
	for i := range gp {
		if math.Abs(float64(gp[i]-gr[i])) > 1e-4 {
			t.Fatalf("nextConv[%d]: padded=%v unpadded=%v (padding leaked into conv state)", i, gp[i], gr[i])
		}
	}

	dp := deltaPad.Floats()
	dr := deltaRef.Floats()
	if len(dp) != len(dr) {
		t.Fatalf("delta state shape mismatch: padded %d vs unpadded %d", len(dp), len(dr))
	}
	for i := range dp {
		if math.Abs(float64(dp[i]-dr[i])) > 1e-3 {
			t.Fatalf("delta state[%d]: padded=%v unpadded=%v (padding leaked into recurrent state)", i, dp[i], dr[i])
		}
	}
}

func TestRecurrentCachePutAdvances(t *testing.T) {
	skipIfNoMLX(t)
	c := NewRecurrentCache(3, 4, 2, 4, 4)
	b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 2), SeqQueryLens: []int32{2}}
	newConv := mlx.Zeros(mlx.DTypeFloat16, 1, 3, 4)
	newDelta := mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4, 4)
	c.Put(b, newConv, newDelta)
	if c.Offset() != 2 {
		t.Fatalf("cache offset not advanced: %d", c.Offset())
	}
}
