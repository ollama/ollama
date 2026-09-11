package cache

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// TestRecurrentCacheRestoreExactOffset verifies that RecurrentCache restore
// only succeeds when target exactly matches the snapshot's offset. Recurrent
// state is cumulative, so it can't be rewound or fast-forwarded.
func TestRecurrentCacheRestoreExactOffset(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := NewRecurrentCache(3, 12, 4, 8, 8)
		b1 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1)}
		c.Get(b1, mlx.DTypeFloat16) // lazy-init

		keep := func() ([]*mlx.Array, []*mlx.Array) {
			s := c.State()
			return []*mlx.Array{s[0]}, []*mlx.Array{s[1]}
		}

		b10 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 10), SeqQueryLens: []int32{10}}
		cs, ds := keep()
		c.Put(b10, cs, ds) // advance to 10

		snap := c.Snapshot(0) // snap.offset == 10

		b5 := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 5), SeqQueryLens: []int32{5}}
		cs, ds = keep()
		c.Put(b5, cs, ds) // cache now at 15

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
	})
}

func TestRecurrentCacheGetLazyInit(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
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
	})
}

// TestRecurrentCachePaddedRoundTrip runs Get → CausalConv1D →
// GatedDelta → Put on a B=1 batch with qLen<L, then again on a
// fresh cache with an unpadded length-qLen batch using the same
// real prefix. After the call, Offset() must equal qLen (not L),
// and the resulting cache state must match the unpadded equivalent.
// Pins the recurrent contract: a forward with padding produces the
// same end-state as a forward with the real-prefix-only input.
func TestRecurrentCachePaddedRoundTrip(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const convTail, convDim = 2, 6
		const numVHeads, headVDim, headKDim = 1, 4, 6
		const L = 4
		const qLen = 2

		// Distinct values for the real prefix and large junk in the padded
		// tail so any leak from padded positions is visible.
		const packedDim = 2*headKDim + numVHeads*headVDim
		mkPacked := func(seed float32, T int) (packed, ba *mlx.Array) {
			pv := make([]float32, T*packedDim)
			bv := make([]float32, T*2*numVHeads)
			for i := range pv {
				pv[i] = seed + 0.05*float32(i)
			}
			for i := range bv {
				bv[i] = seed - 0.02*float32(i)
			}
			return mlx.FromValues(pv, 1, T, packedDim), mlx.FromValues(bv, 1, T, 2*numVHeads)
		}
		mkPackedPadded := func() (packed, ba *mlx.Array) {
			pReal, baReal := mkPacked(0.3, qLen)
			pPad := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L-qLen, packedDim), 99)
			baPad := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, L-qLen, 2*numVHeads), 99)
			return mlx.Concatenate([]*mlx.Array{pReal, pPad}, 1), mlx.Concatenate([]*mlx.Array{baReal, baPad}, 1)
		}
		dtBias := mlx.FromValues([]float32{0.3}, numVHeads)
		aExp := mlx.FromValues([]float32{0.12}, numVHeads)

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
		// Build the depthwise causal Conv1d as the model does at load time: the
		// [C, K] kernel becomes [C, K, 1] and the conv is grouped per channel.
		conv := nn.NewConv1d(mlx.ExpandDims(weight, 2), nil, 1, 0, 1, convDim)

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
			_, convStates := nn.CausalConv1D(b, convInput, conv, convTail,
				nn.WithRecurrentHistory(history))

			var packed, ba *mlx.Array
			if T == L {
				packed, ba = mkPackedPadded()
			} else {
				packed, ba = mkPacked(0.3, T)
			}
			_, deltaStates := nn.GatedDelta(b, packed, ba, dtBias, aExp, nn.WithRecurrentHistory(history))

			c.Put(b, convStates, deltaStates)
			return convStates[len(convStates)-1], deltaStates[len(deltaStates)-1]
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
	})
}

func TestRecurrentCachePutAdvances(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := NewRecurrentCache(3, 4, 2, 4, 4)
		b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 2), SeqQueryLens: []int32{2}}
		newConv := mlx.Zeros(mlx.DTypeFloat16, 1, 3, 4)
		newDelta := mlx.Zeros(mlx.DTypeFloat16, 1, 2, 4, 4)
		c.Put(b, []*mlx.Array{newConv}, []*mlx.Array{newDelta})
		if c.Offset() != 2 {
			t.Fatalf("cache offset not advanced: %d", c.Offset())
		}
	})
}
