package nn

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// lastState returns the forward-end state — the last boundary the recurrent
// wrappers return.
func lastState(states []*mlx.Array) *mlx.Array { return states[len(states)-1] }

// fromValues builds a tensor with sequentially-numbered float32
// values so element-by-element parity actually exercises the kernel.
func fromValues(seed float32, shape ...int) *mlx.Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = seed + 0.1*float32(i)
	}
	return mlx.FromValues(vals, shape...)
}

// convFromKernel builds the depthwise causal Conv1d the model constructs at
// load time from a bare [C, K] kernel, so the wrapper tests drive the same
// mlx.Conv1d path production runs.
func convFromKernel(w *mlx.Array) *Conv1d {
	return NewConv1d(mlx.ExpandDims(w, 2), nil, 1, 0, 1, int32(w.Dim(0)))
}

// Guards a biased conv silently losing the fused kernel: depthwiseConvWeight
// returning nil sends WithConvSiLU down separate graph ops.
func TestCausalConv1DBiasTakesFusedPath(t *testing.T) {
	skipIfNoMLX(t)
	B, L, D, convTail := 2, 3, 4, 2
	K := convTail + 1

	weight := fromValues(-0.3, D, K)
	bias := fromValues(0.4, D)
	conv := NewConv1d(mlx.ExpandDims(weight, 2), bias, 1, 0, 1, int32(D))

	if depthwiseConvWeight(conv) == nil {
		t.Fatal("depthwiseConvWeight = nil for a biased depthwise conv, so the fused path is skipped")
	}

	prior := fromValues(0.2, B, convTail, D)
	input := fromValues(0.1, B, L, D)
	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, B, L),
		SeqOffsets:   []int32{0, 0},
		SeqQueryLens: []int32{int32(L), int32(L)},
	}

	got, _ := CausalConv1D(b, input, conv, convTail, WithRecurrentState(prior, nil), WithConvSiLU())
	want := mlx.SiLU(conv.Forward(mlx.Concatenate([]*mlx.Array{prior, input}, 1)))
	mlx.Eval(got, want)
	floatsClose(t, "biased fused conv+silu", got.Floats(), want.Floats(), 1e-5)
}

// TestCausalConv1DPaddedRowParity drives a B=2 batch with one short
// row (qLen<L). For the short row, (a) `out` positions [0..qLen)
// must equal a B=1 reference at length qLen, (b) `nextConv` for the
// short row must be the row's last convTail real positions (not the
// padded tail), (c) the full row must be unaffected.
func TestCausalConv1DPaddedRowParity(t *testing.T) {
	skipIfNoMLX(t)
	L, D, convTail := 4, 3, 2
	qLenShort := 2
	K := convTail + 1

	weight := fromValues(0.2, D, K)
	conv := convFromKernel(weight)
	priorFull := fromValues(0.5, 2, convTail, D)
	priorShort := mlx.SliceStartStop(priorFull,
		[]int32{1, 0, 0},
		[]int32{2, int32(convTail), int32(D)})

	// Pad row 1 with arbitrary values past qLenShort — the wrapper
	// must zero them before convolving. Distinct values let us catch
	// any leak.
	inputFull := fromValues(1.0, 1, L, D)
	inputShortReal := mlx.FromValues([]float32{
		2.0, 2.1, 2.2,
		2.3, 2.4, 2.5,
	}, 1, qLenShort, D)
	inputShortPad := mlx.FromValues([]float32{
		99, 99, 99,
		99, 99, 99,
	}, 1, L-qLenShort, D)
	inputShortFull := mlx.Concatenate([]*mlx.Array{inputShortReal, inputShortPad}, 1)
	input := mlx.Concatenate([]*mlx.Array{inputFull, inputShortFull}, 0)

	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 2, L),
		SeqOffsets:   []int32{0, 0},
		SeqQueryLens: []int32{int32(L), int32(qLenShort)},
	}

	out, convStates := CausalConv1D(b, input, conv, convTail, WithRecurrentState(priorFull, nil))
	nextConv := lastState(convStates)
	mlx.Eval(out, nextConv)

	// Reference for row 0: B=1 unpadded length-L call.
	refOut0, refConvStates0 := CausalConv1D(&batch.Batch{},
		inputFull, conv, convTail,
		WithRecurrentState(mlx.SliceStartStop(priorFull,
			[]int32{0, 0, 0},
			[]int32{1, int32(convTail), int32(D)}), nil))
	refNextConv0 := lastState(refConvStates0)
	// Reference for row 1: B=1 unpadded length-qLenShort call.
	refOut1, refConvStates1 := CausalConv1D(&batch.Batch{},
		inputShortReal, conv, convTail,
		WithRecurrentState(priorShort, nil))
	refNextConv1 := lastState(refConvStates1)
	mlx.Eval(refOut0, refNextConv0, refOut1, refNextConv1)

	gotOut := out.Floats()
	wantOut0 := refOut0.Floats()
	wantOut1 := refOut1.Floats()

	for q := range L {
		for d := range D {
			i := q*D + d
			if gotOut[i] != wantOut0[i] {
				t.Fatalf("row 0 out[q=%d,d=%d]: got %v, want %v", q, d, gotOut[i], wantOut0[i])
			}
		}
	}
	for q := range qLenShort {
		for d := range D {
			gotI := L*D + q*D + d
			refI := q*D + d
			if math.Abs(float64(gotOut[gotI]-wantOut1[refI])) > 1e-5 {
				t.Fatalf("row 1 real out[q=%d,d=%d]: got %v, want %v", q, d, gotOut[gotI], wantOut1[refI])
			}
		}
	}

	// nextConv: row 0 unaffected, row 1 must be the row's real tail
	// (positions [qLenShort - convTail, qLenShort) of the per-row
	// concat, i.e. the last two real input rows in this setup).
	gotTail := nextConv.Floats()
	wantTail0 := refNextConv0.Floats()
	wantTail1 := refNextConv1.Floats()
	for k := range convTail {
		for d := range D {
			i := k*D + d
			if gotTail[i] != wantTail0[i] {
				t.Fatalf("row 0 nextConv[k=%d,d=%d]: got %v, want %v", k, d, gotTail[i], wantTail0[i])
			}
		}
	}
	for k := range convTail {
		for d := range D {
			gotI := convTail*D + k*D + d
			refI := k*D + d
			if gotTail[gotI] != wantTail1[refI] {
				t.Fatalf("row 1 nextConv[k=%d,d=%d]: got %v, want %v (must come from real positions, not the padded tail)",
					k, d, gotTail[gotI], wantTail1[refI])
			}
		}
	}
}

// gatedDeltaPackedInputs builds deterministic packed conv-output and
// projection rows plus the per-head parameters for a GatedDelta call.
func gatedDeltaPackedInputs(B, T, Hk, Dk, Hv, Dv int) (packed, ba, dtBias, aExp *mlx.Array) {
	packed = fromValues(0.05, B, T, 2*Hk*Dk+Hv*Dv)
	ba = fromValues(-0.2, B, T, 2*Hv)
	dtBias = fromValues(0.3, Hv)
	aExp = fromValues(0.12, Hv)
	return packed, ba, dtBias, aExp
}

// slicePrefix returns rows [lo, hi) of a truncated to the first n positions
// along axis 1.
func slicePrefix(a *mlx.Array, lo, hi, n int32) *mlx.Array {
	dims := a.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	start[0], stop[0] = lo, hi
	for i := 1; i < len(dims); i++ {
		stop[i] = int32(dims[i])
	}
	if len(dims) >= 2 {
		stop[1] = n
	}
	return mlx.SliceStartStop(a, start, stop)
}

// floatsClose compares two flat float slices within tolerance.
func floatsClose(t *testing.T, label string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d]: got %v, want %v", label, i, got[i], want[i])
		}
	}
}

// TestGatedDeltaSegmentEquivalence checks that split forwards match the
// single-shot call — both the per-token split pattern (the kernels'
// captureAll shape) and a sparse split (the per-segment composition) — and
// that each boundary state equals the single-shot state over the
// corresponding prefix.
func TestGatedDeltaSegmentEquivalence(t *testing.T) {
	skipIfNoMLX(t)
	B, T, Hk, Dk, Hv, Dv := 1, 5, 1, 32, 1, 32
	packed, ba, dtBias, aExp := gatedDeltaPackedInputs(B, T, Hk, Dk, Hv, Dv)
	prior := mlx.Zeros(mlx.DTypeFloat32, B, Hv, Dv, Dk)
	full := &batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(T)}}

	refOut, refStates := GatedDelta(full, packed, ba, dtBias, aExp, WithRecurrentState(nil, prior))
	if len(refStates) != 1 {
		t.Fatalf("unsegmented call returned %d states, want 1", len(refStates))
	}

	cases := []struct {
		name   string
		splits []int
	}{
		{"perToken", []int{1, 2, 3, 4}},
		{"sparse", []int{2}},
	}
	for _, tc := range cases {
		segOut, segStates := GatedDelta(full, packed, ba, dtBias, aExp,
			WithRecurrentState(nil, prior), WithSnapshotSplits(tc.splits))
		mlx.Eval(refOut, segOut)
		floatsClose(t, tc.name+" out", segOut.Floats(), refOut.Floats(), 1e-4)
		if len(segStates) != len(tc.splits)+1 {
			t.Fatalf("%s: got %d boundary states, want %d", tc.name, len(segStates), len(tc.splits)+1)
		}
		boundaries := append(append([]int{}, tc.splits...), T)
		for i, n := range boundaries {
			_, want, _ := mlx.GatedDelta(
				slicePrefix(packed, 0, 1, int32(n)), slicePrefix(ba, 0, 1, int32(n)),
				dtBias, aExp, prior, nil, false)
			mlx.Eval(segStates[i], want)
			floatsClose(t, tc.name+" boundary delta", segStates[i].Floats(), want.Floats(), 1e-4)
		}
	}
}

// TestCausalConv1DSegmentEquivalence checks the conv segmented path matches the
// single-shot conv for output, final conv tail, and each boundary conv state.
func TestCausalConv1DSegmentEquivalence(t *testing.T) {
	skipIfNoMLX(t)
	B, L, D, convTail := 1, 4, 3, 2
	K := convTail + 1

	input := fromValues(0.5, B, L, D)
	prior := fromValues(-0.3, B, convTail, D)
	weight := fromValues(0.2, D, K)
	conv := convFromKernel(weight)

	full := &batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(L)}}

	refOut, refStates := CausalConv1D(full, input, conv, convTail, WithRecurrentState(prior, nil))
	if len(refStates) != 1 {
		t.Fatalf("unsegmented call returned %d states, want 1", len(refStates))
	}

	segOut, segStates := CausalConv1D(full, input, conv, convTail,
		WithRecurrentState(prior, nil), WithSnapshotSplits([]int{1, 2, 3}))
	mlx.Eval(refOut, segOut)

	floatsClose(t, "conv out", segOut.Floats(), refOut.Floats(), 1e-4)
	if len(segStates) != 4 {
		t.Fatalf("got %d boundary conv states, want 4", len(segStates))
	}
	mlx.Eval(lastState(segStates), lastState(refStates))
	floatsClose(t, "conv final", lastState(segStates).Floats(), lastState(refStates).Floats(), 1e-4)

	for i := range segStates {
		n := int32(i + 1)
		pb := &batch.Batch{SeqOffsets: []int32{0}, SeqQueryLens: []int32{n}}
		_, want := CausalConv1D(pb,
			mlx.SliceStartStop(input, []int32{0, 0, 0}, []int32{int32(B), n, int32(D)}),
			conv, convTail, WithRecurrentState(prior, nil))
		mlx.Eval(segStates[i], lastState(want))
		floatsClose(t, "boundary conv", segStates[i].Floats(), lastState(want).Floats(), 1e-4)
	}
}

// TestGatedDeltaSegmentEquivalenceBatched checks split forwards match the
// single-shot call for B>1 with a ragged batch, for both the per-token and
// sparse split patterns — the per-segment sliced mask must neutralize each
// row's padded positions so a short row's boundary state freezes at its
// real end.
func TestGatedDeltaSegmentEquivalenceBatched(t *testing.T) {
	skipIfNoMLX(t)
	B, T, Hk, Dk, Hv, Dv := 2, 4, 1, 32, 1, 32
	packed, ba, dtBias, aExp := gatedDeltaPackedInputs(B, T, Hk, Dk, Hv, Dv)
	prior := mlx.Zeros(mlx.DTypeFloat32, B, Hv, Dv, Dk)

	// Row 0 full length T; row 1 ends at 3 (so segment [3,4) is all padding
	// for row 1).
	rowReal := []int32{int32(T), 3}
	full := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, B, T),
		SeqOffsets:   []int32{0, 0},
		SeqQueryLens: rowReal,
	}

	refOut, refStates := GatedDelta(full, packed, ba, dtBias, aExp, WithRecurrentState(nil, prior))

	cases := []struct {
		name   string
		splits []int
	}{
		{"perToken", []int{1, 2, 3}},
		{"sparse", []int{2}},
	}
	for _, tc := range cases {
		segOut, segStates := GatedDelta(full, packed, ba, dtBias, aExp,
			WithRecurrentState(nil, prior), WithSnapshotSplits(tc.splits))
		mlx.Eval(refOut, segOut, lastState(refStates), lastState(segStates))
		floatsClose(t, tc.name+" batched out", segOut.Floats(), refOut.Floats(), 1e-4)
		floatsClose(t, tc.name+" batched final state", lastState(segStates).Floats(), lastState(refStates).Floats(), 1e-4)
		if len(segStates) != len(tc.splits)+1 {
			t.Fatalf("%s: got %d boundary states, want %d", tc.name, len(segStates), len(tc.splits)+1)
		}

		// Each row's boundary must equal a B=1 single-shot call over that
		// row's real prefix: row 0 advances the full length, row 1 freezes
		// once it reaches its real length.
		boundaries := append(append([]int{}, tc.splits...), T)
		for i, bound := range boundaries {
			for r := range B {
				n := min(int32(bound), rowReal[r])
				lo, hi := int32(r), int32(r)+1
				rowPrior := mlx.SliceStartStop(prior, []int32{lo, 0, 0, 0}, []int32{hi, int32(Hv), int32(Dv), int32(Dk)})
				_, want, _ := mlx.GatedDelta(
					slicePrefix(packed, lo, hi, n), slicePrefix(ba, lo, hi, n),
					dtBias, aExp, rowPrior, nil, false)
				gotRow := mlx.SliceStartStop(segStates[i], []int32{lo, 0, 0, 0}, []int32{hi, int32(Hv), int32(Dv), int32(Dk)})
				mlx.Eval(gotRow, want)
				floatsClose(t, tc.name+" batched boundary delta", gotRow.Floats(), want.Floats(), 1e-4)
			}
		}
	}
}

// TestCausalConv1DSegmentEquivalenceBatched is the conv analog of the gated-delta
// batched test: boundary tails from the single conv pass vs per-row single-shot
// references for a ragged B>1 batch, where a short row must freeze its tail at
// its real end rather than reach into padding.
func TestCausalConv1DSegmentEquivalenceBatched(t *testing.T) {
	skipIfNoMLX(t)
	B, L, D, convTail := 2, 4, 3, 2
	K := convTail + 1

	input := fromValues(0.5, B, L, D)
	prior := fromValues(-0.3, B, convTail, D)
	weight := fromValues(0.2, D, K)
	conv := convFromKernel(weight)

	full := &batch.Batch{SeqOffsets: []int32{0, 0}, SeqQueryLens: []int32{int32(L), 3}}

	refOut, refStates := CausalConv1D(full, input, conv, convTail, WithRecurrentState(prior, nil))
	segOut, segStates := CausalConv1D(full, input, conv, convTail,
		WithRecurrentState(prior, nil), WithSnapshotSplits([]int{1, 2, 3}))
	mlx.Eval(refOut, segOut, lastState(refStates), lastState(segStates))

	floatsClose(t, "batched conv out", segOut.Floats(), refOut.Floats(), 1e-4)
	floatsClose(t, "batched conv final", lastState(segStates).Floats(), lastState(refStates).Floats(), 1e-4)
	if len(segStates) != 4 {
		t.Fatalf("got %d boundary conv states, want 4", len(segStates))
	}

	// Each row's boundary i (offset i+1) must equal a B=1 single-shot conv over
	// that row's real prefix: row 0 advances the full length, row 1 freezes once
	// it reaches its real length 3. Per-row B=1 references avoid the ambiguity of
	// redeclaring a ragged length over a uniform input slice.
	rowReal := []int32{int32(L), 3}
	for i := range segStates {
		for r := range B {
			n := min(int32(i+1), rowReal[r])
			rowPrior := mlx.SliceStartStop(prior,
				[]int32{int32(r), 0, 0}, []int32{int32(r) + 1, int32(convTail), int32(D)})
			rowInput := mlx.SliceStartStop(input,
				[]int32{int32(r), 0, 0}, []int32{int32(r) + 1, n, int32(D)})
			_, want := CausalConv1D(&batch.Batch{}, rowInput, conv, convTail,
				WithRecurrentState(rowPrior, nil))
			gotRow := mlx.SliceStartStop(segStates[i],
				[]int32{int32(r), 0, 0}, []int32{int32(r) + 1, int32(convTail), int32(D)})
			mlx.Eval(gotRow, lastState(want))
			floatsClose(t, "batched boundary conv", gotRow.Floats(), lastState(want).Floats(), 1e-4)
		}
	}
}
