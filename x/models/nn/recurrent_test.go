package nn

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func ones(dtype mlx.DType, shape ...int) *mlx.Array {
	return mlx.AddScalar(mlx.Zeros(dtype, shape...), 1)
}

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

// depthwiseCausalRef is a Go-side reference for the depthwise causal
// 1D conv fallback. concat is [B, total, C], weight is [C, K], output
// is [B, total-K+1, C]. Used to anchor the wrapper's parity tests.
func depthwiseCausalRef(concat, weight *mlx.Array) []float32 {
	mlx.Eval(concat, weight)
	cVals := concat.Floats()
	wVals := weight.Floats()
	B := concat.Dim(0)
	total := concat.Dim(1)
	C := concat.Dim(2)
	K := weight.Dim(1)
	outLen := total - K + 1
	out := make([]float32, B*outLen*C)
	for bi := range B {
		for q := range outLen {
			for c := range C {
				var sum float32
				for k := range K {
					x := cVals[bi*total*C+(q+k)*C+c]
					w := wVals[c*K+k]
					sum += x * w
				}
				out[bi*outLen*C+q*C+c] = sum
			}
		}
	}
	return out
}

// TestCausalConv1DParity drives the wrapper with non-trivial prior,
// input, and weight values, then compares against a direct depthwise-
// causal-conv reference.
func TestCausalConv1DParity(t *testing.T) {
	skipIfNoMLX(t)
	B, L, D, convTail := 1, 4, 3, 2
	K := convTail + 1

	input := fromValues(0.5, B, L, D)
	prior := fromValues(-0.3, B, convTail, D)
	weight := fromValues(0.2, D, K)

	out, nextConv := CausalConv1D(&batch.Batch{}, input, nil, weight, convTail, WithRecurrentState(prior, nil))
	mlx.Eval(out, nextConv)

	concat := mlx.Concatenate([]*mlx.Array{prior, input}, 1)
	want := depthwiseCausalRef(concat, weight)
	got := out.Floats()
	if len(got) != len(want) {
		t.Fatalf("out len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]: got %v, want %v", i, got[i], want[i])
		}
	}

	// nextConv (no padding) is the trailing convTail rows of concat.
	mlx.Eval(concat)
	cVals := concat.Floats()
	total := concat.Dim(1)
	wantTail := make([]float32, B*convTail*D)
	for bi := range B {
		for k := range convTail {
			for d := range D {
				wantTail[bi*convTail*D+k*D+d] = cVals[bi*total*D+(total-convTail+k)*D+d]
			}
		}
	}
	tail := nextConv.Floats()
	if len(tail) != len(wantTail) {
		t.Fatalf("nextConv len = %d, want %d", len(tail), len(wantTail))
	}
	for i := range wantTail {
		if tail[i] != wantTail[i] {
			t.Fatalf("nextConv[%d]: got %v, want %v", i, tail[i], wantTail[i])
		}
	}
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

	out, nextConv := CausalConv1D(b, input, nil, weight, convTail, WithRecurrentState(priorFull, nil))
	mlx.Eval(out, nextConv)

	// Reference for row 0: B=1 unpadded length-L call.
	refOut0, refNextConv0 := CausalConv1D(&batch.Batch{},
		inputFull, nil, weight, convTail,
		WithRecurrentState(mlx.SliceStartStop(priorFull,
			[]int32{0, 0, 0},
			[]int32{1, int32(convTail), int32(D)}), nil))
	// Reference for row 1: B=1 unpadded length-qLenShort call.
	refOut1, refNextConv1 := CausalConv1D(&batch.Batch{},
		inputShortReal, nil, weight, convTail,
		WithRecurrentState(priorShort, nil))
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

func TestGatedDeltaZeroFallback(t *testing.T) {
	skipIfNoMLX(t)
	B, L, nK, nV, dK, dV := 1, 2, 1, 1, 4, 4
	q := ones(mlx.DTypeFloat32, B, L, nK, dK)
	k := ones(mlx.DTypeFloat32, B, L, nK, dK)
	v := ones(mlx.DTypeFloat32, B, L, nV, dV)
	gDecay := ones(mlx.DTypeFloat32, B, L, nV)
	beta := ones(mlx.DTypeFloat32, B, L, nV)

	zero := mlx.Zeros(mlx.DTypeFloat32, B, nV, dV, dK)
	outA, stateA := GatedDelta(&batch.Batch{}, q, k, v, gDecay, beta, WithRecurrentState(nil, zero))
	outB, stateB := mlx.FastGatedDelta(q, k, v, gDecay, beta, zero, nil)
	mlx.Eval(outA, stateA, outB, stateB)

	gotOut, wantOut := outA.Floats(), outB.Floats()
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("output[%d]: wrapper=%v direct=%v", i, gotOut[i], wantOut[i])
		}
	}
	gotState, wantState := stateA.Floats(), stateB.Floats()
	for i := range wantState {
		if gotState[i] != wantState[i] {
			t.Fatalf("state[%d]: wrapper=%v direct=%v", i, gotState[i], wantState[i])
		}
	}
}

func TestGatedDeltaUsesPriorState(t *testing.T) {
	skipIfNoMLX(t)
	B, L, nK, nV, dK, dV := 1, 2, 1, 1, 4, 4
	q := ones(mlx.DTypeFloat32, B, L, nK, dK)
	k := ones(mlx.DTypeFloat32, B, L, nK, dK)
	v := ones(mlx.DTypeFloat32, B, L, nV, dV)
	gDecay := ones(mlx.DTypeFloat32, B, L, nV)
	beta := ones(mlx.DTypeFloat32, B, L, nV)

	priorState := mlx.MulScalar(ones(mlx.DTypeFloat32, B, nV, dV, dK), 3)

	outA, _ := GatedDelta(&batch.Batch{}, q, k, v, gDecay, beta, WithRecurrentState(nil, priorState))
	outB, _ := mlx.FastGatedDelta(q, k, v, gDecay, beta, priorState, nil)
	mlx.Eval(outA, outB)

	gotOut, wantOut := outA.Floats(), outB.Floats()
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("output[%d]: wrapper=%v direct=%v", i, gotOut[i], wantOut[i])
		}
	}
}

// TestGatedDeltaPaddedRowParity drives a B=2 batch where row 1 is
// short (qLen < L). The wrapper must substitute neutral values
// (q=k=v=beta=0, g=1) at row 1's padded positions so the recurrence
// is a no-op there — and row 1's final state must equal the state
// after its last real token. Pinned via parity against a B=1 length-
// qLen call on the same row.
func TestGatedDeltaPaddedRowParity(t *testing.T) {
	skipIfNoMLX(t)
	L, nK, nV, dK, dV := 4, 1, 1, 4, 4
	qLenShort := 2

	makeRows := func(seedA, seedB float32, shape ...int) *mlx.Array {
		// Build a rank-(len(shape)+1) tensor with B=2 rows from two
		// distinct seeds so the rows are not accidentally identical.
		n := 1
		for _, d := range shape {
			n *= d
		}
		vals := make([]float32, 2*n)
		for i := range n {
			vals[i] = seedA + 0.1*float32(i)
		}
		for i := range n {
			vals[n+i] = seedB + 0.1*float32(i)
		}
		full := append([]int{2}, shape...)
		return mlx.FromValues(vals, full...)
	}

	q := makeRows(0.5, -0.5, L, nK, dK)
	k := makeRows(0.7, -0.7, L, nK, dK)
	v := makeRows(0.3, -0.3, L, nV, dV)
	gDecay := makeRows(0.1, -0.1, L, nV)
	beta := makeRows(0.4, -0.4, L, nV)
	priorState := makeRows(0.2, -0.2, nV, dV, dK)

	b := &batch.Batch{
		InputIDs:     mlx.Zeros(mlx.DTypeInt32, 2, L),
		SeqOffsets:   []int32{0, 0},
		SeqQueryLens: []int32{int32(L), int32(qLenShort)},
	}
	_, state := GatedDelta(b, q, k, v, gDecay, beta, WithRecurrentState(nil, priorState))
	mlx.Eval(state)

	// Reference for row 1: B=1 length-qLenShort call against the
	// row's real prefix and its prior state slice.
	row1Slice := func(a *mlx.Array, axisLens ...int32) *mlx.Array {
		dims := a.Dims()
		start := make([]int32, len(dims))
		stop := make([]int32, len(dims))
		start[0], stop[0] = 1, 2
		for i := 1; i < len(dims); i++ {
			stop[i] = int32(dims[i])
		}
		// Optionally truncate axis 1 (sequence axis) to qLenShort.
		if len(axisLens) >= 1 && len(dims) >= 2 {
			stop[1] = axisLens[0]
		}
		return mlx.SliceStartStop(a, start, stop)
	}
	q1 := row1Slice(q, int32(qLenShort))
	k1 := row1Slice(k, int32(qLenShort))
	v1 := row1Slice(v, int32(qLenShort))
	gDecay1 := row1Slice(gDecay, int32(qLenShort))
	beta1 := row1Slice(beta, int32(qLenShort))
	priorRow1 := row1Slice(priorState)

	_, refState := mlx.FastGatedDelta(q1, k1, v1, gDecay1, beta1, priorRow1, nil)
	mlx.Eval(refState)

	gotState := state.Floats()
	wantState := refState.Floats()
	row1Stride := nV * dV * dK
	for i := range row1Stride {
		gotV := gotState[row1Stride+i]
		wantV := wantState[i]
		if math.Abs(float64(gotV-wantV)) > 1e-4 {
			t.Fatalf("row 1 final state[%d]: got %v, want %v", i, gotV, wantV)
		}
	}
}
