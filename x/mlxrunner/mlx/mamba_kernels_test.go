package mlx

import (
	"math"
	"testing"
)

func TestMambaGatedGroupRMSNormMatchesFallback(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("MLX Metal not available")
		}

		x := testArrayValues(0.1, 2, 3, 8)
		gate := testArrayValues(-0.2, 2, 3, 8)
		weight := testArrayValues(0.9, 8)
		got, ok := FastMambaGatedGroupRMSNorm(x, gate, weight, 2, 1e-5, DTypeFloat32)
		if !ok {
			t.Fatal("FastMambaGatedGroupRMSNorm returned ok=false")
		}
		want := mambaGatedGroupRMSNormReference(x, gate, weight, 2, 1e-5)
		assertArrayClose(t, "gated group rmsnorm", got, want, 1e-5)
	})
}

func TestMamba2ScanMatchesReference(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("MLX Metal not available")
		}

		hidden := testArrayValues(0.1, 1, 3, 2, 2)
		bState := testArrayValues(0.2, 1, 3, 2, 32)
		cState := testArrayValues(0.3, 1, 3, 2, 32)
		dt := testArrayValues(-0.4, 1, 3, 2)
		state := testArrayValues(0.5, 1, 2, 2, 32)
		a := MulScalar(onesTest(DTypeFloat32, 2), -0.25)
		d := MulScalar(onesTest(DTypeFloat32, 2), 0.1)
		dtBias := Zeros(DTypeFloat32, 2)

		gotY, gotState, ok := FastMamba2Scan(hidden, bState, cState, dt, state, a, d, dtBias)
		if !ok {
			t.Fatal("FastMamba2Scan returned ok=false")
		}
		wantY, wantState := mamba2ScanReference(hidden, bState, cState, dt, state, a, d, dtBias)
		assertArrayClose(t, "mamba2 scan y", gotY, wantY, 1e-5)
		assertArrayClose(t, "mamba2 scan state", gotState, wantState, 1e-5)
	})
}

func TestMamba2ScanWithSnapshotMatchesSegmentedReference(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("MLX Metal not available")
		}

		hidden := testArrayValues(0.1, 1, 4, 2, 2)
		bState := testArrayValues(0.2, 1, 4, 2, 32)
		cState := testArrayValues(0.3, 1, 4, 2, 32)
		dt := testArrayValues(-0.4, 1, 4, 2)
		state := testArrayValues(0.5, 1, 2, 2, 32)
		a := MulScalar(onesTest(DTypeFloat32, 2), -0.25)
		d := MulScalar(onesTest(DTypeFloat32, 2), 0.1)
		dtBias := Zeros(DTypeFloat32, 2)

		gotY, gotEnd, gotSnapshot, ok := FastMamba2ScanWithSnapshot(hidden, bState, cState, dt, state, a, d, dtBias, 2)
		if !ok {
			t.Fatal("FastMamba2ScanWithSnapshot returned ok=false")
		}

		prefixY, wantSnapshot := mamba2ScanReference(
			sliceTimeRange(hidden, 0, 2),
			sliceTimeRange(bState, 0, 2),
			sliceTimeRange(cState, 0, 2),
			sliceTimeRange(dt, 0, 2),
			state,
			a,
			d,
			dtBias,
		)
		suffixY, wantEnd := mamba2ScanReference(
			sliceTimeRange(hidden, 2, 4),
			sliceTimeRange(bState, 2, 4),
			sliceTimeRange(cState, 2, 4),
			sliceTimeRange(dt, 2, 4),
			wantSnapshot,
			a,
			d,
			dtBias,
		)
		wantY := Concatenate([]*Array{prefixY, suffixY}, 1)

		assertArrayClose(t, "mamba2 snapshot y", gotY, wantY, 1e-5)
		assertArrayClose(t, "mamba2 snapshot state", gotEnd, wantEnd, 1e-5)
		assertArrayClose(t, "mamba2 snapshot boundary", gotSnapshot, wantSnapshot, 1e-5)
	})
}

func TestMamba2ScanGroupedStatesMatchRepeatedReference(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("MLX Metal not available")
		}

		hidden := testArrayValues(0.1, 1, 2, 4, 2)
		bGrouped := testArrayValues(0.2, 1, 2, 2, 32)
		cGrouped := testArrayValues(0.3, 1, 2, 2, 32)
		dt := testArrayValues(-0.4, 1, 2, 4)
		state := testArrayValues(0.5, 1, 4, 2, 32)
		a := MulScalar(onesTest(DTypeFloat32, 4), -0.25)
		d := MulScalar(onesTest(DTypeFloat32, 4), 0.1)
		dtBias := Zeros(DTypeFloat32, 4)

		gotY, gotState, ok := FastMamba2Scan(hidden, bGrouped, cGrouped, dt, state, a, d, dtBias)
		if !ok {
			t.Fatal("FastMamba2Scan returned ok=false")
		}
		wantY, wantState := mamba2ScanReference(hidden, repeatMambaGroupsForTest(bGrouped, 2), repeatMambaGroupsForTest(cGrouped, 2), dt, state, a, d, dtBias)
		assertArrayClose(t, "grouped mamba2 y", gotY, wantY, 1e-5)
		assertArrayClose(t, "grouped mamba2 state", gotState, wantState, 1e-5)
	})
}

func TestMamba2ScanRejectsUnsupportedShape(t *testing.T) {
	withMLXThread(t, func() {
		hidden := Zeros(DTypeFloat32, 1, 1, 1, 1)
		bState := Zeros(DTypeFloat32, 1, 1, 1, 31)
		cState := Zeros(DTypeFloat32, 1, 1, 1, 31)
		dt := Zeros(DTypeFloat32, 1, 1, 1)
		state := Zeros(DTypeFloat32, 1, 1, 1, 31)
		a := Zeros(DTypeFloat32, 1)
		d := Zeros(DTypeFloat32, 1)
		dtBias := Zeros(DTypeFloat32, 1)
		if _, _, ok := FastMamba2Scan(hidden, bState, cState, dt, state, a, d, dtBias); ok {
			t.Fatal("FastMamba2Scan ok=true for unsupported S=31 shape")
		}
	})
}

func TestMoEWeightedSumMatchesFallback(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("MLX Metal not available")
		}

		expert := testArrayValues(0.1, 2, 3, 4, 5)
		scores := testArrayValues(0.2, 2, 3, 4)
		got, ok := FastMoEWeightedSum(expert, scores, DTypeFloat32)
		if !ok {
			t.Fatal("FastMoEWeightedSum returned ok=false")
		}
		want := Sum(Mul(expert, ExpandDims(scores.AsType(expert.DType()), -1)), 2, false).AsType(DTypeFloat32)
		assertArrayClose(t, "moe weighted sum", got, want, 1e-5)
	})
}

func testArrayValues(seed float32, shape ...int) *Array {
	n := 1
	for _, d := range shape {
		n *= d
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = seed + 0.001*float32(i)
	}
	return FromValues(vals, shape...)
}

func onesTest(dtype DType, shape ...int) *Array {
	return AddScalar(Zeros(dtype, shape...), 1)
}

func assertArrayClose(t *testing.T, name string, got, want *Array, tol float64) {
	t.Helper()
	Eval(got, want)
	gotF := got.Floats()
	wantF := want.Floats()
	if len(gotF) != len(wantF) {
		t.Fatalf("%s length = %d, want %d", name, len(gotF), len(wantF))
	}
	for i := range gotF {
		if math.Abs(float64(gotF[i]-wantF[i])) > tol {
			t.Fatalf("%s[%d] = %v, want %v", name, i, gotF[i], wantF[i])
		}
	}
}

func mambaGatedGroupRMSNormReference(x, gate, weight *Array, groups int, eps float32) *Array {
	dims := x.Dims()
	B, L, inner := int32(dims[0]), int32(dims[1]), int32(dims[2])
	groupSize := inner / int32(groups)
	y := Mul(x, SiLU(gate.AsType(x.DType())))
	y = Reshape(y, B, L, int32(groups), groupSize)
	variance := Mean(Mul(y, y), 3, true)
	y = Mul(y, RSqrt(AddScalar(variance, eps)))
	w := Reshape(weight.AsType(y.DType()), 1, 1, int32(groups), groupSize)
	y = Mul(y, w)
	return Reshape(y, B, L, inner)
}

func mamba2ScanReference(hidden, bState, cState, dt, state, a, d, dtBias *Array) (*Array, *Array) {
	B := int32(hidden.Dim(0))
	T := int32(hidden.Dim(1))
	H := int32(hidden.Dim(2))
	D := int32(hidden.Dim(3))

	a = Reshape(a, 1, H, 1, 1)
	d = Reshape(d, 1, H, 1)
	outs := make([]*Array, 0, T)
	for t := range T {
		xt := sliceTimeForTest(hidden, t).AsType(DTypeFloat32)
		bt := sliceTimeForTest(bState, t).AsType(DTypeFloat32)
		ct := sliceTimeForTest(cState, t).AsType(DTypeFloat32)
		dtt := Add(sliceTimeForTest(dt, t).AsType(DTypeFloat32), dtBias)
		dtt = Log(AddScalar(Exp(dtt), 1))
		dA := Exp(Mul(Reshape(dtt, B, H, 1, 1), a))
		dB := Mul(Reshape(dtt, B, H, 1), bt)
		state = Add(Mul(state, dA), Mul(ExpandDims(xt, -1), ExpandDims(dB, 2)))
		y := Sum(Mul(state, ExpandDims(ct, 2)), 3, false)
		y = Add(y, Mul(xt, d))
		outs = append(outs, Reshape(y, B, H, D))
	}
	return Stack(outs, 1), state
}

func sliceTimeForTest(x *Array, t int32) *Array {
	dims := x.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[1] = t
	stop[1] = t + 1
	return Squeeze(SliceStartStop(x, start, stop), 1)
}

func sliceTimeRange(x *Array, start, stop int32) *Array {
	dims := x.Dims()
	starts := make([]int32, len(dims))
	stops := make([]int32, len(dims))
	for i, d := range dims {
		stops[i] = int32(d)
	}
	starts[1] = start
	stops[1] = stop
	return SliceStartStop(x, starts, stops)
}

func repeatMambaGroupsForTest(x *Array, repeats int32) *Array {
	if repeats <= 1 {
		return x
	}
	dims := x.Dims()
	x = ExpandDims(x, 3)
	x = Tile(x, []int32{1, 1, 1, repeats, 1})
	return Reshape(x, int32(dims[0]), int32(dims[1]), int32(dims[2])*repeats, int32(dims[3]))
}
