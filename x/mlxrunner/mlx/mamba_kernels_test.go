package mlx

import (
	"fmt"
	"math"
	"testing"
)

func TestMamba2ScanMatchesReference(t *testing.T) {
	requireMamba2Metal(t)
	var failures []error
	withMLXThread(t, func() {
		in := newMamba2TestInputs(1, 3, 2, 2, 2, 32)
		gotY, gotState, interior := Mamba2Scan(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias, nil, false)
		if len(interior) != 0 {
			failures = append(failures, fmt.Errorf("interior states = %d, want 0 without captureAll", len(interior)))
		}
		wantY, wantState := mamba2ScanReference(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias)
		failures = appendArrayCloseError(failures, "mamba2 scan y", gotY, wantY, 1e-5)
		failures = appendArrayCloseError(failures, "mamba2 scan state", gotState, wantState, 1e-5)
	})
	reportMamba2Failures(t, failures)
}

// Every interior state must match, not just one boundary.
func TestMamba2ScanCaptureAllMatchesPerTokenReference(t *testing.T) {
	requireMamba2Metal(t)
	var failures []error
	withMLXThread(t, func() {
		const T = 4
		in := newMamba2TestInputs(1, T, 2, 2, 2, 32)
		gotY, gotEnd, gotInterior := Mamba2Scan(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias, nil, true)
		if len(gotInterior) != T-1 {
			failures = append(failures, fmt.Errorf("interior states = %d, want %d", len(gotInterior), T-1))
		}

		state := in.state
		for ti := range int32(T) {
			var y *Array
			y, state = mamba2ScanReference(
				sliceTimeRange(in.hidden, ti, ti+1),
				sliceTimeRange(in.bState, ti, ti+1),
				sliceTimeRange(in.cState, ti, ti+1),
				sliceTimeRange(in.dt, ti, ti+1),
				state, in.a, in.d, in.dtBias,
			)
			failures = appendArrayCloseError(failures, fmt.Sprintf("token %d y", ti), sliceTimeRange(gotY, ti, ti+1), y, 1e-5)
			if int(ti) < T-1 && int(ti) < len(gotInterior) {
				failures = appendArrayCloseError(failures, fmt.Sprintf("token %d interior state", ti), gotInterior[ti], state, 1e-5)
			}
		}
		failures = appendArrayCloseError(failures, "captureAll end state", gotEnd, state, 1e-5)
	})
	reportMamba2Failures(t, failures)
}

func TestMamba2ScanGroupedStatesMatchRepeatedReference(t *testing.T) {
	requireMamba2Metal(t)
	var failures []error
	withMLXThread(t, func() {
		in := newMamba2TestInputs(1, 2, 4, 2, 2, 32)
		gotY, gotState, _ := Mamba2Scan(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias, nil, false)
		wantY, wantState := mamba2ScanReference(
			in.hidden,
			repeatMambaGroupsForTest(in.bState, 2),
			repeatMambaGroupsForTest(in.cState, 2),
			in.dt, in.state, in.a, in.d, in.dtBias,
		)
		failures = appendArrayCloseError(failures, "grouped mamba2 y", gotY, wantY, 1e-5)
		failures = appendArrayCloseError(failures, "grouped mamba2 state", gotState, wantState, 1e-5)
	})
	reportMamba2Failures(t, failures)
}

// A shape outside the kernel's contract must still compute the right answer
// through the graph implementation rather than fail.
func TestMamba2ScanUnsupportedShapeMatchesGraph(t *testing.T) {
	var failures []error
	withMLXThread(t, func() {
		in := newMamba2TestInputs(1, 2, 2, 2, 2, 31)
		if _, ok := resolveMamba2ScanDims(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias); ok {
			failures = append(failures, fmt.Errorf("resolveMamba2ScanDims ok=true for unsupported S=31 shape"))
		}
		gotY, gotState, _ := Mamba2Scan(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias, nil, false)
		wantY, wantState := mamba2ScanReference(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias)
		failures = appendArrayCloseError(failures, "unsupported-shape y", gotY, wantY, 1e-5)
		failures = appendArrayCloseError(failures, "unsupported-shape state", gotState, wantState, 1e-5)
	})
	reportMamba2Failures(t, failures)
}

// A padded position must be an identity step. Without the mask it still decays
// the state by exp(dt*a).
func TestMamba2ScanPaddedRowIsIdentity(t *testing.T) {
	requireMamba2Metal(t)
	var failures []error
	withMLXThread(t, func() {
		const (
			B = 2
			L = 3
			H = 2
			D = 2
			S = 32
		)
		in := newMamba2TestInputs(B, L, H, H, D, S)
		mask := FromValues([]bool{true, true, true, true, false, false}, B, L)

		gotY, gotState, _ := Mamba2Scan(in.hidden, in.bState, in.cState, in.dt, in.state, in.a, in.d, in.dtBias, mask, false)

		for row, realLen := range []int32{L, 1} {
			r := int32(row)
			wantY, wantState := mamba2ScanReference(
				sliceRowTime(in.hidden, r, realLen),
				sliceRowTime(in.bState, r, realLen),
				sliceRowTime(in.cState, r, realLen),
				sliceRowTime(in.dt, r, realLen),
				sliceRow(in.state, r), in.a, in.d, in.dtBias,
			)
			failures = appendArrayCloseError(failures, fmt.Sprintf("row %d y", row), sliceRowTime(gotY, r, realLen), wantY, 1e-5)
			failures = appendArrayCloseError(failures, fmt.Sprintf("row %d state", row), sliceRow(gotState, r), wantState, 1e-5)
		}

		pad := SliceStartStop(gotY, []int32{1, 1, 0, 0}, []int32{2, L, H, D})
		Eval(pad)
		for i, v := range pad.Floats() {
			if v != 0 {
				failures = append(failures, fmt.Errorf("padded output[%d] = %v, want 0", i, v))
			}
		}
	})
	reportMamba2Failures(t, failures)
}

type mamba2TestInputs struct {
	hidden, bState, cState, dt, state, a, d, dtBias *Array
}

func newMamba2TestInputs(B, T, H, G, D, S int) mamba2TestInputs {
	return mamba2TestInputs{
		hidden: testArrayValues(0.1, B, T, H, D),
		bState: testArrayValues(0.2, B, T, G, S),
		cState: testArrayValues(0.3, B, T, G, S),
		dt:     testArrayValues(-0.4, B, T, H),
		state:  testArrayValues(0.5, B, H, D, S),
		a:      MulScalar(onesTest(DTypeFloat32, H), -0.25),
		d:      MulScalar(onesTest(DTypeFloat32, H), 0.1),
		dtBias: Zeros(DTypeFloat32, H),
	}
}

func sliceRow(x *Array, r int32) *Array {
	dims := x.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[0], stop[0] = r, r+1
	return SliceStartStop(x, start, stop)
}

func sliceRowTime(x *Array, r, realLen int32) *Array {
	dims := x.Dims()
	start := make([]int32, len(dims))
	stop := make([]int32, len(dims))
	for i, d := range dims {
		stop[i] = int32(d)
	}
	start[0], stop[0] = r, r+1
	stop[1] = realLen
	return SliceStartStop(x, start, stop)
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

func requireMamba2Metal(t *testing.T) {
	t.Helper()
	skipIfNoMLX(t)
	if !MetalIsAvailable() {
		t.Skip("MLX Metal not available")
	}
}

func appendArrayCloseError(failures []error, name string, got, want *Array, tol float64) []error {
	Eval(got, want)
	gotF := got.Floats()
	wantF := want.Floats()
	if len(gotF) != len(wantF) {
		return append(failures, fmt.Errorf("%s length = %d, want %d", name, len(gotF), len(wantF)))
	}
	for i := range gotF {
		if math.Abs(float64(gotF[i]-wantF[i])) > tol {
			return append(failures, fmt.Errorf("%s[%d] = %v, want %v", name, i, gotF[i], wantF[i]))
		}
	}
	return failures
}

func reportMamba2Failures(t *testing.T, failures []error) {
	t.Helper()
	for _, err := range failures {
		t.Error(err)
	}
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
