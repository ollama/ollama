package mlx

import (
	"fmt"
	"math"
	"testing"
)

type gatedDeltaTestGeometry struct {
	Hk, Dk, Hv, Dv int
}

func (g gatedDeltaTestGeometry) packedDim() int { return 2*g.Hk*g.Dk + g.Hv*g.Dv }

type gatedDeltaTestInputs struct {
	packed, ba, dtBias, aExp, state *Array
}

func gatedDeltaTestInputs36(g gatedDeltaTestGeometry, T int) gatedDeltaTestInputs {
	return gatedDeltaTestInputs{
		packed: patternArray(DTypeBFloat16, []int{1, T, g.packedDim()}, 0.01, 0.003, 37, 257),
		ba:     patternArray(DTypeBFloat16, []int{1, T, 2 * g.Hv}, 0, 0.4, 13, 37),
		dtBias: patternArray(DTypeBFloat16, []int{g.Hv}, -2.5, 0.08, 5, 29),
		aExp:   patternArray(DTypeFloat32, []int{g.Hv}, 0.12, 0.002, 3, 19),
		state:  patternArray(DTypeFloat32, []int{1, g.Hv, g.Dv, g.Dk}, 0, 0.0001, 17, 101),
	}
}

// gatedDeltaReference runs the graph implementation the fused kernels fall
// back to; the test pins the kernels against it bit-for-bit.
func gatedDeltaReference(in gatedDeltaTestInputs, captureAll bool) (y, nextState *Array, interior []*Array) {
	return gatedDeltaGraph(in.packed, in.ba, in.dtBias, in.aExp, in.state, captureAll)
}

// Exactness is per compiler pair: the metallib and the runtime JIT round
// metal::exp a float ulp apart, which after bf16 rounding leaves rare
// differing inputs (beta sigmoid at -6.84375); the lattice here avoids them.
func TestGatedDeltaMatchesGraph(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		testGatedDeltaMatchesGraph(t)
	})
}

func testGatedDeltaMatchesGraph(t *testing.T) {
	geometries := []gatedDeltaTestGeometry{
		{Hk: 16, Dk: 128, Hv: 32, Dv: 128},
		{Hk: 4, Dk: 64, Hv: 8, Dv: 32},
	}
	for _, g := range geometries {
		for T := 1; T <= gatedDeltaMaxTokens; T++ {
			for _, B := range []int{1, 3} {
				for _, captureAll := range []bool{false, true} {
					name := fmt.Sprintf("hk%d_dk%d_T%d_B%d_all%v", g.Hk, g.Dk, T, B, captureAll)
					in := batchGatedDeltaRows(gatedDeltaTestInputs36(g, T), B)
					refY, refState, refInterior := gatedDeltaReference(in, captureAll)
					y, state, interior := GatedDelta(in.packed, in.ba, in.dtBias, in.aExp, in.state, nil, captureAll)
					if err := requireExact("y", y, refY); err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					if err := requireExact("state", state, refState); err != nil {
						t.Fatalf("%s: %v", name, err)
					}
					if captureAll && len(interior) != len(refInterior) {
						t.Fatalf("%s: interior count = %d, want %d", name, len(interior), len(refInterior))
					}
					for i := range interior {
						if err := requireExact(fmt.Sprintf("interior[%d]", i), interior[i], refInterior[i]); err != nil {
							t.Fatalf("%s: %v", name, err)
						}
					}
				}
			}
		}
	}
}

func TestGatedDeltaGraphRouting(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		testGatedDeltaGraphRouting(t)
	})
}

// Contract misses — T beyond the kernel cap and a float32 packed input —
// run the same step as graph ops.
func testGatedDeltaGraphRouting(t *testing.T) {
	g := gatedDeltaTestGeometry{Hk: 4, Dk: 64, Hv: 8, Dv: 32}

	check := func(name string, in gatedDeltaTestInputs, captureAll bool) {
		y, end, interior := GatedDelta(in.packed, in.ba, in.dtBias, in.aExp, in.state, nil, captureAll)
		refY, refEnd, refInterior := gatedDeltaReference(in, captureAll)
		if err := requireExact("y", y, refY); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := requireExact("state", end, refEnd); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(interior) != len(refInterior) {
			t.Fatalf("%s: interior count = %d, want %d", name, len(interior), len(refInterior))
		}
	}

	check("tooLong", gatedDeltaTestInputs36(g, gatedDeltaMaxTokens+1), true)

	f32 := gatedDeltaTestInputs36(g, 2)
	f32.packed = f32.packed.AsType(DTypeFloat32)
	check("float32", f32, false)
}

// batchGatedDeltaRows widens a single-row input to B rows with distinct
// per-row contents.
func batchGatedDeltaRows(row gatedDeltaTestInputs, B int) gatedDeltaTestInputs {
	if B == 1 {
		return row
	}
	packed := make([]*Array, B)
	ba := make([]*Array, B)
	state := make([]*Array, B)
	for i := range B {
		r := row
		if i > 0 {
			r = scaledGatedDeltaRow(row, 1-0.3*float32(i))
		}
		packed[i], ba[i], state[i] = r.packed, r.ba, r.state
	}
	return gatedDeltaTestInputs{
		packed: Concatenate(packed, 0),
		ba:     Concatenate(ba, 0),
		dtBias: row.dtBias,
		aExp:   row.aExp,
		state:  Concatenate(state, 0),
	}
}

func scaledGatedDeltaRow(base gatedDeltaTestInputs, scale float32) gatedDeltaTestInputs {
	return gatedDeltaTestInputs{
		packed: MulScalar(base.packed, scale),
		ba:     MulScalar(base.ba, -scale),
		dtBias: base.dtBias,
		aExp:   base.aExp,
		state:  MulScalar(base.state, scale),
	}
}

func TestGatedDeltaBatchedRows(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		testGatedDeltaBatchedRows(t)
	})
}

// Each batched row must match its own single-row launch bit-for-bit.
func testGatedDeltaBatchedRows(t *testing.T) {
	g := gatedDeltaTestGeometry{Hk: 4, Dk: 64, Hv: 8, Dv: 32}
	for _, T := range []int{1, 5, 11} {
		rows := []gatedDeltaTestInputs{gatedDeltaTestInputs36(g, T)}
		rows = append(rows, scaledGatedDeltaRow(rows[0], 0.7), scaledGatedDeltaRow(rows[0], 0.4))

		packed := Concatenate([]*Array{rows[0].packed, rows[1].packed, rows[2].packed}, 0)
		ba := Concatenate([]*Array{rows[0].ba, rows[1].ba, rows[2].ba}, 0)
		state := Concatenate([]*Array{rows[0].state, rows[1].state, rows[2].state}, 0)

		y, end, _ := GatedDelta(packed, ba, rows[0].dtBias, rows[0].aExp, state, nil, false)
		for i, row := range rows {
			refY, refEnd, _ := GatedDelta(row.packed, row.ba, row.dtBias, row.aExp, row.state, nil, false)
			gotY := SliceStartStop(y,
				[]int32{int32(i), 0, 0, 0},
				[]int32{int32(i) + 1, int32(T), int32(g.Hv), int32(g.Dv)})
			gotEnd := SliceStartStop(end,
				[]int32{int32(i), 0, 0, 0},
				[]int32{int32(i) + 1, int32(g.Hv), int32(g.Dv), int32(g.Dk)})
			if err := requireExact("y", gotY, refY); err != nil {
				t.Fatalf("T=%d row %d: %v", T, i, err)
			}
			if err := requireExact("state", gotEnd, refEnd); err != nil {
				t.Fatalf("T=%d row %d: %v", T, i, err)
			}
		}
	}
}

func TestGatedDeltaRaggedRows(t *testing.T) {
	skipIfNoMLX(t)
	withMLXThread(t, func() {
		testGatedDeltaRaggedRows(t)
	})
}

// A padded row neutralized the way the nn seam does it — zeroed conv rows,
// ba tail poisoned to -inf — runs identity steps with zero output there: its
// full output and final state must match a launch of only its real tokens
// with zeros appended, while the full-length row is unaffected.
func testGatedDeltaRaggedRows(t *testing.T) {
	g := gatedDeltaTestGeometry{Hk: 4, Dk: 64, Hv: 8, Dv: 32}
	const T, realLen = 6, 4
	row0 := gatedDeltaTestInputs36(g, T)
	row1 := scaledGatedDeltaRow(row0, 0.7)

	pad := make([]float32, (T-realLen)*2*g.Hv)
	negInf := float32(math.Inf(-1))
	for i := range pad {
		pad[i] = negInf
	}
	row1BA := Concatenate([]*Array{
		SliceStartStop(row1.ba, []int32{0, 0, 0}, []int32{1, realLen, int32(2 * g.Hv)}),
		FromValues(pad, 1, T-realLen, 2*g.Hv).AsType(DTypeBFloat16),
	}, 1)
	row1Packed := Concatenate([]*Array{
		SliceStartStop(row1.packed, []int32{0, 0, 0}, []int32{1, realLen, int32(g.packedDim())}),
		Zeros(DTypeBFloat16, 1, T-realLen, g.packedDim()),
	}, 1)

	packed := Concatenate([]*Array{row0.packed, row1Packed}, 0)
	ba := Concatenate([]*Array{row0.ba, row1BA}, 0)
	state := Concatenate([]*Array{row0.state, row1.state}, 0)

	y, end, _ := GatedDelta(packed, ba, row0.dtBias, row0.aExp, state, nil, false)

	ref0Y, ref0End, _ := GatedDelta(row0.packed, row0.ba, row0.dtBias, row0.aExp, row0.state, nil, false)
	got0Y := SliceStartStop(y, []int32{0, 0, 0, 0}, []int32{1, T, int32(g.Hv), int32(g.Dv)})
	got0End := SliceStartStop(end, []int32{0, 0, 0, 0}, []int32{1, int32(g.Hv), int32(g.Dv), int32(g.Dk)})
	if err := requireExact("row0 y", got0Y, ref0Y); err != nil {
		t.Fatal(err)
	}
	if err := requireExact("row0 state", got0End, ref0End); err != nil {
		t.Fatal(err)
	}

	prefixPacked := SliceStartStop(row1.packed, []int32{0, 0, 0}, []int32{1, realLen, int32(g.packedDim())})
	prefixBA := SliceStartStop(row1.ba, []int32{0, 0, 0}, []int32{1, realLen, int32(2 * g.Hv)})
	ref1Y, ref1End, _ := GatedDelta(prefixPacked, prefixBA, row1.dtBias, row1.aExp, row1.state, nil, false)
	want1Y := Concatenate([]*Array{ref1Y, Zeros(DTypeBFloat16, 1, T-realLen, g.Hv, g.Dv)}, 1)
	got1Y := SliceStartStop(y, []int32{1, 0, 0, 0}, []int32{2, T, int32(g.Hv), int32(g.Dv)})
	got1End := SliceStartStop(end, []int32{1, 0, 0, 0}, []int32{2, int32(g.Hv), int32(g.Dv), int32(g.Dk)})
	if err := requireExact("row1 y", got1Y, want1Y); err != nil {
		t.Fatal(err)
	}
	if err := requireExact("row1 state", got1End, ref1End); err != nil {
		t.Fatal(err)
	}
}
