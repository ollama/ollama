package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthreadtest"
)

// fp4Values decodes an fp4 (E2M1) code to its value.
var fp4Values = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

func TestDequantizeGlobalScale(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		testDequantizeGlobalScale(t)
	})
}

// The quantized payload is built directly, the way an nvfp4 checkpoint ships
// it: packed fp4 codes, e4m3 group-scale bytes, and a separate global scale.
// Only the dequantize consumer path runs, so expectations are exact.
func testDequantizeGlobalScale(t *mlxthreadtest.T) {
	const rows, cols, group = 4, 64, 16
	// Every group cycles through all 16 codes; group g of row r has scale
	// 2^((r+g)%4-1), a power of two so every expected product is exact.
	scaleOf := func(r, g int) float32 {
		return float32(math.Ldexp(1, (r+g)%4-1))
	}

	packed := make([]uint32, rows*cols/8)
	for i := range packed {
		for j := range 8 {
			packed[i] |= uint32((i*8+j)%16) << (4 * j)
		}
	}
	scaleBits := make([]uint8, rows*(cols/group))
	for i := range scaleBits {
		exp := (i/(cols/group)+i%(cols/group))%4 - 1
		scaleBits[i] = uint8((exp + 7) << 3)
	}
	wq := FromValues(packed, rows, cols/8)
	scales := FromValues(scaleBits, rows, cols/group)

	check := func(name string, got *Array, gs func(r int) float32) {
		t.Helper()
		g32 := got.AsType(DTypeFloat32)
		Eval(g32)
		values := g32.Floats()
		if len(values) != rows*cols {
			t.Errorf("%s: length = %d, want %d", name, len(values), rows*cols)
			return
		}
		for i, v := range values {
			r, c := i/cols, i%cols
			if want := fp4Values[c%16] * scaleOf(r, c/group) * gs(r); v != want {
				t.Errorf("%s[%d] = %v, want %v", name, i, v, want)
				return
			}
		}
	}

	base := Dequantize(wq, scales, nil, group, 4, "nvfp4", nil)
	check("no scale", base, func(int) float32 { return 1 })

	perRow := []float32{0.5, 1, 2, 4}
	cases := []struct {
		name  string
		scale *Array
		gs    func(r int) float32
	}{
		{"scalar", FromValues([]float32{2}, 1), func(int) float32 { return 2 }},
		{"perRow", FromValues(perRow, rows), func(r int) float32 { return perRow[r] }},
	}
	for _, tc := range cases {
		got := Dequantize(wq, scales, nil, group, 4, "nvfp4", tc.scale)
		if got.DType() != base.DType() {
			t.Errorf("%s: dtype = %v, want %v", tc.name, got.DType(), base.DType())
		}
		check(tc.name, got, tc.gs)
	}

	const experts = 2
	expertPacked := make([]uint32, experts*len(packed))
	expertScales := make([]uint8, experts*len(scaleBits))
	for e := range experts {
		copy(expertPacked[e*len(packed):], packed)
		copy(expertScales[e*len(scaleBits):], scaleBits)
	}
	expertWeights := FromValues(expertPacked, experts, rows, cols/8)
	expertBlockScales := FromValues(expertScales, experts, rows, cols/group)
	expertGlobalScales := []float32{0.5, 2}
	expertOut := Dequantize(
		expertWeights,
		expertBlockScales,
		nil,
		group,
		4,
		"nvfp4",
		FromValues(expertGlobalScales, experts),
	).AsType(DTypeFloat32)
	Eval(expertOut)
	for i, got := range expertOut.Floats() {
		e := i / (rows * cols)
		r := (i / cols) % rows
		c := i % cols
		want := fp4Values[c%16] * scaleOf(r, c/group) * expertGlobalScales[e]
		if got != want {
			t.Fatalf("expert bank[%d] = %v, want %v", i, got, want)
		}
	}
}
