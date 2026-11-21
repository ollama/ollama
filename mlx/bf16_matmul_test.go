package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestMatmulBF16Weight(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !SupportsBF16WeightMatmul() {
			t.Skip("requires a qualified BF16 projection")
		}
		// At this output width stock MLX selects its FP32 matrix reduction.
		// Fractional inputs distinguish it from a lower-precision accumulation.
		const width, columns = 2304, 4090
		weights := make([]float32, columns*width)
		for i := range weights {
			weights[i] = float32(math.Cos(float64(i) * 0.0237))
		}
		w := FromValues(weights, columns, width).AsType(DTypeBFloat16)
		stock := w.AsType(DTypeFloat32).Transpose(1, 0)
		for _, rows := range []int{1, 7, 8, 14, 16} {
			values := make([]float32, rows*(width+4))
			for i := range values {
				values[i] = float32(math.Sin(float64(i) * 0.15891))
			}
			x := FromValues(values, rows, width+4).Slice(Slice(), Slice(4, width+4))
			got := MatmulBF16Weight(x, w)
			if got == nil {
				t.Fatalf("declined %d rows", rows)
			}
			// The SIMD matrix contracts eight rows, including the short final tile.
			padded := x
			if rows%8 != 0 {
				padded = PadConstant(x, []int{0}, []int{0}, []int{8 - rows%8})
			}
			want := padded.Matmul(stock).Slice(Slice(0, rows), Slice())
			checkBF16Matmul(t, got, want)
		}
	})
}

func TestMatmulBF16WeightEdges(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !SupportsBF16WeightMatmul() {
			t.Skip("requires a qualified BF16 projection")
		}
		// All partial sums are exactly representable even if stock MLX picks a
		// different reduction on smaller matrices. Both operands are strided.
		for _, shape := range [][3]int{{1, 1, 1}, {5, 13, 8}, {5, 13, 19}, {11, 35, 13}, {11, 35, 41}, {9, 16, 17}, {16, 32, 64}} {
			rows, width, columns := shape[0], shape[1], shape[2]
			matrix := func(n, stride int, dtype DType) *Array {
				values := make([]float32, n*stride)
				for i := range values {
					values[i] = float32(i%23-11) / 8
				}
				return FromValues(values, n, stride).AsType(dtype).Slice(Slice(), Slice(1, width+1))
			}
			w := matrix(columns, width+3, DTypeBFloat16)
			stock := w.AsType(DTypeFloat32).Transpose(1, 0)
			for _, dtype := range []DType{DTypeFloat32, DTypeBFloat16} {
				x := matrix(rows, width+5, dtype)
				got := MatmulBF16Weight(x, w)
				if usesBF16TensorMatmul() && columns < width {
					if got != nil {
						t.Fatalf("tensor projection accepted contracting shape %v", shape)
					}
					continue
				}
				if got == nil {
					t.Fatalf("declined %v, dtype %v", shape, dtype)
				}
				checkBF16Matmul(t, got, x.AsType(DTypeFloat32).Matmul(stock))
			}
		}
		w := Zeros(DTypeBFloat16, 11, 9)
		x := Zeros(DTypeFloat32, 2, 9)
		for _, inputs := range [][2]*Array{
			{nil, w},
			{x, nil},
			{Zeros(DTypeFloat32, 17, 9), w},
			{Zeros(DTypeFloat32, 1, 1, 9), w},
			{x, w.Reshape(1, 11, 9)},
			{x, Zeros(DTypeBFloat16, 11, 8)},
			{x, w.AsType(DTypeFloat32)},
			{x.AsType(DTypeFloat16), w},
			{x, Zeros(DTypeBFloat16, 0, 9)},
			{Zeros(DTypeFloat32, 0, 9), w},
			{Zeros(DTypeFloat32, 1, 0), Zeros(DTypeBFloat16, 2, 0)},
		} {
			if MatmulBF16Weight(inputs[0], inputs[1]) != nil {
				t.Fatal("unexpected projection for an unsupported input")
			}
		}
	})
}

func TestMatmulBF16WeightSplitK(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !SupportsBF16WeightMatmul() {
			t.Skip("requires a qualified BF16 projection")
		}
		const rows, width = 8, 128
		values := make([]float32, (width+1)*width)
		for i := range values {
			values[i] = float32(math.Sin(float64(i) * 0.15891))
		}
		x := FromValues(values[:rows*width], rows, width)
		for _, columns := range []int{127, 128, 129} {
			w := FromValues(values[:columns*width], columns, width).AsType(DTypeBFloat16)
			got := MatmulBF16Weight(x, w)
			if columns <= width {
				if got != nil {
					t.Fatalf("accepted possible split-K projection [%d,%d] x [%d,%d]", rows, width, columns, width)
				}
				continue
			}
			if got == nil {
				t.Fatal("declined expanding projection")
			}
			checkBF16Matmul(t, got, x.Matmul(w.AsType(DTypeFloat32).Transpose(1, 0)))
		}
	})
}

func checkBF16Matmul(t *mlxthreadtest.T, got, want *Array) {
	if got.NumDims() != 2 || got.Dim(0) != want.Dim(0) || got.Dim(1) != want.Dim(1) || got.DType() != DTypeFloat32 {
		t.Fatalf("output %v %v, want %v FP32", got.Dims(), got.DType(), want.Dims())
	}
	matches := got.View(DTypeUint32).Equal(want.View(DTypeUint32)).AsType(DTypeInt32).Reshape(-1).SumAxis(0, false).Int()
	if int(matches) != want.Size() {
		t.Errorf("%v differs at %d FP32 elements", got.Dims(), want.Size()-int(matches))
	}
}
