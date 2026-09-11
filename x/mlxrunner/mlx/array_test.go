package mlx

import (
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthreadtest"
)

func TestFromValue(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		for got, want := range map[*Array]DType{
			FromValue(true):              DTypeBool,
			FromValue(false):             DTypeBool,
			FromValue(int(7)):            DTypeInt32,
			FromValue(float32(3.14)):     DTypeFloat32,
			FromValue(float64(2.71)):     DTypeFloat64,
			FromValue(complex64(1 + 2i)): DTypeComplex64,
		} {
			if got.DType() != want {
				t.Errorf("%s: want %v, got %v", want, want, got)
			}
		}
	})
}

func TestFromValues(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		for got, want := range map[*Array]DType{
			FromValues([]bool{true, false, true}, 3):           DTypeBool,
			FromValues([]uint8{1, 2, 3}, 3):                    DTypeUint8,
			FromValues([]uint16{1, 2, 3}, 3):                   DTypeUint16,
			FromValues([]uint32{1, 2, 3}, 3):                   DTypeUint32,
			FromValues([]uint64{1, 2, 3}, 3):                   DTypeUint64,
			FromValues([]int8{-1, -2, -3}, 3):                  DTypeInt8,
			FromValues([]int16{-1, -2, -3}, 3):                 DTypeInt16,
			FromValues([]int32{-1, -2, -3}, 3):                 DTypeInt32,
			FromValues([]int64{-1, -2, -3}, 3):                 DTypeInt64,
			FromValues([]float32{3.14, 2.71, 1.61}, 3):         DTypeFloat32,
			FromValues([]float64{3.14, 2.71, 1.61}, 3):         DTypeFloat64,
			FromValues([]complex64{1 + 2i, 3 + 4i, 5 + 6i}, 3): DTypeComplex64,
		} {
			if got.DType() != want {
				t.Errorf("%s: want %v, got %v", want, want, got)
			}
		}
	})
}

func TestComparisonOpsAndBernoulli(t *testing.T) {
	var tests []struct {
		name string
		got  []int32
		want []int32
	}
	withMLXThread(t, func(*mlxthreadtest.T) {
		a := FromValues([]float32{1, 2, 3}, 3)
		b := FromValues([]float32{1, 1, 4}, 3)
		eq := a.Equal(b).AsType(DTypeInt32)
		gt := a.Greater(b).AsType(DTypeInt32)
		le := a.LessEqual(b).AsType(DTypeInt32)
		bern := Bernoulli(FromValues([]float32{1, 0}, 2)).AsType(DTypeInt32)
		Eval(eq, gt, le, bern)

		tests = []struct {
			name string
			got  []int32
			want []int32
		}{
			{name: "equal", got: eq.Ints(), want: []int32{1, 0, 0}},
			{name: "greater", got: gt.Ints(), want: []int32{0, 1, 0}},
			{name: "lessEqual", got: le.Ints(), want: []int32{1, 0, 1}},
			{name: "bernoulli", got: bern.Ints(), want: []int32{1, 0}},
		}
	})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.got) != len(tt.want) {
				t.Fatalf("got %v, want %v", tt.got, tt.want)
			}
			for i := range tt.want {
				if tt.got[i] != tt.want[i] {
					t.Fatalf("got %v, want %v", tt.got, tt.want)
				}
			}
		})
	}
}

// An empty array has no buffer, so its data pointer is null without an error.
func TestEmptyArrayData(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if got := Zeros(DTypeFloat32, 0).Floats(); len(got) != 0 {
			t.Fatalf("Floats() = %v, want empty", got)
		}
		if got := Zeros(DTypeInt32, 0).Ints(); len(got) != 0 {
			t.Fatalf("Ints() = %v, want empty", got)
		}
	})
}
