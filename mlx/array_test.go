package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestContiguousDetachesLargeView(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		ClearCache()
		baseline := ActiveMemory()
		holder := NewScope()
		defer holder.Close()
		var compact *Array
		const stateElements = 16 * 128 * 128
		const stateBytes = stateElements * 4

		Scoped(func() {
			values := make([]float32, 3*stateElements)
			values[stateElements] = math.Float32frombits(0x80000000)
			values[stateElements+1] = math.Float32frombits(0x7fc01234)
			values[stateElements+2] = 1
			source := FromValues(values, 3, 1, 16, 128, 128)
			Eval(source)
			view := SliceStartStop(source,
				[]int32{1, 0, 0, 0, 0}, []int32{2, 1, 16, 128, 128}).Reshape(1, 16, 128, 128)
			compact = Contiguous(view.Clone(), false)
			Eval(compact)
			holder.Attach(compact)
		})
		copied := compact.Floats()
		for i, want := range []uint32{0x80000000, 0x7fc01234, 0x3f800000} {
			if got := math.Float32bits(copied[i]); got != want {
				t.Fatalf("copied state bits[%d] = %08x, want %08x", i, got, want)
			}
		}

		// A dependent GPU operation drains any in-flight use of the source.
		// A shallow clone still keeps the full 3 MiB after this point.
		Scoped(func() {
			Eval(Add(compact, FromValues([]float32{2}, 1)))
		})
		ClearCache()
		// The compact state and dependent output can each occupy one state.
		// A shared view would also keep the three-state source buffer.
		if retained := ActiveMemory() - baseline; retained > 2*stateBytes+stateBytes/2 {
			t.Fatalf("one-state copy retained %d bytes of its 3 MiB source", retained)
		}
	})
}

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
