package mlx

import "testing"

func TestFromValue(t *testing.T) {
	withMLXThread(t, func() {
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
	withMLXThread(t, func() {
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
	skipIfNoMLX(t)

	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{1, 1, 4}, 3)
	eq := a.Equal(b).AsType(DTypeInt32)
	gt := a.Greater(b).AsType(DTypeInt32)
	le := a.LessEqual(b).AsType(DTypeInt32)
	bern := Bernoulli(FromValues([]float32{1, 0}, 2)).AsType(DTypeInt32)
	Eval(eq, gt, le, bern)

	for name, tc := range map[string]struct {
		got  []int
		want []int
	}{
		"equal":     {eq.Ints(), []int{1, 0, 0}},
		"greater":   {gt.Ints(), []int{0, 1, 0}},
		"lessEqual": {le.Ints(), []int{1, 0, 1}},
		"bernoulli": {bern.Ints(), []int{1, 0}},
	} {
		t.Run(name, func(t *testing.T) {
			if len(tc.got) != len(tc.want) {
				t.Fatalf("got %v, want %v", tc.got, tc.want)
			}
			for i := range tc.want {
				if tc.got[i] != tc.want[i] {
					t.Fatalf("got %v, want %v", tc.got, tc.want)
				}
			}
		})
	}
}
