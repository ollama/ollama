package mlx

import (
	"math"
	"testing"
)

func TestFastMoEWeightedSum(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		expert := FromValues([]float32{
			1, 2, 3, 4,
			10, 20, 30, 40,
			-1, -2, -3, -4,
			2, 0, -2, 4,
			1, 1, 1, 1,
			3, 3, 3, 3,
		}, 1, 2, 3, 4)
		scores := FromValues([]float32{
			0.5, 0.25, 0.25,
			1, -1, 0.5,
		}, 1, 2, 3)
		Pin(expert, scores)
		defer Unpin(expert, scores)

		got, ok := FastMoEWeightedSum(expert, scores, nil, nil, DTypeFloat32, 1)
		if !ok {
			t.Fatal("FastMoEWeightedSum returned ok=false")
		}
		Eval(got)

		if got.DType() != DTypeFloat32 {
			t.Fatalf("dtype = %v, want %v", got.DType(), DTypeFloat32)
		}
		if dims := got.Dims(); len(dims) != 3 || dims[0] != 1 || dims[1] != 2 || dims[2] != 4 {
			t.Fatalf("dims = %v, want [1 2 4]", dims)
		}

		want := []float32{2.75, 5.5, 8.25, 11, 2.5, 0.5, -1.5, 4.5}
		for i, v := range got.Floats() {
			if diff := math.Abs(float64(v - want[i])); diff > 1e-6 {
				t.Fatalf("got[%d] = %v, want %v", i, v, want[i])
			}
		}
	})
}

func TestFastMoEWeightedSumWithScale(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		expert := FromValues([]float32{
			1, 2, 3, 4,
			10, 20, 30, 40,
			-1, -2, -3, -4,
			2, 0, -2, 4,
			1, 1, 1, 1,
			3, 3, 3, 3,
		}, 1, 2, 3, 4)
		scores := FromValues([]float32{
			0.5, 0.25, 0.25,
			1, -1, 0.5,
		}, 1, 2, 3)
		Pin(expert, scores)
		defer Unpin(expert, scores)

		got, ok := FastMoEWeightedSum(expert, scores, nil, nil, DTypeFloat32, 2.5)
		if !ok {
			t.Fatal("FastMoEWeightedSum returned ok=false")
		}
		Eval(got)

		want := []float32{6.875, 13.75, 20.625, 27.5, 6.25, 1.25, -3.75, 11.25}
		for i, v := range got.Floats() {
			if diff := math.Abs(float64(v - want[i])); diff > 1e-6 {
				t.Fatalf("got[%d] = %v, want %v", i, v, want[i])
			}
		}
	})
}

func TestFastMoEWeightedSumWithAdd(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		expert := FromValues([]float32{
			1, 2, 3, 4,
			10, 20, 30, 40,
			-1, -2, -3, -4,
			2, 0, -2, 4,
			1, 1, 1, 1,
			3, 3, 3, 3,
		}, 1, 2, 3, 4)
		scores := FromValues([]float32{
			0.5, 0.25, 0.25,
			1, -1, 0.5,
		}, 1, 2, 3)
		shared := FromValues([]float32{
			100, 200, 300, 400,
			-10, -20, -30, -40,
		}, 1, 2, 4)
		Pin(expert, scores, shared)
		defer Unpin(expert, scores, shared)

		got, ok := FastMoEWeightedSum(expert, scores, shared, nil, DTypeFloat32, 2.5)
		if !ok {
			t.Fatal("FastMoEWeightedSum returned ok=false")
		}
		Eval(got)

		want := []float32{106.875, 213.75, 320.625, 427.5, -3.75, -18.75, -33.75, -28.75}
		for i, v := range got.Floats() {
			if diff := math.Abs(float64(v - want[i])); diff > 1e-6 {
				t.Fatalf("got[%d] = %v, want %v", i, v, want[i])
			}
		}
	})
}

func TestFastMoEWeightedSumWithTwoAddends(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		expert := FromValues([]float32{
			1, 2, 3, 4,
			10, 20, 30, 40,
			-1, -2, -3, -4,
			2, 0, -2, 4,
			1, 1, 1, 1,
			3, 3, 3, 3,
		}, 1, 2, 3, 4)
		scores := FromValues([]float32{
			0.5, 0.25, 0.25,
			1, -1, 0.5,
		}, 1, 2, 3)
		addA := FromValues([]float32{
			100, 200, 300, 400,
			-10, -20, -30, -40,
		}, 1, 2, 4)
		addB := FromValues([]float32{
			1, 2, 3, 4,
			10, 20, 30, 40,
		}, 1, 2, 4)
		Pin(expert, scores, addA, addB)
		defer Unpin(expert, scores, addA, addB)

		got, ok := FastMoEWeightedSum(expert, scores, addA, addB, DTypeFloat32, 2.5)
		if !ok {
			t.Fatal("FastMoEWeightedSum returned ok=false")
		}
		Eval(got)

		want := []float32{107.875, 215.75, 323.625, 431.5, 6.25, 1.25, -3.75, 11.25}
		for i, v := range got.Floats() {
			if diff := math.Abs(float64(v - want[i])); diff > 1e-6 {
				t.Fatalf("got[%d] = %v, want %v", i, v, want[i])
			}
		}
	})
}
