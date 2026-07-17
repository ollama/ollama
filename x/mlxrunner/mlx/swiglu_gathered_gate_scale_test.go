package mlx

import "testing"

func TestFastSwiGLUGatheredGateScale(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		gate := FromValues([]float32{
			-2, -1, 0,
			1, 2, 3,
			-3, 0.5, 4,
			2, -0.5, 1.5,
		}, 2, 2, 1, 3)
		up := FromValues([]float32{
			1, 2, 3,
			4, 5, 6,
			-1, -2, -3,
			0.5, 1.5, 2.5,
		}, 2, 2, 1, 3)
		scales := FromValues([]float32{0.5, 2, 3}, 3)
		indices := FromValues([]int32{0, 2, 1, 0}, 2, 2)
		Pin(gate, up, scales, indices)
		defer Unpin(gate, up, scales, indices)

		got, ok := FastSwiGLUGatheredGateScale(gate, up, scales, indices)
		if !ok {
			t.Fatal("FastSwiGLUGatheredGateScale returned ok=false")
		}
		want := SwiGLU(Mul(gate, ExpandDims(ExpandDims(Take(scales, indices, 0), -1), -1)), up)
		Eval(got, want)

		assertFloat32Close(t, got.Floats(), want.Floats(), 1e-5)
	})
}

func TestFastSwiGLUGatheredGateScaleScalarScale(t *testing.T) {
	withMLXThread(t, func() {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}

		gate := FromValues([]float32{
			-2, -1, 0,
			1, 2, 3,
			-3, 0.5, 4,
			2, -0.5, 1.5,
		}, 2, 2, 1, 3)
		up := FromValues([]float32{
			1, 2, 3,
			4, 5, 6,
			-1, -2, -3,
			0.5, 1.5, 2.5,
		}, 2, 2, 1, 3)
		scales := FromValues([]float32{0.5}, 1)
		indices := FromValues([]int32{0, 2, 1, 0}, 2, 2)
		Pin(gate, up, scales, indices)
		defer Unpin(gate, up, scales, indices)

		got, ok := FastSwiGLUGatheredGateScale(gate, up, scales, indices)
		if !ok {
			t.Fatal("FastSwiGLUGatheredGateScale returned ok=false")
		}
		want := SwiGLU(Mul(gate, Reshape(scales, 1, 1, 1, 1)), up)
		Eval(got, want)

		assertFloat32Close(t, got.Floats(), want.Floats(), 1e-5)
	})
}
