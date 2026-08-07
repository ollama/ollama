package mlx

import "testing"

func TestReLUSquared(t *testing.T) {
	withMLXThread(t, func() {
		x := FromValues([]float32{-2, -0, 0.5, 2}, 4)
		Pin(x)
		defer Unpin(x)

		y := ReLUSquared(x)
		Eval(y)

		got := y.Floats()
		want := []float32{0, 0, 0.25, 4}
		for i, v := range got {
			if v != want[i] {
				t.Fatalf("got[%d]=%v want %v", i, v, want[i])
			}
		}
	})
}
