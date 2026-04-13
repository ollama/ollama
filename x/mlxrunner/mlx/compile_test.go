package mlx

import (
	"math"
	"testing"
)

func TestCompileCallbackPanicRecovers(t *testing.T) {
	skipIfNoMLX(t)

	boom := Compile1("boom", func(a *Array) *Array {
		panic("intentional test panic")
	})

	x := FromValues([]float32{1}, 1)
	Pin(x)
	defer Unpin(x)

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic from Call, got none")
		}
		if _, ok := r.(string); !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		// MLX overwrites the message; what matters is that the Go
		// panic was caught before unwinding into C (which would be UB)
		// and that Call surfaced a failure instead.
	}()
	boom(x)
}

func TestCompileUnary(t *testing.T) {
	skipIfNoMLX(t)

	square := Compile1("square", func(a *Array) *Array {
		return a.Multiply(a)
	})

	x := FromValues([]float32{1, 2, 3, 4}, 4)
	Pin(x)
	defer Unpin(x)

	y := square(x)
	Eval(y)

	got := y.Floats()
	want := []float32{1, 4, 9, 16}
	for i, v := range got {
		if v != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, v, want[i])
		}
	}
}

func TestCompileBinary(t *testing.T) {
	skipIfNoMLX(t)

	add := Compile2("add", func(a, b *Array) *Array {
		return a.Add(b)
	})

	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{10, 20, 30}, 3)
	Pin(a, b)
	defer Unpin(a, b)

	c := add(a, b)
	Eval(c)

	got := c.Floats()
	want := []float32{11, 22, 33}
	for i, v := range got {
		if v != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, v, want[i])
		}
	}
}

func TestCompileShapelessReshape(t *testing.T) {
	skipIfNoMLX(t)

	// A shapeless compiled kernel should accept inputs of different shapes
	// on subsequent calls without recompiling or erroring.
	fn := Compile1("square", func(a *Array) *Array {
		return a.Multiply(a)
	}, Shapeless())

	for _, n := range []int{2, 4, 8} {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i + 1)
		}
		x := FromValues(data, n)
		Pin(x)
		y := fn(x)
		Eval(y)
		got := y.Floats()
		for i, v := range got {
			want := float32((i + 1) * (i + 1))
			if v != want {
				t.Fatalf("n=%d got[%d]=%v want %v", n, i, v, want)
			}
		}
		Unpin(x)
	}
}

func TestSwiGLU(t *testing.T) {
	skipIfNoMLX(t)

	gate := FromValues([]float32{-1, 0, 1, 2}, 4)
	up := FromValues([]float32{1, 2, 3, 4}, 4)
	Pin(gate, up)
	defer Unpin(gate, up)

	y := SwiGLU(gate, up)
	Eval(y)

	got := y.Floats()
	// Reference: silu(g) * u = g * sigmoid(g) * u
	wantVals := []float32{-1, 0, 1, 2}
	upVals := []float32{1, 2, 3, 4}
	for i, g := range wantVals {
		silu := g / float32(1+math.Exp(float64(-g)))
		want := silu * upVals[i]
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Fatalf("i=%d got=%v want=%v", i, got[i], want)
		}
	}
}

func TestCompileNoTrackingGrowth(t *testing.T) {
	skipIfNoMLX(t)

	// Repeated invocations of a compiled kernel should not grow the
	// tracked-arrays list by the number of internal ops each call — the
	// callback's scratch list frees them before return.
	fn := Compile2("mul_add", func(a, b *Array) *Array {
		// Two ops per call; if we leaked, we'd see growth proportional to
		// iterations * 2 in the tracked list.
		return a.Multiply(b).Add(b)
	})

	a := FromValues([]float32{1, 2}, 2)
	b := FromValues([]float32{3, 4}, 2)
	Pin(a, b)
	defer Unpin(a, b)

	// Prime so the initial trace's allocations are already accounted for.
	_ = fn(a, b)
	Eval()
	Sweep()

	before := len(arrays)
	for range 100 {
		_ = fn(a, b)
		Eval()
		Sweep()
	}
	after := len(arrays)
	if after > before+2 {
		t.Fatalf("tracked arrays grew from %d to %d across 100 calls", before, after)
	}
}
