package mlx

import (
	"testing"
)

func TestCompileFusion(t *testing.T) {
	skipIfNoMLX(t)

	// Compile fuses the ops inside a function body into a single kernel,
	// eliminating intermediate buffers. Use a diamond-shaped graph where
	// two branches must be materialized simultaneously without fusion,
	// then compare peak memory against the compiled version which fuses
	// everything into one kernel with no intermediates.
	const n = 1024 * 1024 // 4MB per float32 array
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i + 1)
	}

	// Diamond: both a*b and a+b must be live for the final multiply.
	// Without fusion: peak includes both intermediates (~8MB extra).
	// With fusion: single kernel, no intermediates.
	body := func(a, b *Array) *Array {
		return a.Multiply(b).Multiply(a.Add(b))
	}

	a := FromValues(data, n)
	b := FromValues(data, n)
	Pin(a, b)
	defer Unpin(a, b)

	// Compiled: ops fused into a single kernel.
	EnableCompile()
	fn := Compile2("diamond", body, Shapeless())
	warm := fn(a, b)
	Eval(warm)
	Sweep()
	ClearCache()
	ResetPeakMemory()
	y := fn(a, b)
	Eval(y)
	compiledPeak := PeakMemory()
	Sweep()

	// Uncompiled: ops evaluated individually, intermediates materialized.
	ClearCache()
	ResetPeakMemory()
	z := body(a, b)
	Eval(z)
	uncompiledPeak := PeakMemory()
	Sweep()

	if compiledPeak == 0 && uncompiledPeak == 0 {
		t.Skip("peak memory tracking not available")
	}

	t.Logf("peak memory: compiled=%d uncompiled=%d", compiledPeak, uncompiledPeak)

	if compiledPeak >= uncompiledPeak {
		t.Fatalf("compilation did not reduce peak memory: compiled=%d uncompiled=%d", compiledPeak, uncompiledPeak)
	}
}

func TestCompileNested(t *testing.T) {
	skipIfNoMLX(t)

	// A compiled function that calls another compiled function should
	// produce correct results. The inner function inlines via isTracing()
	// during the outer's trace.
	inner := Compile1("silu", func(a *Array) *Array {
		return a.Multiply(a.Sigmoid())
	}, Shapeless())

	outer := Compile2("swiglu", func(gate, up *Array) *Array {
		return inner(gate).Multiply(up)
	}, Shapeless())

	gate := FromValues([]float32{0, 1, 2}, 3)
	up := FromValues([]float32{1, 1, 1}, 3)
	Pin(gate, up)
	defer Unpin(gate, up)

	y := outer(gate, up)
	Eval(y)

	// silu(x) = x * sigmoid(x); for x=0 → 0, x=1 → ~0.7311, x=2 → ~1.7616
	got := y.Floats()
	want := []float32{0, 0.7310586, 1.7615942}
	for i, v := range got {
		if v-want[i] > 1e-4 || want[i]-v > 1e-4 {
			t.Fatalf("got[%d]=%v want %v", i, v, want[i])
		}
	}
}

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
	}()
	boom(x)
}

func TestCompileNoTrackingGrowth(t *testing.T) {
	skipIfNoMLX(t)

	// Repeated invocations of a compiled kernel should not grow the
	// tracked-arrays list — the callback's traceScratch collects
	// intermediates during tracing and frees them when the callback returns.
	fn := Compile2("mul_add", func(a, b *Array) *Array {
		return a.Multiply(b).Add(b)
	})

	a := FromValues([]float32{1, 2}, 2)
	b := FromValues([]float32{3, 4}, 2)
	Pin(a, b)
	defer Unpin(a, b)

	Sweep()
	before := len(arrays)

	for range 100 {
		_ = fn(a, b)
		Sweep()
	}

	after := len(arrays)
	if after > before+2 {
		t.Fatalf("tracked arrays grew from %d to %d across 100 calls (includes initial trace)", before, after)
	}
}
