package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// makeKV creates a [1, 1, seqLen, 1] float32 array filled with val.
func makeKV(t *testing.T, seqLen int, val float32) *mlx.Array {
	t.Helper()
	data := make([]float32, seqLen)
	for i := range data {
		data[i] = val
	}
	return mlx.FromValues(data, 1, 1, seqLen, 1)
}

// readValues evaluates an array and returns its float32 values.
func readValues(t *testing.T, a *mlx.Array) []float32 {
	t.Helper()
	mlx.Eval(a)
	return a.Floats()
}

func TestKVCacheBasic(t *testing.T) {
	skipIfNoMLX(t)

	c := NewKVCache()

	// Insert 3 tokens
	k, v := c.Update(makeKV(t, 3, 1.0), makeKV(t, 3, 10.0))
	got := readValues(t, k)
	if len(got) != 3 {
		t.Fatalf("expected 3 values, got %d", len(got))
	}
	for i, v := range got {
		if v != 1.0 {
			t.Errorf("k[%d] = %f, want 1.0", i, v)
		}
	}
	gotV := readValues(t, v)
	for i, v := range gotV {
		if v != 10.0 {
			t.Errorf("v[%d] = %f, want 10.0", i, v)
		}
	}

	if c.Offset() != 3 {
		t.Errorf("offset = %d, want 3", c.Offset())
	}

	// Trim 1 token
	trimmed := c.Trim(1)
	if trimmed != 1 {
		t.Errorf("trimmed = %d, want 1", trimmed)
	}
	if c.Offset() != 2 {
		t.Errorf("offset after trim = %d, want 2", c.Offset())
	}
}

func TestRotatingKVCacheBasicFill(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRotatingKVCache(4)

	// Prefill 3 tokens via concat path (seqLen > 1)
	k, _ := c.Update(makeKV(t, 3, 1.0), makeKV(t, 3, 10.0))
	got := readValues(t, k)
	if len(got) != 3 {
		t.Fatalf("expected 3 values after prefill, got %d", len(got))
	}
	if c.Offset() != 3 {
		t.Errorf("offset = %d, want 3", c.Offset())
	}

	// Add 1 more token via update path (seqLen == 1)
	k, _ = c.Update(makeKV(t, 1, 2.0), makeKV(t, 1, 20.0))
	got = readValues(t, k)
	if len(got) != 4 {
		t.Fatalf("expected 4 values, got %d", len(got))
	}
	if c.Offset() != 4 {
		t.Errorf("offset = %d, want 4", c.Offset())
	}
}

// TestRotatingKVCacheWrapAndConcat tests that after the sliding window wraps
// (offset > maxSize), a subsequent concat (multi-token update) correctly
// linearizes the circular buffer and produces a valid result. Before the fix,
// concat didn't handle the wrapped case (offset > maxSize) and would produce
// garbled sequences or panic.
func TestRotatingKVCacheWrapAndConcat(t *testing.T) {
	skipIfNoMLX(t)

	maxSize := 4
	c := NewRotatingKVCache(maxSize)

	// Prefill to fill the window: 4 tokens
	c.Update(makeKV(t, 4, 1.0), makeKV(t, 4, 1.0))

	// Generate 3 more single tokens to wrap the buffer.
	// After these, offset=7 > maxSize=4, buffer is circular.
	c.Update(makeKV(t, 1, 5.0), makeKV(t, 1, 5.0))
	c.Update(makeKV(t, 1, 6.0), makeKV(t, 1, 6.0))
	k, _ := c.Update(makeKV(t, 1, 7.0), makeKV(t, 1, 7.0))

	if c.Offset() != 7 {
		t.Fatalf("offset = %d, want 7", c.Offset())
	}

	// After wrapping, the returned view should have maxSize elements.
	got := readValues(t, k)
	if len(got) != maxSize {
		t.Fatalf("expected %d values after wrap, got %d", maxSize, len(got))
	}

	// Now do a concat (multi-token update) on the wrapped buffer.
	// This exercises the linearization path (offset > maxSize in concat).
	// Before the fix, this would either panic or produce wrong results.
	k, _ = c.Update(makeKV(t, 2, 8.0), makeKV(t, 2, 8.0))
	got = readValues(t, k)

	// After concat with linearization: should have maxSize elements, and
	// the last 2 should be the newly inserted values (8.0).
	if len(got) != maxSize {
		t.Fatalf("expected %d values after concat on wrapped cache, got %d", maxSize, len(got))
	}
	// The last two entries must be the newly concatenated values.
	if got[maxSize-2] != 8.0 || got[maxSize-1] != 8.0 {
		t.Errorf("last two values after concat = [%f, %f], want [8.0, 8.0]", got[maxSize-2], got[maxSize-1])
	}

	// Verify offset advanced correctly
	if c.Offset() != 9 {
		t.Errorf("offset after concat = %d, want 9", c.Offset())
	}
}

// TestRotatingKVCacheTrimWrapped tests that Trim returns 0 when the cache
// has wrapped past maxSize. Before the fix, Trim would blindly subtract from
// offset and idx, leaving the cache in a corrupt state where update() would
// compute negative growth sizes.
func TestRotatingKVCacheTrimWrapped(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRotatingKVCache(4)

	// Fill and wrap: 4 prefill + 2 generate = offset 6 > maxSize 4
	c.Update(makeKV(t, 4, 1.0), makeKV(t, 4, 1.0))
	c.Update(makeKV(t, 1, 2.0), makeKV(t, 1, 2.0))
	c.Update(makeKV(t, 1, 3.0), makeKV(t, 1, 3.0))

	if c.Offset() != 6 {
		t.Fatalf("offset = %d, want 6", c.Offset())
	}

	// Trim should return 0 — cannot safely trim a wrapped circular buffer.
	trimmed := c.Trim(2)
	if trimmed != 0 {
		t.Errorf("Trim(2) on wrapped cache returned %d, want 0", trimmed)
	}

	// Offset should be unchanged since trim was refused.
	if c.Offset() != 6 {
		t.Errorf("offset after failed trim = %d, want 6", c.Offset())
	}
}

// TestRotatingKVCacheStateAfterWrap tests that State() returns valid arrays
// when offset > physical array dimension. Before the fix, State() used offset
// directly as a slice bound, which could exceed the array size.
func TestRotatingKVCacheStateAfterWrap(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRotatingKVCache(4)

	// Fill and wrap
	c.Update(makeKV(t, 4, 1.0), makeKV(t, 4, 1.0))
	c.Update(makeKV(t, 1, 2.0), makeKV(t, 1, 2.0))
	c.Update(makeKV(t, 1, 3.0), makeKV(t, 1, 3.0))

	// offset=6, but physical array is maxSize=4
	k, v := c.State()
	if k == nil || v == nil {
		t.Fatal("State() returned nil after wrap")
	}

	// Should not panic and should return maxSize elements
	kVals := readValues(t, k)
	if len(kVals) != 4 {
		t.Errorf("State() returned %d values, want 4", len(kVals))
	}
}

// TestRotatingKVCacheTrimUnwrapped tests that Trim works normally when the
// cache hasn't wrapped (offset <= maxSize).
func TestRotatingKVCacheTrimUnwrapped(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRotatingKVCache(8)

	// Prefill 3 tokens, generate 2 more = offset 5, maxSize 8, no wrap
	c.Update(makeKV(t, 3, 1.0), makeKV(t, 3, 1.0))
	c.Update(makeKV(t, 1, 2.0), makeKV(t, 1, 2.0))
	c.Update(makeKV(t, 1, 3.0), makeKV(t, 1, 3.0))

	if c.Offset() != 5 {
		t.Fatalf("offset = %d, want 5", c.Offset())
	}

	// Trim 2 tokens — should succeed since we haven't wrapped.
	trimmed := c.Trim(2)
	if trimmed != 2 {
		t.Errorf("Trim(2) returned %d, want 2", trimmed)
	}
	if c.Offset() != 3 {
		t.Errorf("offset after trim = %d, want 3", c.Offset())
	}

	// Can still generate after trimming
	k, _ := c.Update(makeKV(t, 1, 4.0), makeKV(t, 1, 4.0))
	got := readValues(t, k)
	if len(got) != 4 {
		t.Errorf("expected 4 values after trim+generate, got %d", len(got))
	}
}

// TestRotatingKVCacheLenAfterWrap tests that Len() returns the capped value
// (maxSize) rather than the raw offset when the cache has wrapped.
func TestRotatingKVCacheLenAfterWrap(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRotatingKVCache(4)

	c.Update(makeKV(t, 4, 1.0), makeKV(t, 4, 1.0))
	c.Update(makeKV(t, 1, 2.0), makeKV(t, 1, 2.0))

	// offset=5, maxSize=4
	if c.Len() != 4 {
		t.Errorf("Len() = %d, want 4 (capped to maxSize)", c.Len())
	}
	if c.Offset() != 5 {
		t.Errorf("Offset() = %d, want 5 (raw offset)", c.Offset())
	}
}
