package convert

import "testing"

// ollama/ollama#17177: a crafted num_hidden_layers made HiddenLayers*3 wrap
// around under uint32 arithmetic, so `merges` was allocated far smaller than
// the loop below it assumed, causing an out-of-bounds slice write panic
// ("index out of range [2] with length 2") on the very first iteration.
// validate() (wired in via parseMore) now rejects such values up front with
// a clean error instead of letting conversion crash the server process.
func TestDeepSeek2ValidateRejectsOverflowInducingHiddenLayers(t *testing.T) {
	p := &deepseek2Model{HiddenLayers: 1431655766} // 3*1431655766 wraps to 2 (uint32)
	if err := p.validate(); err == nil {
		t.Fatal("validate() = nil, want an error for an overflow-inducing num_hidden_layers")
	}
	// NOTE: intentionally not also calling p.Tensors() here with this value.
	// validate() is what must stop it; actually running Tensors() on an
	// unvalidated ~1.4e9 layer count would (correctly) no longer panic after
	// the append-based rewrite below, but it would still iterate ~1.4e9
	// times, so it belongs behind validate(), not inside a fast unit test.
}

func TestDeepSeek2ValidateRejectsZeroHiddenLayers(t *testing.T) {
	p := &deepseek2Model{HiddenLayers: 0}
	if err := p.validate(); err == nil {
		t.Fatal("validate() = nil, want an error for num_hidden_layers=0")
	}
}

func TestDeepSeek2ValidateRejectsExcessiveHiddenLayers(t *testing.T) {
	// A moderately large value that does NOT overflow uint32*3, to isolate
	// the sanity-bound check from the overflow case above.
	p := &deepseek2Model{HiddenLayers: maxDeepSeek2HiddenLayers + 1}
	if err := p.validate(); err == nil {
		t.Fatalf("validate() = nil, want an error for num_hidden_layers=%d (max is %d)", p.HiddenLayers, maxDeepSeek2HiddenLayers)
	}
}

func TestDeepSeek2ValidateAcceptsRealisticHiddenLayers(t *testing.T) {
	// DeepSeek-V3/V3.1 style configs use ~61 layers; well within bounds.
	p := &deepseek2Model{HiddenLayers: 61}
	if err := p.validate(); err != nil {
		t.Fatalf("validate() = %v, want nil for a realistic num_hidden_layers", err)
	}

	out := p.Tensors(nil)
	if len(out) != 0 {
		t.Fatalf("Tensors(nil) = %d tensors, want 0 (no input tensors supplied)", len(out))
	}
}

// TestDeepSeek2TensorsDoesNotPanicOnRejectedButBoundedValue is a fast defense-
// in-depth check: even for a HiddenLayers value validate() would reject
// (just above the max), Tensors() itself must not panic if some future call
// path skips validation -- the append-based construction must stay safe on
// its own, independent of the bound check. Kept small so the loop is cheap.
func TestDeepSeek2TensorsDoesNotPanicOnRejectedButBoundedValue(t *testing.T) {
	p := &deepseek2Model{HiddenLayers: maxDeepSeek2HiddenLayers + 1}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Tensors() panicked despite the append-based rewrite: %v", r)
		}
	}()
	_ = p.Tensors(nil)
}
