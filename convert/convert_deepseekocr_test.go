package convert

import "testing"

// Same overflow-panic class as ollama/ollama#17177 (convert_deepseek2.go),
// found by auditing sibling converters that share the same merges := make +
// direct-index pattern. See convert_deepseek2_test.go for the full writeup.
func TestDeepSeekOCRValidateRejectsOverflowInducingHiddenLayers(t *testing.T) {
	m := &deepseekocr{}
	m.LanguageConfig.HiddenLayers = 1431655766 // 3*1431655766 wraps to 2 (uint32)
	if err := m.validate(); err == nil {
		t.Fatal("validate() = nil, want an error for an overflow-inducing num_hidden_layers")
	}
}

func TestDeepSeekOCRValidateRejectsZeroHiddenLayers(t *testing.T) {
	m := &deepseekocr{}
	if err := m.validate(); err == nil {
		t.Fatal("validate() = nil, want an error for num_hidden_layers=0")
	}
}

func TestDeepSeekOCRValidateAcceptsRealisticHiddenLayers(t *testing.T) {
	m := &deepseekocr{}
	m.LanguageConfig.HiddenLayers = 27
	if err := m.validate(); err != nil {
		t.Fatalf("validate() = %v, want nil for a realistic num_hidden_layers", err)
	}
	out := m.Tensors(nil)
	if len(out) != 0 {
		t.Fatalf("Tensors(nil) = %d tensors, want 0 (no input tensors supplied)", len(out))
	}
}

func TestDeepSeekOCRTensorsDoesNotPanicOnRejectedButBoundedValue(t *testing.T) {
	m := &deepseekocr{}
	m.LanguageConfig.HiddenLayers = maxDeepSeekOCRHiddenLayers + 1
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Tensors() panicked despite the append-based rewrite: %v", r)
		}
	}()
	_ = m.Tensors(nil)
}
