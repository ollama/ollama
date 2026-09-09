package convert

import "testing"

// Same overflow-panic class as ollama/ollama#17177 (convert_deepseek2.go),
// found by auditing sibling converters that share the same merges := make +
// direct-index pattern. See convert_deepseek2_test.go for the full writeup.
func TestGLM4MoeLiteValidateRejectsOverflowInducingHiddenLayers(t *testing.T) {
	p := &glm4MoeLiteModel{HiddenLayers: 1431655766} // 3*1431655766 wraps to 2 (uint32)
	if err := p.validate(); err == nil {
		t.Fatal("validate() = nil, want an error for an overflow-inducing num_hidden_layers")
	}
}

func TestGLM4MoeLiteValidateRejectsZeroHiddenLayers(t *testing.T) {
	p := &glm4MoeLiteModel{HiddenLayers: 0}
	if err := p.validate(); err == nil {
		t.Fatal("validate() = nil, want an error for num_hidden_layers=0")
	}
}

func TestGLM4MoeLiteTensorsDoesNotPanicOnRejectedButBoundedValue(t *testing.T) {
	p := &glm4MoeLiteModel{HiddenLayers: maxGLM4MoeLiteHiddenLayers + 1}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Tensors() panicked despite the append-based rewrite: %v", r)
		}
	}()
	_ = p.Tensors(nil)
}
