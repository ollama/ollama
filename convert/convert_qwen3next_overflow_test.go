package convert

import "testing"

// Same overflow-panic class as ollama/ollama#17177 (convert_deepseek2.go),
// found by auditing sibling converters that share the same merges := make +
// direct-index pattern -- here blockCount = NumHiddenLayers +
// NumNextNPredictLayers, so either summand can push the total over the
// bound. See convert_deepseek2_test.go for the full writeup.
func TestQwen3NextTensorsDoesNotPanicOnBoundedBlockCount(t *testing.T) {
	q := &qwen3NextModel{}
	q.NumHiddenLayers = maxQwen3NextBlockCount + 1
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Tensors() panicked despite the append-based rewrite: %v", r)
		}
	}()
	_ = q.Tensors(nil)
}
