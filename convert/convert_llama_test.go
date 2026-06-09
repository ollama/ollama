package convert

import "testing"

func TestLlama3RopeFactorsTensorDoesNotDependOnKVOrder(t *testing.T) {
	m := &llamaModel{
		HiddenSize:        2048,
		NumAttentionHeads: 32,
		RopeTheta:         500000,
	}
	m.RopeScaling.RopeType = "llama3"
	m.RopeScaling.Factor = 32
	m.RopeScaling.LowFrequencyFactor = 1
	m.RopeScaling.HighFrequencyFactor = 4
	m.RopeScaling.OriginalMaxPositionEmbeddings = 8192

	tensors := m.Tensors(nil)
	if len(tensors) != 1 {
		t.Fatalf("expected rope tensor only, got %d tensors", len(tensors))
	}
	if tensors[0].Name != "rope_freqs.weight" {
		t.Fatalf("expected rope_freqs.weight, got %q", tensors[0].Name)
	}
	if len(tensors[0].Shape) != 1 || tensors[0].Shape[0] != 32 {
		t.Fatalf("expected rope tensor shape [32], got %v", tensors[0].Shape)
	}

	_ = m.KV(&Tokenizer{Vocabulary: &Vocabulary{}})

	afterKV := m.Tensors(nil)
	if len(afterKV) != 1 || afterKV[0].Name != "rope_freqs.weight" {
		t.Fatalf("expected one rope tensor after KV call, got %#v", afterKV)
	}
}
