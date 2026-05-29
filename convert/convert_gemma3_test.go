package convert

import (
	"slices"
	"testing"
)

func TestGemma3TensorsWithTokenizerTruncatesPaddedEmbedding(t *testing.T) {
	p := gemma3Model{}
	embedding := &fakeTensor{
		name:  "token_embd.weight",
		shape: []uint64{5, 2},
		data:  []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
	}

	out := p.TensorsWithTokenizer([]Tensor{embedding}, &Tokenizer{
		Vocabulary: &Vocabulary{Tokens: []string{"a", "b", "<image>"}},
	})

	if len(out) != 1 {
		t.Fatalf("expected 1 tensor, got %d", len(out))
	}
	if got, want := out[0].Shape, []uint64{3, 2}; !slices.Equal(got, want) {
		t.Fatalf("token_embd.weight shape = %v, want %v", got, want)
	}

	got, err := embedding.repacker(embedding.name, embedding.data, embedding.shape)
	if err != nil {
		t.Fatalf("unexpected repacker error: %v", err)
	}
	if want := embedding.data[:6]; !slices.Equal(got, want) {
		t.Fatalf("truncated embedding = %v, want %v", got, want)
	}
}
