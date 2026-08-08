package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/structured"
)

// TestConstraintBiasForcesAllowedToken proves the mask bias composes with
// the real sampler: logits strongly favor disallowed ids (including a
// padded position past the tokenizer vocab), yet sampling returns the one
// grammar-legal token.
func TestConstraintBiasForcesAllowedToken(t *testing.T) {
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	// Vocab pieces "5","6","7","8" (ids 0-3); grammar {"enum":[7]} allows
	// only "7" (id 2) at the start state.
	v := structured.NewVocab([][]byte{[]byte("5"), []byte("6"), []byte("7"), []byte("8")}, nil)
	g, err := structured.Compile([]byte(`{"enum":[7]}`))
	if err != nil {
		t.Fatal(err)
	}
	mask := v.Mask(g.NewMatcher())

	// Model logits are wider than the tokenizer vocab (padded embeddings):
	// id 4 is garbage that must stay masked.
	logits := mlx.FromValues([]float32{10, 8, -5, 6, 12}, 1, 5)

	var buf []float32
	bias, buf := constraintBias(mask, 5, buf)
	masked := mlx.Add(logits, bias)

	s := sample.New(4096)
	s.Add(0, sample.Options{Temperature: 0}, nil)
	defer s.Remove(0)
	res := s.Sample([]int{0}, masked)
	if got := res.Token.Int(); got != 2 {
		t.Fatalf("sampled token %d, want 2 (the only grammar-legal id)", got)
	}

	// Stochastic path must respect the mask too.
	s2 := sample.New(4096)
	s2.Add(1, sample.Options{Temperature: 1.0, Seed: 42, UseSeed: true}, nil)
	defer s2.Remove(1)
	bias2, _ := constraintBias(mask, 5, buf)
	masked2 := mlx.Add(logits, bias2)
	for range 8 {
		res := s2.Sample([]int{1}, masked2)
		if got := res.Token.Int(); got != 2 {
			t.Fatalf("stochastic sample returned %d, want 2", got)
		}
	}
}
