package gemma4

import (
	"os"
	"testing"

	"github.com/ollama/ollama/x/models/testutil"
)

// TestPerplexitySmoke is a small CI regression check that the gemma4 model
// loads and produces a sane perplexity number on a few documents of
// WikiText-2. It uses the new generic perplexity core (testutil.RunPerplexity)
// in llamacpp mode, which limits forward-pass batch size to n_ctx and
// avoids the long-sequence forward bug tracked as task #46.
//
// This is NOT meant to validate quality — for that, run ollama-ppl
// (x/cmd/ppl) directly with -mode harness against an lm-evaluation-harness
// reference. This test only catches gross regressions: the model loads, the
// forward pass runs, and PPL is finite and not absurdly large.
//
// Set GEMMA4_OLLAMA_MODEL to override the default ollama model tag.
func TestPerplexitySmoke(t *testing.T) {
	testutil.SkipIfNoMLX(t)

	modelName := os.Getenv("GEMMA4_OLLAMA_MODEL")
	if modelName == "" {
		modelName = "gemma4:e2b-base-mlx-bf16"
	}

	m, cleanup, err := testutil.LoadModelByNameOrErr(modelName)
	if err != nil {
		t.Skipf("model %q not available: %v", modelName, err)
	}
	defer cleanup()

	// Build a self-contained synthetic corpus long enough for one
	// reasonably-sized chunk. We don't care about the absolute PPL value
	// here — just that the forward pass runs and the result is finite.
	docs := []testutil.Document{
		{Text: "The quick brown fox jumps over the lazy dog. " +
			"A bright sunny day in the meadow where flowers bloom and " +
			"birds sing their morning songs while a gentle breeze " +
			"carries the sweet scent of pine trees across rolling " +
			"hills toward distant mountains covered with fresh snow " +
			"beneath a clear blue sky. Children play near the old " +
			"stone bridge that crosses the winding river as it flows " +
			"toward the great ocean. Once upon a time, in a kingdom " +
			"far away, there lived a curious child who loved to read " +
			"books about science and history and faraway places. " +
			"Perplexity measures how well a language model predicts " +
			"a held-out sequence of tokens, computed as the exponential " +
			"of the average negative log likelihood per token."},
	}

	opts := testutil.PPLOptions{
		Mode:            testutil.ModeLlamaCpp, // bounded chunk size keeps memory predictable
		MaxLength:       64,                    // short context: smoke-test cost is minimal
		BOSSwapLlamaCpp: true,
	}
	result, err := testutil.RunPerplexity(m, docs, opts, testutil.NewWriterLogger(nil))
	if err != nil {
		t.Fatalf("RunPerplexity: %v", err)
	}

	if result.TotalTokens == 0 {
		t.Fatal("no tokens scored")
	}
	if result.TokenPerplexity <= 1.0 {
		t.Errorf("token PPL too low to be plausible: %f", result.TokenPerplexity)
	}
	if result.TokenPerplexity >= 100000.0 {
		t.Errorf("token PPL absurdly high: %f", result.TokenPerplexity)
	}
	t.Logf("smoke test PPL: %.4f over %d tokens", result.TokenPerplexity, result.TotalTokens)
}
