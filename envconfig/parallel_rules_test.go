package envconfig

import (
	"testing"
)

func TestParallelRulesOrder(t *testing.T) {
	// Valid YAML with ordered rules.
	yamlValid := `
- pattern: "gemma.*"
  count: 3
- pattern: "gemma3:.*"
  count: 2
`

	NumParallelForModel.Set(yamlValid)
	parallel := NumParallelForModel.Get("gemma3:1b")
	if parallel != 3 {
		t.Fatalf("expected first rule to apply (value 3), got %d", parallel)
	}

	// Test reading a model not present in the rules – should fall back to the global NumParallel.
	if fallback := NumParallelForModel.Get("unknownmodel"); fallback != NumParallel() {
		t.Fatalf("expected fallback to global NumParallel for unknown model, got %d", fallback)
	}

	// Invalid YAML should fall back to global NumParallel (default 1).
	yamlInvalid := `invalid yaml`

	NumParallelForModel.Set(yamlInvalid)
	parallel = NumParallelForModel.Get("anymodel")
	if parallel != NumParallel() {
		t.Fatalf("expected fallback to global NumParallel, got %d", parallel)
	}
}

func TestInvalidRegexpAndEmptyRules(t *testing.T) {
	// --- Invalid regular expression ---
	invalidYAML := `
- pattern: "*invalid["
  count: 5
`
	NumParallelForModel.Set(invalidYAML)

	// Should fall back to the global value because the rule is invalid.
	if got := NumParallelForModel.Get("anymodel"); got != NumParallel() {
		t.Fatalf("expected fallback to global NumParallel on invalid regexp, got %d", got)
	}

	// --- Empty OLLAMA_NUM_PARALLEL_RULES ---
	NumParallelForModel.Set("")
	if got := NumParallelForModel.Get("modelX"); got != NumParallel() {
		t.Fatalf("expected fallback to global NumParallel when rules are empty, got %d", got)
	}
}

func TestRawMethodValidYAML(t *testing.T) {
	// Valid YAML should be stored and returned unchanged.
	yamlValid := `
- pattern: "gemma.*"
  count: 3
- pattern: "gemma3:.*"
  count: 2
`
	NumParallelForModel.Set(yamlValid)
	if got := NumParallelForModel.Raw(); got != yamlValid {
		t.Fatalf("expected Raw() to return the original YAML, got %q", got)
	}
}

func TestRawMethodInvalidYAML(t *testing.T) {
	// Invalid YAML should result in Raw() returning "[]" (fallback value).
	NumParallelForModel.Set(`invalid yaml`)
	if got := NumParallelForModel.Raw(); got != "[]" {
		t.Fatalf("expected Raw() to return fallback \"[]\" on parse error, got %q", got)
	}
}
