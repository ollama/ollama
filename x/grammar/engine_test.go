//go:build mlx

package grammar

import (
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// newTestEngine creates a JSON engine for testing
func newTestEngine(t testing.TB, vocab []string) *Engine {
	t.Helper()
	grammar, err := JSONGrammar()
	if err != nil {
		t.Fatalf("failed to create JSON grammar: %v", err)
	}
	e, err := NewEngine(grammar, vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	return e
}

// Mock vocabulary for testing
func testVocab() []string {
	return []string{
		"{",       // 0: object start
		"}",       // 1: object end
		"[",       // 2: array start
		"]",       // 3: array end
		":",       // 4: colon
		",",       // 5: comma
		"\"key\"", // 6: string (quoted)
		"\"val\"", // 7: string (quoted)
		"123",     // 8: number
		"-42.5",   // 9: number
		"true",    // 10: boolean
		"false",   // 11: boolean
		"null",    // 12: null
		" ",       // 13: whitespace (should be ignored)
		"\n",      // 14: whitespace (should be ignored)
		"subword", // 15: bare word (NOT valid JSON - requires quotes)
		"hello",   // 16: bare word (NOT valid JSON - requires quotes)
	}
}

func TestNewEngine(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	if e.vocabSize != int32(len(vocab)) {
		t.Errorf("vocabSize = %d, want %d", e.vocabSize, len(vocab))
	}

	// Verify grammar is set
	if e.grammar == nil {
		t.Error("grammar should not be nil")
	}

	// Verify analyzer is set
	if e.analyzer == nil {
		t.Error("analyzer should not be nil")
	}
}

func TestEngineValidTokens(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// At start, any value type should be valid
	validTokens := e.validTokens()

	// Should include object start, array start, strings, numbers, booleans, null
	// Note: bare words like "subword" and "hello" are NOT valid JSON strings
	// (JSON strings must be quoted)
	expectedTokens := map[int]bool{
		0:  true, // {
		2:  true, // [
		6:  true, // "key"
		7:  true, // "val"
		8:  true, // 123
		9:  true, // -42.5
		10: true, // true
		11: true, // false
		12: true, // null
	}

	// Check that expected tokens are present
	validSet := make(map[int]bool)
	for _, idx := range validTokens {
		validSet[idx] = true
	}

	for idx := range expectedTokens {
		if !validSet[idx] {
			t.Errorf("expected token %d (%s) to be valid", idx, vocab[idx])
		}
	}

	if validSet[15] || validSet[16] {
		t.Error("bare words should not be valid JSON at the start state")
	}
}

func TestEngineAccept(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Accept { should work
	if !e.Accept(0) { // {
		t.Error("should accept {")
	}

	// After {, valid tokens should be STRING or }
	validTokens := e.validTokens()

	validSet := make(map[int]bool)
	for _, idx := range validTokens {
		validSet[idx] = true
	}

	// STRING tokens (indices 6, 7) and } (index 1) should be valid
	if !validSet[1] {
		t.Error("} should be valid after {")
	}
	if !validSet[6] && !validSet[7] {
		t.Error("STRING should be valid after { (for keys)")
	}
}

func TestEngineAcceptSequence(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Accept {"key": "val"}
	sequence := []int{0, 6, 4, 7, 1} // {, "key", :, "val", }

	for i, tokenID := range sequence {
		if !e.Accept(tokenID) {
			t.Fatalf("failed to accept token %d (%s) at position %d",
				tokenID, vocab[tokenID], i)
		}
	}

	if !e.IsComplete() {
		t.Error("should be in complete state after valid JSON")
	}
}

func TestEngineReset(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Accept some tokens
	e.Accept(0) // {
	e.Accept(1) // }

	if !e.IsComplete() {
		t.Error("should be complete after {}")
	}

	// Reset
	e.Reset()

	// Should be back to initial state
	if e.IsComplete() {
		t.Error("should not be complete after reset")
	}

	// Should be able to accept new sequence
	if !e.Accept(0) { // {
		t.Error("should accept { after reset")
	}
}

func TestEngineInvalidTokenRejection(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Accept { first
	if !e.Accept(0) {
		t.Fatal("should accept {")
	}

	// Now try to accept [ which is invalid after {
	// (After {, only STRING or } are valid)
	if e.Accept(2) { // [
		t.Error("should not accept [ after { (expecting STRING or })")
	}
}

func TestEngineAcceptString(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Accept using string directly
	if !e.AcceptString("{") {
		t.Error("should accept {")
	}
	if !e.AcceptString("\"key\"") {
		t.Error("should accept string key")
	}
	if !e.AcceptString(":") {
		t.Error("should accept :")
	}
	if !e.AcceptString("123") {
		t.Error("should accept number")
	}
	if !e.AcceptString("}") {
		t.Error("should accept }")
	}

	if !e.IsComplete() {
		t.Error("should be complete after valid JSON")
	}
}

func TestJSONBackslashEscape(t *testing.T) {
	vocab := []string{`"`, `\`, "n", "a"}
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Valid escape: "\n"
	if !e.AcceptString(`"`) {
		t.Fatal("should accept string start")
	}
	if !e.AcceptString(`\`) {
		t.Fatal("should accept escape prefix")
	}
	if !e.AcceptString("n") {
		t.Fatal("should accept escape code")
	}
	if !e.AcceptString(`"`) {
		t.Fatal("should accept string end")
	}
	if !e.IsComplete() {
		t.Error("should be complete after escaped string")
	}

	// Invalid escape: "\a"
	e.Reset()
	if !e.AcceptString(`"`) {
		t.Fatal("should accept string start")
	}
	if !e.AcceptString(`\`) {
		t.Fatal("should accept escape prefix")
	}
	if e.AcceptString("a") {
		t.Error("should reject invalid escape code")
	}
}

func TestEngineNegInfMask(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Verify negInfMask exists and has correct shape
	if e.negInfMask == nil {
		t.Fatal("negInfMask should not be nil")
	}
}

func TestEngineMaskCache(t *testing.T) {
	vocab := testVocab()
	e := newTestEngine(t, vocab)
	defer e.Close()

	// Create test logits
	logits := mlx.Ones(int32(len(vocab)))

	// Apply mask - should populate cache
	_ = e.ApplyMask(logits)

	// Check cache was populated
	cacheSize := e.maskCache.size()
	if cacheSize == 0 {
		t.Error("mask cache should have at least one entry after ApplyMask")
	}
}

func TestEngineEmptyVocab(t *testing.T) {
	e := newTestEngine(t, []string{})
	defer e.Close()

	if e.vocabSize != 0 {
		t.Errorf("vocabSize = %d, want 0", e.vocabSize)
	}
}

func TestEngineLargeVocab(t *testing.T) {
	// Create a large vocabulary (simulating real model vocab)
	vocab := make([]string, 32000)
	for i := range vocab {
		vocab[i] = "token"
	}
	// Add some actual JSON tokens
	vocab[0] = "{"
	vocab[1] = "}"
	vocab[2] = "["
	vocab[3] = "]"
	vocab[4] = ":"
	vocab[5] = ","
	vocab[6] = "\"test\""
	vocab[7] = "123"
	vocab[8] = "true"
	vocab[9] = "false"
	vocab[10] = "null"

	e := newTestEngine(t, vocab)
	defer e.Close()

	if e.vocabSize != 32000 {
		t.Errorf("vocabSize = %d, want 32000", e.vocabSize)
	}

	// Test that it still works correctly
	if !e.Accept(0) { // {
		t.Error("should accept {")
	}
	if !e.Accept(1) { // }
		t.Error("should accept }")
	}
	if !e.IsComplete() {
		t.Error("should be complete after {}")
	}
}

// TestE2E_JSONDecoding tests end-to-end JSON constrained decoding.
func TestE2E_JSONDecoding(t *testing.T) {
	// Create a realistic vocabulary with JSON tokens
	vocab := []string{
		// Structural tokens
		"{", "}", "[", "]", ":", ",",
		// Keywords
		"true", "false", "null",
		// Quoted strings
		`"name"`, `"value"`, `"items"`, `"count"`, `"enabled"`,
		`"hello"`, `"world"`, `"test"`,
		// Numbers
		"0", "1", "2", "3", "42", "123", "-1", "-42",
		// Whitespace
		" ", "\n", "\t",
		// Multi-terminal tokens (span multiple JSON lexemes)
		`"key":`, `},`, `],`, `{"`, `["`,
		// Partial/invalid tokens (should be rejected)
		"invalid", "foo", "bar",
	}

	grammar, err := JSONGrammar()
	if err != nil {
		t.Fatalf("failed to create JSON grammar: %v", err)
	}

	engine, err := NewEngine(grammar, vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Close()

	tests := []struct {
		name     string
		tokens   []string
		wantPass bool
	}{
		// Simple values
		{"empty object", []string{"{", "}"}, true},
		{"empty array", []string{"[", "]"}, true},
		{"true literal", []string{"true"}, true},
		{"null literal", []string{"null"}, true},
		{"number", []string{"42"}, true},
		{"negative number", []string{"-42"}, true},
		{"quoted string", []string{`"hello"`}, true},

		// Objects
		{"simple object", []string{"{", `"name"`, ":", `"value"`, "}"}, true},
		{"object with single-digit numbers", []string{"{", `"count"`, ":", "1", ",", `"value"`, ":", "2", "}"}, true},
		{"multi-terminal key", []string{"{", `"key":`, `"value"`, "}"}, true},

		// Arrays
		{"array of numbers", []string{"[", "42", "]"}, true},
		{"array of single digits", []string{"[", "1", ",", "2", "]"}, true},
		{"array of strings", []string{"[", `"hello"`, ",", `"world"`, "]"}, true},
		{"nested array", []string{"[", "[", "42", "]", "]"}, true},

		// Nested structures
		{"nested object", []string{"{", `"items"`, ":", "{", `"count"`, ":", "42", "}", "}"}, true},
		{"object with array", []string{"{", `"items"`, ":", "[", "42", "]", "}"}, true},

		// Invalid sequences
		{"unclosed object", []string{"{", `"name"`, ":"}, false},          // incomplete
		{"double comma", []string{"[", "42", ",", ",", "42", "]"}, false}, // invalid
		{"missing value", []string{"{", `"name"`, ":", "}"}, false},       // missing value
		{"bare word", []string{"invalid"}, false},                         // not valid JSON
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine.Reset()

			// Process each token
			allAccepted := true
			for i, token := range tt.tokens {
				if !engine.AcceptString(token) {
					if tt.wantPass {
						t.Errorf("token %d (%q) rejected unexpectedly", i, token)
					}
					allAccepted = false
					break
				}
			}

			if tt.wantPass {
				if !allAccepted {
					return // Already reported error
				}
				if !engine.IsComplete() {
					t.Errorf("expected complete parse, but not in accepting state")
				}
			} else {
				// For invalid sequences, we expect either rejection or incomplete
				if allAccepted && engine.IsComplete() {
					t.Errorf("expected rejection or incomplete, but parse succeeded")
				}
			}
		})
	}
}

// TestE2E_SimpleExpressionGrammar tests a custom expression grammar.
func TestE2E_SimpleExpressionGrammar(t *testing.T) {
	// Simple expression grammar: expr = term { ("+" | "-") term }
	// term = number | "(" expr ")"
	// number = digit { digit }
	// digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
	exprGrammar := `
		expr = term { addop term } .
		addop = "+" | "-" .
		term = factor { mulop factor } .
		mulop = "*" | "/" .
		factor = number | "(" expr ")" .
		number = digit { digit } .
		digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" .
	`

	grammar, err := ParseEBNF(exprGrammar, "expr")
	if err != nil {
		t.Fatalf("failed to parse expression grammar: %v", err)
	}

	// Vocabulary for expression tokens
	vocab := []string{
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
		"+", "-", "*", "/",
		"(", ")",
		// Multi-digit numbers as single tokens
		"10", "42", "100", "123",
		// Invalid tokens
		"x", "y", "invalid",
	}

	engine, err := NewEngine(grammar, vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Close()

	tests := []struct {
		name     string
		tokens   []string
		wantPass bool
	}{
		{"single digit", []string{"5"}, true},
		{"multi-digit", []string{"1", "2", "3"}, true},
		{"addition", []string{"1", "+", "2"}, true},
		{"subtraction", []string{"5", "-", "3"}, true},
		{"multiplication", []string{"2", "*", "3"}, true},
		{"division", []string{"8", "/", "2"}, true},
		{"complex expr", []string{"1", "+", "2", "*", "3"}, true},
		{"parentheses", []string{"(", "1", "+", "2", ")", "*", "3"}, true},
		{"nested parens", []string{"(", "(", "1", ")", ")"}, true},

		// Invalid
		{"just operator", []string{"+"}, false},
		{"double operator", []string{"1", "+", "+", "2"}, false},
		{"unclosed paren", []string{"(", "1", "+", "2"}, false},
		{"variable", []string{"x"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine.Reset()

			allAccepted := true
			for i, token := range tt.tokens {
				if !engine.AcceptString(token) {
					if tt.wantPass {
						t.Errorf("token %d (%q) rejected unexpectedly", i, token)
					}
					allAccepted = false
					break
				}
			}

			if tt.wantPass {
				if !allAccepted {
					return
				}
				if !engine.IsComplete() {
					t.Errorf("expected complete parse, but not in accepting state")
				}
			} else {
				if allAccepted && engine.IsComplete() {
					t.Errorf("expected rejection or incomplete, but parse succeeded")
				}
			}
		})
	}
}

// TestE2E_IdentifierGrammar tests a grammar with character ranges.
func TestE2E_IdentifierGrammar(t *testing.T) {
	// Identifier grammar using character ranges
	identGrammar := `
		ident = letter { letter | digit } .
		letter = "a" … "z" | "A" … "Z" | "_" .
		digit = "0" … "9" .
	`

	grammar, err := ParseEBNF(identGrammar, "ident")
	if err != nil {
		t.Fatalf("failed to parse identifier grammar: %v", err)
	}

	// Vocabulary with letters and digits
	vocab := []string{
		"a", "b", "c", "x", "y", "z",
		"A", "B", "C", "X", "Y", "Z",
		"_",
		"0", "1", "2", "9",
		// Multi-char tokens
		"foo", "bar", "myVar", "test123",
		// Invalid starting chars
		"1abc", "123",
	}

	engine, err := NewEngine(grammar, vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Close()

	tests := []struct {
		name     string
		tokens   []string
		wantPass bool
	}{
		{"single letter", []string{"a"}, true},
		{"uppercase", []string{"A"}, true},
		{"underscore", []string{"_"}, true},
		{"multi-letter", []string{"a", "b", "c"}, true},
		{"letter then digit", []string{"x", "1"}, true},
		{"underscore prefix", []string{"_", "a", "1"}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine.Reset()

			allAccepted := true
			for i, token := range tt.tokens {
				if !engine.AcceptString(token) {
					if tt.wantPass {
						t.Errorf("token %d (%q) rejected unexpectedly", i, token)
					}
					allAccepted = false
					break
				}
			}

			if tt.wantPass && allAccepted && !engine.IsComplete() {
				t.Errorf("expected complete parse, but not in accepting state")
			}
		})
	}
}

// TestE2E_UnicodeRange ensures unicode ranges compile and match tokens.
func TestE2E_UnicodeRange(t *testing.T) {
	greekGrammar := `
		greek = "α" … "ω" .
	`

	grammar, err := ParseEBNF(greekGrammar, "greek")
	if err != nil {
		t.Fatalf("failed to parse unicode grammar: %v", err)
	}

	vocab := []string{"α", "β", "ω", "a"}
	engine, err := NewEngine(grammar, vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Close()

	if !engine.AcceptString("β") {
		t.Error("should accept beta")
	}
	if !engine.IsComplete() {
		t.Error("should be complete after single rune")
	}

	engine.Reset()
	if engine.AcceptString("a") {
		t.Error("should reject ASCII outside unicode range")
	}
}

// TestE2E_NondeterminismPreserved tests that nondeterministic paths are preserved.
func TestE2E_NondeterminismPreserved(t *testing.T) {
	// This grammar has nondeterminism: "ab" could be parsed as
	// a single token or as two tokens "a" "b"
	ambiguousGrammar := `
		start = item item .
		item = "a" | "b" | "ab" .
	`

	grammar, err := ParseEBNF(ambiguousGrammar, "start")
	if err != nil {
		t.Fatalf("failed to parse grammar: %v", err)
	}

	// Vocabulary with both single and combined tokens
	vocab := []string{"a", "b", "ab"}

	engine, err := NewEngine(grammar, vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer engine.Close()

	// Test: "ab" "a" should be valid (ab as first item, a as second)
	t.Run("ab then a", func(t *testing.T) {
		engine.Reset()
		if !engine.AcceptString("ab") {
			t.Error("should accept ab")
		}
		if !engine.AcceptString("a") {
			t.Error("should accept a after ab")
		}
		if !engine.IsComplete() {
			t.Error("should be complete")
		}
	})

	t.Run("a then ab", func(t *testing.T) {
		engine.Reset()
		if !engine.AcceptString("a") {
			t.Error("should accept a")
		}
		if !engine.AcceptString("ab") {
			t.Error("should accept ab after a")
		}
		if !engine.IsComplete() {
			t.Error("should be complete")
		}
	})

	t.Run("a then a", func(t *testing.T) {
		engine.Reset()
		if !engine.AcceptString("a") {
			t.Error("should accept first a")
		}
		if !engine.AcceptString("a") {
			t.Error("should accept second a")
		}
		if !engine.IsComplete() {
			t.Error("should be complete")
		}
	})
}
