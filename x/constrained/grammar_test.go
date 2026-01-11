//go:build mlx

package constrained

import (
	"testing"
)

func TestCompileSimpleGrammar(t *testing.T) {
	// Simple grammar: S = "a" "b" .
	grammar := `S = "a" "b" .`

	pda, err := CompileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	if pda == nil {
		t.Fatal("pda is nil")
	}

	// Should have terminals "a" and "b"
	if len(pda.Terminals) != 2 {
		t.Errorf("expected 2 terminals, got %d: %v", len(pda.Terminals), pda.Terminals)
	}

	// Test runtime
	rt := NewRuntime(pda)

	// Should accept "a" then "b"
	if !rt.Accept("a") {
		t.Error("should accept 'a'")
	}
	if !rt.Accept("b") {
		t.Error("should accept 'b'")
	}
	if !rt.IsAccepting() {
		t.Error("should be in accepting state")
	}
}

func TestCompileAlternative(t *testing.T) {
	// Grammar: S = "a" | "b" .
	grammar := `S = "a" | "b" .`

	pda, err := CompileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// Test accepting "a"
	rt := NewRuntime(pda)
	if !rt.Accept("a") {
		t.Error("should accept 'a'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after 'a'")
	}

	// Test accepting "b"
	rt.Reset()
	if !rt.Accept("b") {
		t.Error("should accept 'b'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after 'b'")
	}

	// Test rejecting "c"
	rt.Reset()
	if rt.Accept("c") {
		t.Error("should not accept 'c'")
	}
}

func TestCompileRepetition(t *testing.T) {
	// Grammar: S = {"a"} .
	grammar := `S = {"a"} .`

	pda, err := CompileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// Empty should be accepted (zero repetitions)
	rt := NewRuntime(pda)
	if !rt.IsAccepting() {
		t.Error("empty should be accepting")
	}

	// "a" should be accepted
	rt.Reset()
	if !rt.Accept("a") {
		t.Error("should accept first 'a'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after one 'a'")
	}

	// "aa" should be accepted
	if !rt.Accept("a") {
		t.Error("should accept second 'a'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after two 'a's")
	}
}

func TestCompileOption(t *testing.T) {
	// Grammar: S = ["a"] "b" .
	grammar := `S = ["a"] "b" .`

	pda, err := CompileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// "b" alone should be accepted
	rt := NewRuntime(pda)
	if !rt.Accept("b") {
		t.Error("should accept 'b' alone")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after 'b'")
	}

	// "ab" should be accepted
	rt.Reset()
	if !rt.Accept("a") {
		t.Error("should accept 'a'")
	}
	if !rt.Accept("b") {
		t.Error("should accept 'b' after 'a'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after 'ab'")
	}
}

func TestCompileRecursive(t *testing.T) {
	// Grammar with recursion: S = "(" S ")" | "x" .
	grammar := `S = "(" S ")" | "x" .`

	pda, err := CompileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// "x" should be accepted
	rt := NewRuntime(pda)
	if !rt.Accept("x") {
		t.Error("should accept 'x'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after 'x'")
	}

	// "(x)" should be accepted
	rt.Reset()
	if !rt.Accept("(") {
		t.Error("should accept '('")
	}
	if !rt.Accept("x") {
		t.Error("should accept 'x' inside parens")
	}
	if !rt.Accept(")") {
		t.Error("should accept ')'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after '(x)'")
	}

	// "((x))" should be accepted
	rt.Reset()
	if !rt.Accept("(") {
		t.Error("should accept first '('")
	}
	if !rt.Accept("(") {
		t.Error("should accept second '('")
	}
	if !rt.Accept("x") {
		t.Error("should accept 'x'")
	}
	if !rt.Accept(")") {
		t.Error("should accept first ')'")
	}
	if !rt.Accept(")") {
		t.Error("should accept second ')'")
	}
	if !rt.IsAccepting() {
		t.Error("should be accepting after '((x))'")
	}
}

func TestValidInputs(t *testing.T) {
	// Grammar: S = "a" | "b" .
	grammar := `S = "a" | "b" .`

	pda, err := CompileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := NewRuntime(pda)
	valid := rt.ValidInputs()

	// Should have both "a" and "b" as valid
	hasA, hasB := false, false
	for _, v := range valid {
		if v == "a" {
			hasA = true
		}
		if v == "b" {
			hasB = true
		}
	}

	if !hasA {
		t.Error("'a' should be valid input")
	}
	if !hasB {
		t.Error("'b' should be valid input")
	}
}
