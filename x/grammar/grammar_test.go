//go:build mlx

package grammar

import (
	"testing"
)

func TestCompileSimpleGrammar(t *testing.T) {
	// Simple grammar: S = "a" "b" .
	grammar := `S = "a" "b" .`

	pda, err := compileString(grammar, "S")
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
	rt := newRuntime(pda)

	// Should accept "a" then "b"
	if !rt.Accept("a") {
		t.Error("should accept 'a'")
	}
	if !rt.Accept("b") {
		t.Error("should accept 'b'")
	}
	if !rt.isAccepting() {
		t.Error("should be in accepting state")
	}
}

func TestCompileAlternative(t *testing.T) {
	// Grammar: S = "a" | "b" .
	grammar := `S = "a" | "b" .`

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// Test accepting "a"
	rt := newRuntime(pda)
	if !rt.Accept("a") {
		t.Error("should accept 'a'")
	}
	if !rt.isAccepting() {
		t.Error("should be accepting after 'a'")
	}

	// Test accepting "b"
	rt.Reset()
	if !rt.Accept("b") {
		t.Error("should accept 'b'")
	}
	if !rt.isAccepting() {
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

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// Empty should be accepted (zero repetitions)
	rt := newRuntime(pda)
	if !rt.isAccepting() {
		t.Error("empty should be accepting")
	}

	// "a" should be accepted
	rt.Reset()
	if !rt.Accept("a") {
		t.Error("should accept first 'a'")
	}
	if !rt.isAccepting() {
		t.Error("should be accepting after one 'a'")
	}

	// "aa" should be accepted
	if !rt.Accept("a") {
		t.Error("should accept second 'a'")
	}
	if !rt.isAccepting() {
		t.Error("should be accepting after two 'a's")
	}
}

func TestCompileOption(t *testing.T) {
	// Grammar: S = ["a"] "b" .
	grammar := `S = ["a"] "b" .`

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// "b" alone should be accepted
	rt := newRuntime(pda)
	if !rt.Accept("b") {
		t.Error("should accept 'b' alone")
	}
	if !rt.isAccepting() {
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
	if !rt.isAccepting() {
		t.Error("should be accepting after 'ab'")
	}
}

func TestCompileRecursive(t *testing.T) {
	// Grammar with recursion: S = "(" S ")" | "x" .
	grammar := `S = "(" S ")" | "x" .`

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	// "x" should be accepted
	rt := newRuntime(pda)
	if !rt.Accept("x") {
		t.Error("should accept 'x'")
	}
	if !rt.isAccepting() {
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
	if !rt.isAccepting() {
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
	if !rt.isAccepting() {
		t.Error("should be accepting after '((x))'")
	}
}

func TestValidInputs(t *testing.T) {
	// Grammar: S = "a" | "b" .
	grammar := `S = "a" | "b" .`

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := newRuntime(pda)
	valid := rt.validInputs()

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

// TestValidInputsAfterAccept tests that validInputs returns correct values
// after accepting tokens, ensuring proper stack simulation.
func TestValidInputsAfterAccept(t *testing.T) {
	// Grammar: S = "a" "b" "c" .
	grammar := `S = "a" "b" "c" .`

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := newRuntime(pda)

	// Initially only "a" should be valid
	valid := rt.validInputs()
	if len(valid) != 1 || valid[0] != "a" {
		t.Errorf("initially expected only 'a', got %v", valid)
	}

	// After accepting "a", only "b" should be valid
	if !rt.Accept("a") {
		t.Fatal("failed to accept 'a'")
	}
	valid = rt.validInputs()
	if len(valid) != 1 || valid[0] != "b" {
		t.Errorf("after 'a', expected only 'b', got %v", valid)
	}

	// After accepting "b", only "c" should be valid
	if !rt.Accept("b") {
		t.Fatal("failed to accept 'b'")
	}
	valid = rt.validInputs()
	if len(valid) != 1 || valid[0] != "c" {
		t.Errorf("after 'ab', expected only 'c', got %v", valid)
	}
}

// TestValidInputsWithRepetitionInProduction tests the critical case where
// a repetition exists inside a called production. This requires proper
// stack simulation to determine when closing symbols are valid.
func TestValidInputsWithRepetitionInProduction(t *testing.T) {
	// Grammar similar to JSON:
	// S = "(" items ")" .
	// items = item { "," item } .
	// item = "x" .
	grammar := `
S = "(" items ")" .
items = item { "," item } .
item = "x" .
`
	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := newRuntime(pda)

	// Initially only "(" should be valid
	valid := rt.validInputs()
	if len(valid) != 1 || valid[0] != "(" {
		t.Errorf("initially expected only '(', got %v", valid)
	}

	// Accept "("
	if !rt.Accept("(") {
		t.Fatal("failed to accept '('")
	}
	// After "(", should be able to accept "x" (item)
	valid = rt.validInputs()
	hasX := false
	for _, v := range valid {
		if v == "x" {
			hasX = true
		}
	}
	if !hasX {
		t.Errorf("after '(', expected 'x' to be valid, got %v", valid)
	}

	// Accept first item "x"
	if !rt.Accept("x") {
		t.Fatal("failed to accept 'x'")
	}
	// After "(x", should be able to accept "," (more items) OR ")" (end)
	valid = rt.validInputs()
	hasComma, hasClose := false, false
	for _, v := range valid {
		if v == "," {
			hasComma = true
		}
		if v == ")" {
			hasClose = true
		}
	}
	if !hasComma {
		t.Errorf("after '(x', expected ',' to be valid, got %v", valid)
	}
	if !hasClose {
		t.Errorf("after '(x', expected ')' to be valid, got %v", valid)
	}

	// Accept comma for another item
	if !rt.Accept(",") {
		t.Fatal("failed to accept ','")
	}
	// After "(x,", should only be able to accept "x" (next item)
	valid = rt.validInputs()
	if len(valid) != 1 || valid[0] != "x" {
		t.Errorf("after '(x,', expected only 'x', got %v", valid)
	}

	// Accept second item "x"
	if !rt.Accept("x") {
		t.Fatal("failed to accept second 'x'")
	}
	// CRITICAL: After "(x,x", should be able to accept "," OR ")"
	// This tests the stack simulation fix - we need to properly
	// follow epsilon transitions through the production call stack.
	valid = rt.validInputs()
	hasComma, hasClose = false, false
	for _, v := range valid {
		if v == "," {
			hasComma = true
		}
		if v == ")" {
			hasClose = true
		}
	}
	if !hasComma {
		t.Errorf("after '(x,x', expected ',' to be valid, got %v", valid)
	}
	if !hasClose {
		t.Errorf("after '(x,x', expected ')' to be valid, got %v", valid)
	}

	// Close with ")"
	if !rt.Accept(")") {
		t.Fatal("failed to accept ')'")
	}
	if !rt.isAccepting() {
		t.Error("should be accepting after '(x,x)'")
	}
}

// TestValidInputsNestedCalls tests validInputs with deeply nested production calls.
func TestValidInputsNestedCalls(t *testing.T) {
	// Grammar: A = "start" B "end" .  B = "middle" .
	grammar := `
A = "start" B "end" .
B = "middle" .
`
	pda, err := compileString(grammar, "A")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := newRuntime(pda)

	// After "start", should accept "middle" (from B)
	rt.Accept("start")
	valid := rt.validInputs()
	if len(valid) != 1 || valid[0] != "middle" {
		t.Errorf("after 'start', expected 'middle', got %v", valid)
	}

	// After "start middle", should accept "end"
	rt.Accept("middle")
	valid = rt.validInputs()
	if len(valid) != 1 || valid[0] != "end" {
		t.Errorf("after 'start middle', expected 'end', got %v", valid)
	}
}

func TestReturnAddressDisambiguation(t *testing.T) {
	// Grammar where the same production is called from different contexts:
	// S = A "x" | "c" A "y" .
	// A = "a" .
	grammar := `
S = A "x" | "c" A "y" .
A = "a" .
`
	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := newRuntime(pda)

	if !rt.Accept("c") {
		t.Fatal("failed to accept 'c'")
	}
	if !rt.Accept("a") {
		t.Fatal("failed to accept 'a'")
	}

	valid := rt.validInputs()
	if len(valid) != 1 || valid[0] != "y" {
		t.Errorf("after 'ca', expected only 'y', got %v", valid)
	}

	rt.Reset()
	rt.Accept("c")
	rt.Accept("a")
	if rt.Accept("x") {
		t.Error("should not accept 'x' after 'ca'")
	}
}

// TestValidInputsRecursiveWithStack tests validInputs with recursive grammars
// which heavily exercise the stack simulation.
func TestValidInputsRecursiveWithStack(t *testing.T) {
	// Grammar: S = "(" S ")" | "x" .
	grammar := `S = "(" S ")" | "x" .`

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := newRuntime(pda)

	// Initially: "(" or "x" should be valid
	valid := rt.validInputs()
	hasParen, hasX := false, false
	for _, v := range valid {
		if v == "(" {
			hasParen = true
		}
		if v == "x" {
			hasX = true
		}
	}
	if !hasParen || !hasX {
		t.Errorf("initially expected '(' and 'x', got %v", valid)
	}

	// After "(": "(" or "x" should be valid (nested S)
	rt.Accept("(")
	valid = rt.validInputs()
	hasParen, hasX = false, false
	for _, v := range valid {
		if v == "(" {
			hasParen = true
		}
		if v == "x" {
			hasX = true
		}
	}
	if !hasParen || !hasX {
		t.Errorf("after '(', expected '(' and 'x', got %v", valid)
	}

	// After "((": "(" or "x" should still be valid
	rt.Accept("(")
	valid = rt.validInputs()
	hasParen, hasX = false, false
	for _, v := range valid {
		if v == "(" {
			hasParen = true
		}
		if v == "x" {
			hasX = true
		}
	}
	if !hasParen || !hasX {
		t.Errorf("after '((', expected '(' and 'x', got %v", valid)
	}

	// After "((x": only ")" should be valid
	rt.Accept("x")
	valid = rt.validInputs()
	if len(valid) != 1 || valid[0] != ")" {
		t.Errorf("after '((x', expected only ')', got %v", valid)
	}

	// After "((x)": only ")" should be valid (closing outer)
	rt.Accept(")")
	valid = rt.validInputs()
	if len(valid) != 1 || valid[0] != ")" {
		t.Errorf("after '((x)', expected only ')', got %v", valid)
	}
}

// TestRejectionAfterValid tests that invalid inputs are rejected
// at various points in the grammar.
func TestRejectionAfterValid(t *testing.T) {
	// Grammar: S = "a" "b" .
	grammar := `S = "a" "b" .`

	pda, err := compileString(grammar, "S")
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}

	rt := newRuntime(pda)

	// "b" should be rejected initially
	if rt.Accept("b") {
		t.Error("'b' should be rejected initially")
	}

	// Accept "a"
	rt.Accept("a")

	// "a" should be rejected after "a"
	if rt.Accept("a") {
		t.Error("'a' should be rejected after 'a'")
	}

	// "c" should be rejected (not in grammar)
	if rt.Accept("c") {
		t.Error("'c' should be rejected (not in grammar)")
	}
}
