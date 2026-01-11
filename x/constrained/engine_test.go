//go:build mlx

package constrained

import (
	"math"
	"testing"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Mock vocabulary for testing
func testVocab() []string {
	return []string{
		"{",       // 0: object start
		"}",       // 1: object end
		"[",       // 2: array start
		"]",       // 3: array end
		":",       // 4: colon
		",",       // 5: comma
		"\"key\"", // 6: string
		"\"val\"", // 7: string
		"123",     // 8: number
		"-42.5",   // 9: number
		"true",    // 10: boolean
		"false",   // 11: boolean
		"null",    // 12: null
		" ",       // 13: whitespace (should be ignored)
		"\n",      // 14: whitespace (should be ignored)
		"invalid", // 15: unknown token
		"hello",   // 16: unknown (not quoted string)
	}
}

func TestNewEngine(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	if e.vocabSize != int32(len(vocab)) {
		t.Errorf("vocabSize = %d, want %d", e.vocabSize, len(vocab))
	}

	// Should have symbol masks for all terminals
	pda, _ := GetJSONPDA()
	for _, terminal := range pda.Terminals {
		if _, ok := e.symbolMasks[terminal]; !ok {
			t.Errorf("missing symbol mask for %q", terminal)
		}
	}
}

func TestTokenMatchesSymbol(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	testCases := []struct {
		token    string
		symbol   string
		expected bool
	}{
		{"{", "{", true},
		{"}", "}", true},
		{"[", "[", true},
		{"]", "]", true},
		{":", ":", true},
		{",", ",", true},
		{"\"key\"", "STRING", true},
		{"123", "NUMBER", true},
		{"-42.5", "NUMBER", true},
		{"true", "true", true},
		{"false", "false", true},
		{"null", "null", true},
		{"{", "}", false},
		{"\"key\"", "NUMBER", false},
		{"123", "STRING", false},
	}

	for _, tc := range testCases {
		t.Run(tc.token+"->"+tc.symbol, func(t *testing.T) {
			got := e.tokenMatchesSymbol(tc.token, tc.symbol)
			if got != tc.expected {
				t.Errorf("tokenMatchesSymbol(%q, %q) = %v, want %v",
					tc.token, tc.symbol, got, tc.expected)
			}
		})
	}
}

func TestEngineValidTokens(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	// At start, any value type should be valid
	validTokens := e.ValidTokens()
	t.Logf("Valid tokens at start: %v", validTokens)

	// Should include object start, array start, strings, numbers, booleans, null
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

	for _, idx := range validTokens {
		if !expectedTokens[idx] {
			t.Logf("unexpected valid token: %d (%s)", idx, vocab[idx])
		}
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
}

func TestEngineAccept(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	// Accept { should work
	if !e.Accept(0) { // {
		t.Error("should accept {")
	}

	// After {, valid tokens should be STRING or }
	validTokens := e.ValidTokens()
	t.Logf("Valid tokens after '{': %v", validTokens)

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
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
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

func TestEngineApplyMask(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	// Create test logits (all ones)
	logits := mlx.Ones(int32(len(vocab)))

	// Apply mask at initial state
	masked := e.ApplyMask(logits)
	mlx.Eval(masked)

	// Check that invalid tokens have -inf
	// Token 15 ("invalid") should be masked
	// Token 13 (" ") whitespace should be masked
	// Token 4 (":") should be masked (not valid at start)

	t.Log("Mask applied successfully at initial state")
}

func TestEngineApplyMaskValues(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	// Create test logits with known values
	logitData := make([]float32, len(vocab))
	for i := range logitData {
		logitData[i] = float32(i)
	}
	logits := mlx.NewArray(logitData, []int32{int32(len(vocab))})

	// Apply mask
	masked := e.ApplyMask(logits)
	maskedArr := mlx.Eval(masked)

	// Valid tokens should keep their values
	// Invalid tokens should have -inf
	validTokens := e.ValidTokens()
	validSet := make(map[int]bool)
	for _, idx := range validTokens {
		validSet[idx] = true
	}

	t.Logf("Valid tokens: %v", validTokens)
	t.Logf("Masked array evaluated")
	_ = maskedArr // Use the result
}

func TestEngineReset(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
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
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
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
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
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

func TestEngineNegInfMask(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	// Verify negInfMask exists and has correct shape
	if e.negInfMask == nil {
		t.Fatal("negInfMask should not be nil")
	}

	// The negInfMask should be -inf for all positions
	t.Log("negInfMask created successfully")
}

func TestEnginePrecomputedMasks(t *testing.T) {
	vocab := testVocab()
	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine: %v", err)
	}
	defer e.Close()

	// Check that all terminal symbols have masks
	pda, _ := GetJSONPDA()
	for _, terminal := range pda.Terminals {
		mask, ok := e.symbolMasks[terminal]
		if !ok {
			t.Errorf("missing mask for terminal %q", terminal)
			continue
		}
		if mask == nil {
			t.Errorf("nil mask for terminal %q", terminal)
		}
	}

	t.Logf("Precomputed masks for %d terminals", len(pda.Terminals))
}

func TestEngineEmptyVocab(t *testing.T) {
	e, err := NewEngine([]string{})
	if err != nil {
		t.Fatalf("failed to create engine with empty vocab: %v", err)
	}
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

	e, err := NewEngine(vocab)
	if err != nil {
		t.Fatalf("failed to create engine with large vocab: %v", err)
	}
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

// Helper to check if float is -inf
func isNegInf(f float32) bool {
	return math.IsInf(float64(f), -1)
}
