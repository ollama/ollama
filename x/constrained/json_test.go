//go:build mlx

package constrained

import (
	"testing"
)

func TestJSONGrammarCompiles(t *testing.T) {
	pda, err := GetJSONPDA()
	if err != nil {
		t.Fatalf("failed to compile JSON grammar: %v", err)
	}

	if pda == nil {
		t.Fatal("PDA is nil")
	}

	t.Logf("JSON PDA: %d states, %d terminals", pda.States, len(pda.Terminals))
	t.Logf("Terminals: %v", pda.Terminals)
}

func TestJSONRuntimeSimpleObject(t *testing.T) {
	rt, err := NewJSONRuntime()
	if err != nil {
		t.Fatalf("failed to create runtime: %v", err)
	}

	// {"key": "value"}
	tokens := []string{"{", "STRING", ":", "STRING", "}"}
	for i, tok := range tokens {
		if !rt.Accept(tok) {
			t.Fatalf("failed to accept token %d (%s)", i, tok)
		}
	}

	if !rt.IsAccepting() {
		t.Error("should be accepting after valid JSON object")
	}
}

func TestJSONRuntimeNestedObject(t *testing.T) {
	rt, err := NewJSONRuntime()
	if err != nil {
		t.Fatalf("failed to create runtime: %v", err)
	}

	// {"a": {"b": "c"}}
	tokens := []string{"{", "STRING", ":", "{", "STRING", ":", "STRING", "}", "}"}
	for i, tok := range tokens {
		if !rt.Accept(tok) {
			t.Fatalf("failed to accept token %d (%s)", i, tok)
		}
	}

	if !rt.IsAccepting() {
		t.Error("should be accepting after nested object")
	}
}

func TestJSONRuntimeArray(t *testing.T) {
	rt, err := NewJSONRuntime()
	if err != nil {
		t.Fatalf("failed to create runtime: %v", err)
	}

	// [1, 2, 3]
	tokens := []string{"[", "NUMBER", ",", "NUMBER", ",", "NUMBER", "]"}
	for i, tok := range tokens {
		if !rt.Accept(tok) {
			t.Fatalf("failed to accept token %d (%s)", i, tok)
		}
	}

	if !rt.IsAccepting() {
		t.Error("should be accepting after array")
	}
}

func TestJSONRuntimeMixedTypes(t *testing.T) {
	rt, err := NewJSONRuntime()
	if err != nil {
		t.Fatalf("failed to create runtime: %v", err)
	}

	// {"str": "val", "num": 42, "bool": true, "nil": null, "arr": [1, 2]}
	tokens := []string{
		"{",
		"STRING", ":", "STRING", ",",
		"STRING", ":", "NUMBER", ",",
		"STRING", ":", "true", ",",
		"STRING", ":", "null", ",",
		"STRING", ":", "[", "NUMBER", ",", "NUMBER", "]",
		"}",
	}
	for i, tok := range tokens {
		if !rt.Accept(tok) {
			t.Fatalf("failed to accept token %d (%s)", i, tok)
		}
	}

	if !rt.IsAccepting() {
		t.Error("should be accepting after mixed types")
	}
}

func TestJSONRuntimeEmptyStructures(t *testing.T) {
	// Empty object {}
	t.Run("empty object", func(t *testing.T) {
		rt, _ := NewJSONRuntime()
		if !rt.Accept("{") {
			t.Fatal("failed to accept {")
		}
		if !rt.Accept("}") {
			t.Fatal("failed to accept }")
		}
		if !rt.IsAccepting() {
			t.Error("should be accepting")
		}
	})

	// Empty array []
	t.Run("empty array", func(t *testing.T) {
		rt, _ := NewJSONRuntime()
		if !rt.Accept("[") {
			t.Fatal("failed to accept [")
		}
		if !rt.Accept("]") {
			t.Fatal("failed to accept ]")
		}
		if !rt.IsAccepting() {
			t.Error("should be accepting")
		}
	})
}

func TestJSONRuntimePrimitives(t *testing.T) {
	testCases := []struct {
		name   string
		tokens []string
	}{
		{"string", []string{"STRING"}},
		{"number", []string{"NUMBER"}},
		{"true", []string{"true"}},
		{"false", []string{"false"}},
		{"null", []string{"null"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rt, _ := NewJSONRuntime()
			for i, tok := range tc.tokens {
				if !rt.Accept(tok) {
					t.Fatalf("failed to accept token %d (%s)", i, tok)
				}
			}
			if !rt.IsAccepting() {
				t.Error("should be accepting")
			}
		})
	}
}

func TestJSONRuntimeInvalid(t *testing.T) {
	testCases := []struct {
		name   string
		tokens []string
	}{
		{"missing closing brace", []string{"{", "STRING", ":", "STRING"}},
		{"double comma", []string{"[", "NUMBER", ",", ",", "NUMBER", "]"}},
		{"missing colon", []string{"{", "STRING", "STRING", "}"}},
		{"trailing comma object", []string{"{", "STRING", ":", "STRING", ",", "}"}},
		{"trailing comma array", []string{"[", "NUMBER", ",", "]"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rt, _ := NewJSONRuntime()
			accepted := true
			for _, tok := range tc.tokens {
				if !rt.Accept(tok) {
					accepted = false
					break
				}
			}
			// Either should fail to accept some token, or not be in accepting state
			if accepted && rt.IsAccepting() {
				t.Errorf("should not accept invalid JSON: %v", tc.tokens)
			}
		})
	}
}

func TestValidTokenTypes(t *testing.T) {
	rt, err := NewJSONRuntime()
	if err != nil {
		t.Fatalf("failed to create runtime: %v", err)
	}

	// At start, should be able to accept any value type
	valid := rt.ValidTokenTypes()
	t.Logf("Valid at start: %v", valid)

	// Should include object start, array start, primitives
	hasObjectStart := false
	hasArrayStart := false
	hasString := false
	for _, v := range valid {
		switch v {
		case TokenObjectStart:
			hasObjectStart = true
		case TokenArrayStart:
			hasArrayStart = true
		case TokenString:
			hasString = true
		}
	}

	if !hasObjectStart {
		t.Error("should have { as valid")
	}
	if !hasArrayStart {
		t.Error("should have [ as valid")
	}
	if !hasString {
		t.Error("should have STRING as valid")
	}

	// After {, should be able to accept STRING (for key) or }
	rt.Accept("{")
	valid = rt.ValidTokenTypes()
	t.Logf("Valid after '{': %v", valid)

	hasString = false
	hasObjectEnd := false
	for _, v := range valid {
		switch v {
		case TokenString:
			hasString = true
		case TokenObjectEnd:
			hasObjectEnd = true
		}
	}

	if !hasString {
		t.Error("should have STRING as valid after {")
	}
	if !hasObjectEnd {
		t.Error("should have } as valid after { (empty object)")
	}
}

func TestClassifyToken(t *testing.T) {
	testCases := []struct {
		input    string
		expected TokenType
	}{
		{"{", TokenObjectStart},
		{"}", TokenObjectEnd},
		{"[", TokenArrayStart},
		{"]", TokenArrayEnd},
		{":", TokenColon},
		{",", TokenComma},
		{"true", TokenTrue},
		{"false", TokenFalse},
		{"null", TokenNull},
		{`"hello"`, TokenString},
		{`"`, TokenString}, // Partial string
		{"123", TokenNumber},
		{"-45.67", TokenNumber},
		{" ", TokenWhitespace},
		{"\n", TokenWhitespace},
		{"xyz", TokenUnknown},
	}

	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			got := ClassifyToken(tc.input)
			if got != tc.expected {
				t.Errorf("ClassifyToken(%q) = %v, want %v", tc.input, got, tc.expected)
			}
		})
	}
}
