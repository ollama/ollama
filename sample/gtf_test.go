package sample

import (
	"testing"
)

func TestGrammarParsing(t *testing.T) {
	tests := []struct {
		name      string
		grammar   map[string]string
		startRule string
		input     string
		want      bool
	}{
		{
			name: "simple object",
			grammar: map[string]string{
				"object": `"{" "}"`,
			},
			startRule: "object",
			input:     "{}",
			want:      true,
		},
		{
			name: "simple array",
			grammar: map[string]string{
				"array": `"[" "]"`,
			},
			startRule: "array",
			input:     "[]",
			want:      true,
		},
		{
			name: "character class",
			grammar: map[string]string{
				"digit": `[0-9]`,
			},
			startRule: "digit",
			input:     "5",
			want:      true,
		},
		{
			name: "alternation",
			grammar: map[string]string{
				"bool": `"true" | "false"`,
			},
			startRule: "bool",
			input:     "true",
			want:      true,
		},
		{
			name: "repetition",
			grammar: map[string]string{
				"digits": `[0-9]+`,
			},
			startRule: "digits",
			input:     "123",
			want:      true,
		},
		{
			name: "nested rules",
			grammar: map[string]string{
				"value":  `object | array`,
				"object": `"{" "}"`,
				"array":  `"[" "]"`,
			},
			startRule: "value",
			input:     "{}",
			want:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := NewParser(tt.grammar)
			machine, err := parser.Parse(tt.startRule)
			if err != nil {
				t.Fatalf("Parse() error = %v", err)
			}

			matcher := NewMatcher(machine)
			got, err := matcher.Match(tt.input)
			if err != nil {
				t.Fatalf("Match() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("Match() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestJSONGrammar(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"empty object", "{}", true},
		{"empty array", "[]", true},
		{"simple string", `"hello"`, true},
		{"simple number", "123", true},
		{"simple boolean", "true", true},
		{"simple null", "null", true},
		{"object with string", `{"key": "value"}`, true},
		{"array with numbers", "[1, 2, 3]", true},
		{"nested object", `{"obj": {"key": "value"}}`, true},
		{"nested array", `[1, [2, 3], 4]`, true},
		{"invalid object", "{", false},
		{"invalid array", "[1, 2", false},
		{"invalid string", `"hello`, false},
	}

	parser := NewParser(DefaultGrammar)
	machine, err := parser.Parse("value")
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}

	matcher := NewMatcher(machine)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := matcher.Match(tt.input)
			if tt.want {
				if err != nil {
					t.Errorf("Match() error = %v", err)
				}
				if !got {
					t.Errorf("Match() = false, want true")
				}
			} else {
				if err == nil && got {
					t.Errorf("Match() = true, want false")
				}
			}
		})
	}
}
