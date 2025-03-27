package sample

import (
	"testing"

	"github.com/ollama/ollama/model"
)

func TestBuildGraph(t *testing.T) {
	tests := []struct {
		name    string
		grammar []byte
		wantErr bool
	}{
		{
			name:    "empty grammar",
			grammar: []byte{},
			wantErr: false,
		},
		{
			name: "valid grammar",
			grammar: []byte(`root ::= value
value ::= string | number`),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				proc:    &mockProcessor{},
				grammar: tt.grammar,
				rules:   make(map[string]string),
			}

			node := &Node{
				TransitionEdges: make(map[rune]*Node),
			}

			err := g.BuildGraph(node)
			if (err != nil) != tt.wantErr {
				t.Errorf("BuildGraph() error = %v, wantErr %v", err, tt.wantErr)
			}

			if !tt.wantErr {
				if len(g.decodedToks) == 0 {
					t.Error("Expected decoded tokens, got none")
				}
				if len(g.rules) == 0 {
					t.Error("Expected rules to be populated")
				}
			}
		})
	}
}

func TestRootPrefixes(t *testing.T) {
	tests := []struct {
		name     string
		grammar  []byte
		expected map[string]string
	}{
		{
			name:     "empty grammar",
			grammar:  []byte{},
			expected: map[string]string{},
		},
		{
			name: "grammar with root prefix",
			grammar: []byte(`root ::= value
root_string ::= string`),
			expected: map[string]string{
				"root":        "value",
				"root_string": "string",
			},
		},
		{
			name: "grammar with comments and empty lines",
			grammar: []byte(`# comment
root ::= value

# another comment
root_number ::= number`),
			expected: map[string]string{
				"root":        "value",
				"root_number": "number",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				grammar: tt.grammar,
				rules:   make(map[string]string),
			}

			g.rootPrefixes()

			for k, v := range tt.expected {
				if actual, ok := g.rules[k]; !ok || actual != v {
					t.Errorf("Expected rule %s = %s, got %s", k, v, actual)
				}
			}
		})
	}
}

func TestParseRule(t *testing.T) {
	tests := []struct {
		name     string
		rule     string
		expected string
	}{
		{
			name:     "empty rule",
			rule:     "",
			expected: "",
		},
		{
			name:     "simple string",
			rule:     "root ::= \"test_string\"",
			expected: "test_string",
		},
		{
			name:     "simple string",
			rule:     "root ::= \"test_string\" | \"test_string2\"",
			expected: "test_stringtest_string2",
		},
		{
			name: "integer",
			rule: "root ::= [0-9]+",
			// TODO: this is infinite acutally
			expected: "0123456789",
		},
		// TODO: handle left recursion
		// {
		// 	name:     "left recursion",
		// 	rule:     "root ::= root \"test_string\"",
		// 	expected: "test_string",
		// },
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Graph{
				rules: make(map[string]string),
			}

			rootNode := &Node{
				TransitionEdges: make(map[rune]*Node),
			}
			curNode := rootNode
			g.parseRule(tt.rule, curNode)
			sb := ""
			for {
				if len(curNode.TransitionEdges) == 0 {
					break
				}

				for r, n := range curNode.TransitionEdges {
					sb += string(r)
					curNode = n
				}
				t.Logf("sb: %s", sb)
			}

			if sb != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, sb)
			}
		})
	}
}

// mockProcessor implements the TextProcessor interface for testing
type mockProcessor struct{}

func (m *mockProcessor) Decode(tokens []int32) (string, error) {
	return "test", nil
}

func (m *mockProcessor) Vocab() *model.Vocabulary {
	return &model.Vocabulary{
		Values: []string{"test1", "test2"},
	}
}

func (m *mockProcessor) Encode(s string, addSpecial bool) ([]int32, error) {
	return []int32{0, 1}, nil
}

func (m *mockProcessor) Is(token int32, special model.Special) bool {
	return false
}
