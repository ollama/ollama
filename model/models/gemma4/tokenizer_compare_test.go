package gemma4

import (
	"os"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/tokenizer"
)

// TestTokenizerMatchesHF compares our tokenizer output against HuggingFace reference tokens.
func TestTokenizerMatchesHF(t *testing.T) {
	modelPath := os.Getenv("GEMMA4_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set GEMMA4_MODEL_PATH to a gemma4 GGUF file")
	}

	m, err := model.New(modelPath, ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}
	defer m.Backend().Close()

	tok := m.(tokenizer.Tokenizer)

	tests := []struct {
		name     string
		input    string
		expected []int32
	}{
		{
			name:     "simple",
			input:    "Hello, world!",
			expected: []int32{9259, 236764, 1902, 236888},
		},
		{
			name:     "special_tokens",
			input:    "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
			expected: []int32{105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107},
		},
		{
			name:     "tool_declaration",
			input:    "<|tool>declaration:bash{description:<|\"|>Run a command<|\"|>}<tool|>",
			expected: []int32{46, 163688, 236787, 42422, 236782, 7777, 236787, 52, 7306, 496, 4991, 52, 236783, 47},
		},
		{
			name:     "tool_call",
			input:    "<|tool_call>call:bash{command:<|\"|>ls -la<|\"|>}<tool_call|>",
			expected: []int32{48, 6639, 236787, 42422, 236782, 7674, 236787, 52, 5629, 753, 2149, 52, 236783, 49},
		},
		{
			name:     "thinking",
			input:    "<|channel>thought\nLet me think about this...<channel|>The answer is 42.",
			expected: []int32{100, 45518, 107, 6481, 786, 1751, 1003, 672, 1390, 101, 818, 3890, 563, 236743, 236812, 236778, 236761},
		},
		{
			name:     "code",
			input:    "func main() { fmt.Println(\"hello\") }",
			expected: []int32{6823, 1689, 825, 642, 22766, 236761, 29006, 885, 23391, 1373, 682},
		},
		{
			name:     "numbers",
			input:    "The answer is 42, not 43.5 or -1",
			expected: []int32{818, 3890, 563, 236743, 236812, 236778, 236764, 711, 236743, 236812, 236800, 236761, 236810, 653, 753, 236770},
		},
		{
			name:     "mixed_chat_with_tools",
			input:    "<|turn>system\nYou are a helpful assistant.\n<|tool>declaration:get_weather{description:<|\"|>Get weather<|\"|>,parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n<|turn>user\nWhat's the weather in Paris?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
			expected: []int32{105, 9731, 107, 3048, 659, 496, 11045, 16326, 236761, 107, 46, 163688, 236787, 828, 236779, 19323, 236782, 7777, 236787, 52, 3407, 7606, 52, 236764, 19031, 29616, 15921, 29616, 13319, 29616, 2084, 236787, 52, 35410, 52, 5237, 2084, 236787, 52, 60688, 52, 1807, 47, 106, 107, 105, 2364, 107, 3689, 236789, 236751, 506, 7606, 528, 9079, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokens, err := tok.Encode(tt.input, false) // no BOS
			if err != nil {
				t.Fatalf("encode error: %v", err)
			}

			if len(tokens) != len(tt.expected) {
				t.Errorf("token count mismatch: got %d, want %d", len(tokens), len(tt.expected))
				t.Logf("got:  %v", tokens)
				t.Logf("want: %v", tt.expected)
				return
			}

			mismatches := 0
			for i := range tokens {
				if tokens[i] != tt.expected[i] {
					mismatches++
					if mismatches <= 5 {
						t.Errorf("mismatch at [%d]: got %d, want %d", i, tokens[i], tt.expected[i])
					}
				}
			}
			if mismatches > 5 {
				t.Errorf("... and %d more mismatches", mismatches-5)
			}
		})
	}
}
