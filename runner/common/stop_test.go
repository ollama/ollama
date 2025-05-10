package common

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestTruncateStop(t *testing.T) {
	tests := []struct {
		name          string
		pieces        []llm.CompletionResponse
		stop          string
		expected      []llm.CompletionResponse
		expectedTrunc bool
	}{
		{
			name: "Single word",
			pieces: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: "world"},
			},
			stop: "world",
			expected: []llm.CompletionResponse{
				{Content: "Hello"},
			},
			expectedTrunc: false,
		},
		{
			name: "Partial",
			pieces: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: " wor"},
			},
			stop: "or",
			expected: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: " w"},
			},
			expectedTrunc: true,
		},
		{
			name: "Suffix",
			pieces: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: " there"},
				{Content: "!"},
			},
			stop: "!",
			expected: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: " there"},
			},
			expectedTrunc: false,
		},
		{
			name: "Suffix partial",
			pieces: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: " the"},
				{Content: "re!"},
			},
			stop: "there!",
			expected: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: " "},
			},
			expectedTrunc: true,
		},
		{
			name: "Middle",
			pieces: []llm.CompletionResponse{
				{Content: "Hello"},
				{Content: " wo"},
			},
			stop: "llo w",
			expected: []llm.CompletionResponse{
				{Content: "He"},
			},
			expectedTrunc: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, resultTrunc := TruncateStop(tt.pieces, tt.stop)
			if !reflect.DeepEqual(result, tt.expected) || resultTrunc != tt.expectedTrunc {
				t.Errorf("truncateStop(%v, %v):\n%shave truncated %v\nwant truncated %v",
					tt.pieces, tt.stop, formatContentDiff(result, tt.expected), resultTrunc, tt.expectedTrunc)
			}
		})
	}
}

func formatContentDiff(result, expected []llm.CompletionResponse) string {
	var s string
	for i := 0; i < len(result) || i < len(expected); i++ {
		if i < len(result) && i < len(expected) && result[i].Content != expected[i].Content {
			s += fmt.Sprintf("[%d] %q vs %q\n", i, result[i].Content, expected[i].Content)
		} else if i < len(result) && i >= len(expected) {
			s += fmt.Sprintf("[%d] extra %q\n", i, result[i].Content)
		} else if i >= len(result) && i < len(expected) {
			s += fmt.Sprintf("[%d] missing %q\n", i, expected[i].Content)
		}
	}
	return s
}

func TestIncompleteUnicode(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name:     "Basic",
			input:    "hi",
			expected: false,
		},
		{
			name:     "Two byte",
			input:    "hi" + string([]byte{0xc2, 0xa3}),
			expected: false,
		},
		{
			name:     "Two byte - missing last",
			input:    "hi" + string([]byte{0xc2}),
			expected: true,
		},
		{
			name:     "Three byte",
			input:    "hi" + string([]byte{0xe0, 0xA0, 0x80}),
			expected: false,
		},
		{
			name:     "Three byte - missing last",
			input:    "hi" + string([]byte{0xe0, 0xA0}),
			expected: true,
		},
		{
			name:     "Three byte - missing last 2",
			input:    "hi" + string([]byte{0xe0}),
			expected: true,
		},
		{
			name:     "Four byte",
			input:    "hi" + string([]byte{0xf0, 0x92, 0x8a, 0xb7}),
			expected: false,
		},
		{
			name:     "Four byte - missing last",
			input:    "hi" + string([]byte{0xf0, 0x92, 0x8a}),
			expected: true,
		},
		{
			name:     "Four byte - missing last 2",
			input:    "hi" + string([]byte{0xf0, 0x92}),
			expected: true,
		},
		{
			name:     "Four byte - missing last 3",
			input:    "hi" + string([]byte{0xf0}),
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IncompleteUnicode(tt.input)
			if result != tt.expected {
				t.Errorf("incompleteUnicode(%s): have %v; want %v", tt.input, result, tt.expected)
			}
		})
	}
}
