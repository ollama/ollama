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
