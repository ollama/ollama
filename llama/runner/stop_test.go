package runner

import (
	"testing"
)

func TestTruncateStop(t *testing.T) {
	tests := []struct {
		name          string
		sequence      string
		stop          string
		expected      string
		expectedTrunc bool
	}{
		{
			name:          "Single word",
			sequence:      "helloworld",
			stop:          "world",
			expected:      "hello",
			expectedTrunc: true,
		},
		{
			name:          "Partial",
			sequence:      "hellowor",
			stop:          "or",
			expected:      "hellow",
			expectedTrunc: true,
		},
		{
			name:          "Suffix",
			sequence:      "Hello there!",
			stop:          "!",
			expected:      "Hello there",
			expectedTrunc: true,
		},
		{
			name:          "Middle",
			sequence:      "hello wor",
			stop:          "llo w",
			expected:      "he",
			expectedTrunc: true,
		},
		{
			name:          "No stop found",
			sequence:      "hello world",
			stop:          "xyz",
			expected:      "hello world",
			expectedTrunc: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, truncated := truncateStop(tt.sequence, tt.stop)
			if result != tt.expected || truncated != tt.expectedTrunc {
				t.Errorf("truncateStop(%q, %q): have %q (%v); want %q (%v)",
					tt.sequence, tt.stop, result, truncated, tt.expected, tt.expectedTrunc)
			}
		})
	}
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
			result := incompleteUnicode(tt.input)
			if result != tt.expected {
				t.Errorf("incompleteUnicode(%s): have %v; want %v", tt.input, result, tt.expected)
			}
		})
	}
}
