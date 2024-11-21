package main

import (
	"reflect"
	"testing"
)

func TestTruncateStop(t *testing.T) {
	tests := []struct {
		name          string
		pieces        []string
		stop          string
		expected      []string
		expectedTrunc bool
	}{
		{
			name:          "Single word",
			pieces:        []string{"hello", "world"},
			stop:          "world",
			expected:      []string{"hello"},
			expectedTrunc: false,
		},
		{
			name:          "Partial",
			pieces:        []string{"hello", "wor"},
			stop:          "or",
			expected:      []string{"hello", "w"},
			expectedTrunc: true,
		},
		{
			name:          "Suffix",
			pieces:        []string{"Hello", " there", "!"},
			stop:          "!",
			expected:      []string{"Hello", " there"},
			expectedTrunc: false,
		},
		{
			name:          "Suffix partial",
			pieces:        []string{"Hello", " the", "re!"},
			stop:          "there!",
			expected:      []string{"Hello", " "},
			expectedTrunc: true,
		},
		{
			name:          "Middle",
			pieces:        []string{"hello", " wor"},
			stop:          "llo w",
			expected:      []string{"he"},
			expectedTrunc: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, resultTrunc := truncateStop(tt.pieces, tt.stop)
			if !reflect.DeepEqual(result, tt.expected) || resultTrunc != tt.expectedTrunc {
				t.Errorf("truncateStop(%v, %s): have %v (%v); want %v (%v)", tt.pieces, tt.stop, result, resultTrunc, tt.expected, tt.expectedTrunc)
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
