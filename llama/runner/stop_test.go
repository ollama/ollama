package main

import (
	"reflect"
	"testing"
)

func TestTruncateStop(t *testing.T) {
	tests := []struct {
		name     string
		pieces   []string
		stop     string
		expected []string
	}{
		{
			name:     "Single word",
			pieces:   []string{"hello", "world"},
			stop:     "world",
			expected: []string{"hello"},
		},
		{
			name:     "Partial",
			pieces:   []string{"hello", "wor"},
			stop:     "or",
			expected: []string{"hello", "w"},
		},
		{
			name:     "Suffix",
			pieces:   []string{"Hello", " there", "!"},
			stop:     "!",
			expected: []string{"Hello", " there"},
		},
		{
			name:     "Middle",
			pieces:   []string{"hello", " wor"},
			stop:     "llo w",
			expected: []string{"he"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := truncateStop(tt.pieces, tt.stop)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("truncateStop(%v, %s): have %v; want %v", tt.pieces, tt.stop, result, tt.expected)
			}
		})
	}
}
