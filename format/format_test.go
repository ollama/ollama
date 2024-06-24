package format

import (
	"testing"
)

func TestHumanNumber(t *testing.T) {
	type testCase struct {
		input    uint64
		expected string
	}

	testCases := []testCase{
		{26000000, "26.0M"},
		{26000000000, "26.0B"},
		{1000, "1.00K"},
		{1000000, "1.00M"},
		{1000000000, "1.00B"},
		{1000000000000, "1.00T"},
		{100, "100"},
		{206000000, "206M"},
	}

	for _, tc := range testCases {
		t.Run(tc.expected, func(t *testing.T) {
			result := HumanNumber(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}
