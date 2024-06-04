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
		{0, "0"},
		{1000000, "1.00M"},
		{125000000, "125M"},
		{500500000, "500M"},
		{500550000, "501M"},
		{1000000000, "1.00B"},
		{2800000000, "2.80B"},
		{2850000000, "2.85B"},
		{1000000000000, "1.00T"},
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
