package format

import (
	"testing"
)

func TestRoundedParameter(t *testing.T) {
	type testCase struct {
		input    uint64
		expected string
	}

	testCases := []testCase{
		{0, "0"},
		{1000000, "1M"},
		{125000000, "125M"},
		{500500000, "500.50M"},
		{500550000, "500.55M"},
		{1000000000, "1B"},
		{2800000000, "2.8B"},
		{2850000000, "2.9B"},
		{1000000000000, "1000B"},
	}

	for _, tc := range testCases {
		t.Run(tc.expected, func(t *testing.T) {
			result := RoundedParameter(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}

func TestParameters(t *testing.T) {
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
			result := Parameters(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}
