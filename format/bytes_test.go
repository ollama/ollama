package format

import (
	"testing"
)

func TestHumanBytes(t *testing.T) {
	type testCase struct {
		input    int64
		expected string
	}

	tests := []testCase{
		// Test bytes (B)
		{0, "0 B"},
		{1, "1 B"},
		{1023, "1023 B"},

		// Test kibibytes (KiB)
		{1024, "1 KiB"},
		{1536, "1.5 KiB"},
		{1048575, "1023.9 KiB"},

		// Test mebibytes (MiB)
		{1048576, "1 MiB"},
		{1572864, "1.5 MiB"},
		{1073741823, "1024.0 MiB"},

		// Test gibibytes (GiB)
		{1073741824, "1 GiB"},
		{1610612736, "1.5 GiB"},
		{2147483647, "2.0 GiB"},

		// Test tebibytes (TiB)
		{1099511627776, "1 TiB"},
		{1649267441664, "1.5 TiB"},
		{2199023255551, "2.0 TiB"},

		// Test fractional values
		{1234, "1.2 KiB"},
		{1261586, "1.2 MiB"},
		{1291271369, "1.2 GiB"},
	}

	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			result := HumanBytes(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}

func TestHumanBytes2(t *testing.T) {
	type testCase struct {
		input    uint64
		expected string
	}

	tests := []testCase{
		// Test bytes (B)
		{0, "0 B"},
		{1, "1 B"},
		{1023, "1023 B"},

		// Test kibibytes (KiB)
		{1024, "1.0 KiB"},
		{1536, "1.5 KiB"},
		{1048575, "1024.0 KiB"},

		// Test mebibytes (MiB)
		{1048576, "1.0 MiB"},
		{1572864, "1.5 MiB"},
		{1073741823, "1024.0 MiB"},

		// Test gibibytes (GiB)
		{1073741824, "1.0 GiB"},
		{1610612736, "1.5 GiB"},
		{2147483648, "2.0 GiB"},

		// Test tebibytes (TiB)
		{1099511627776, "1.0 TiB"},
		{1649267441664, "1.5 TiB"},
		{2199023255552, "2.0 TiB"},
	}

	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			result := HumanBytes2(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}
