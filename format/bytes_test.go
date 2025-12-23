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

		// Test kilobytes (KB) - using binary (1024)
		{1024, "1 KB"},
		{1536, "1.5 KB"},
		{3072, "3 KB"},
		{4096, "4 KB"},
		{1048575, "1023 KB"},

		// Test megabytes (MB) - using binary (1024*1024)
		{1048576, "1 MB"},
		{1572864, "1.5 MB"},
		{1073741823, "1023 MB"},

		// Test gigabytes (GB) - using binary (1024*1024*1024)
		{1073741824, "1 GB"},
		{1610612736, "1.5 GB"},
		{1099511627775, "1023 GB"},

		// Test terabytes (TB) - using binary (1024^4)
		{1099511627776, "1 TB"},
		{1649267441664, "1.5 TB"},
		{2199023255551, "2.0 TB"},

		// Test fractional values
		{1280, "1.2 KB"},
		{1310720, "1.2 MB"},
		{1288490189, "1.2 GB"},
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
