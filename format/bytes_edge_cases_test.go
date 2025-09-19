package format

import (
	"math"
	"testing"
)

func TestHumanBytesEdgeCases(t *testing.T) {
	type testCase struct {
		name     string
		input    int64
		expected string
	}

	tests := []testCase{
		// Negative values (function treats them as bytes)
		{
			name:     "negative bytes",
			input:    -1,
			expected: "-1 B",
		},
		{
			name:     "negative value treated as bytes",
			input:    -1500,
			expected: "-1500 B",
		},
		{
			name:     "large negative value treated as bytes",
			input:    -1500000,
			expected: "-1500000 B",
		},

		// Boundary values
		{
			name:     "exactly 1 KB boundary",
			input:    1000,
			expected: "1 KB",
		},
		{
			name:     "just below 1 KB boundary",
			input:    999,
			expected: "999 B",
		},
		{
			name:     "exactly 1 MB boundary",
			input:    1000000,
			expected: "1 MB",
		},
		{
			name:     "just below 1 MB boundary",
			input:    999999,
			expected: "999 KB",
		},
		{
			name:     "exactly 1 GB boundary",
			input:    1000000000,
			expected: "1 GB",
		},
		{
			name:     "just below 1 GB boundary",
			input:    999999999,
			expected: "999 MB",
		},
		{
			name:     "exactly 1 TB boundary",
			input:    1000000000000,
			expected: "1 TB",
		},
		{
			name:     "just below 1 TB boundary",
			input:    999999999999,
			expected: "999 GB",
		},

		// Large values
		{
			name:     "very large TB value",
			input:    9223372036854775807, // math.MaxInt64
			expected: "9223372 TB",
		},
		{
			name:     "large TB with decimal",
			input:    1234567890123456,
			expected: "1234 TB",
		},

		// Precision edge cases
		{
			name:     "KB with exact .0 decimal",
			input:    2000,
			expected: "2 KB",
		},
		{
			name:     "KB with .1 decimal",
			input:    2100,
			expected: "2.1 KB",
		},
		{
			name:     "KB with .9 decimal",
			input:    2900,
			expected: "2.9 KB",
		},
		{
			name:     "MB with exact .0 decimal",
			input:    3000000,
			expected: "3 MB",
		},
		{
			name:     "MB with .1 decimal",
			input:    3100000,
			expected: "3.1 MB",
		},
		{
			name:     "GB with exact .0 decimal",
			input:    4000000000,
			expected: "4 GB",
		},
		{
			name:     "GB with .1 decimal",
			input:    4100000000,
			expected: "4.1 GB",
		},

		// Values that result in >= 10 units (should be integers)
		{
			name:     "10 KB exactly",
			input:    10000,
			expected: "10 KB",
		},
		{
			name:     "10.5 KB (should round to 10)",
			input:    10500,
			expected: "10 KB",
		},
		{
			name:     "15.7 MB (should round to 15)",
			input:    15700000,
			expected: "15 MB",
		},
		{
			name:     "99.9 GB (should round to 99)",
			input:    99900000000,
			expected: "99 GB",
		},

		// Small fractional values
		{
			name:     "1.01 KB",
			input:    1010,
			expected: "1.0 KB",
		},
		{
			name:     "1.05 KB",
			input:    1050,
			expected: "1.1 KB",
		},
		{
			name:     "1.001 MB",
			input:    1001000,
			expected: "1.0 MB",
		},
		{
			name:     "1.009 MB",
			input:    1009000,
			expected: "1.0 MB",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := HumanBytes(tc.input)
			if result != tc.expected {
				t.Errorf("HumanBytes(%d): expected %s, got %s", tc.input, tc.expected, result)
			}
		})
	}
}

func TestHumanBytes2EdgeCases(t *testing.T) {
	type testCase struct {
		name     string
		input    uint64
		expected string
	}

	tests := []testCase{
		// Boundary values for binary units
		{
			name:     "exactly 1 KiB boundary",
			input:    1024,
			expected: "1.0 KiB",
		},
		{
			name:     "just below 1 KiB boundary",
			input:    1023,
			expected: "1023 B",
		},
		{
			name:     "exactly 1 MiB boundary",
			input:    1048576,
			expected: "1.0 MiB",
		},
		{
			name:     "just below 1 MiB boundary",
			input:    1048575,
			expected: "1024.0 KiB",
		},
		{
			name:     "exactly 1 GiB boundary",
			input:    1073741824,
			expected: "1.0 GiB",
		},
		{
			name:     "just below 1 GiB boundary",
			input:    1073741823,
			expected: "1024.0 MiB",
		},

		// Large values
		{
			name:     "very large GiB value",
			input:    math.MaxUint64,
			expected: "17179869184.0 GiB",
		},
		{
			name:     "large GiB with fractional part",
			input:    1234567890123,
			expected: "1149.8 GiB",
		},

		// Precision edge cases
		{
			name:     "KiB with exact .0 decimal",
			input:    2048,
			expected: "2.0 KiB",
		},
		{
			name:     "KiB with .5 decimal",
			input:    1536,
			expected: "1.5 KiB",
		},
		{
			name:     "MiB with exact .0 decimal",
			input:    3145728,
			expected: "3.0 MiB",
		},
		{
			name:     "MiB with .5 decimal",
			input:    1572864,
			expected: "1.5 MiB",
		},
		{
			name:     "GiB with exact .0 decimal",
			input:    4294967296,
			expected: "4.0 GiB",
		},
		{
			name:     "GiB with .5 decimal",
			input:    1610612736,
			expected: "1.5 GiB",
		},

		// Small values
		{
			name:     "1 byte",
			input:    1,
			expected: "1 B",
		},
		{
			name:     "small KiB value",
			input:    1025,
			expected: "1.0 KiB",
		},
		{
			name:     "small MiB value",
			input:    1048577,
			expected: "1.0 MiB",
		},

		// Values that test rounding
		{
			name:     "KiB rounding down",
			input:    1024 + 51, // 1075 bytes = 1.049... KiB
			expected: "1.0 KiB",
		},
		{
			name:     "KiB rounding up",
			input:    1024 + 52, // 1076 bytes = 1.051... KiB
			expected: "1.1 KiB",
		},
		{
			name:     "MiB rounding down",
			input:    1048576 + 52428, // ~1.049 MiB
			expected: "1.0 MiB",
		},
		{
			name:     "MiB rounding up",
			input:    1048576 + 52429, // ~1.051 MiB
			expected: "1.1 MiB",
		},

		// Edge cases with powers of 2
		{
			name:     "2^10 bytes (1 KiB)",
			input:    1 << 10,
			expected: "1.0 KiB",
		},
		{
			name:     "2^20 bytes (1 MiB)",
			input:    1 << 20,
			expected: "1.0 MiB",
		},
		{
			name:     "2^30 bytes (1 GiB)",
			input:    1 << 30,
			expected: "1.0 GiB",
		},
		{
			name:     "2^31 bytes (2 GiB)",
			input:    1 << 31,
			expected: "2.0 GiB",
		},
		{
			name:     "2^32 bytes (4 GiB)",
			input:    1 << 32,
			expected: "4.0 GiB",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := HumanBytes2(tc.input)
			if result != tc.expected {
				t.Errorf("HumanBytes2(%d): expected %s, got %s", tc.input, tc.expected, result)
			}
		})
	}
}

func TestHumanBytesConsistency(t *testing.T) {
	// Test that the same input always produces the same output
	testValues := []int64{0, 1, 999, 1000, 1500, 1000000, 1500000000}
	
	for _, val := range testValues {
		t.Run("consistency test", func(t *testing.T) {
			result1 := HumanBytes(val)
			result2 := HumanBytes(val)
			if result1 != result2 {
				t.Errorf("HumanBytes(%d) inconsistent: got %s and %s", val, result1, result2)
			}
		})
	}
}

func TestHumanBytes2Consistency(t *testing.T) {
	// Test that the same input always produces the same output
	testValues := []uint64{0, 1, 1023, 1024, 1536, 1048576, 1610612736}
	
	for _, val := range testValues {
		t.Run("consistency test", func(t *testing.T) {
			result1 := HumanBytes2(val)
			result2 := HumanBytes2(val)
			if result1 != result2 {
				t.Errorf("HumanBytes2(%d) inconsistent: got %s and %s", val, result1, result2)
			}
		})
	}
}

func BenchmarkHumanBytes(b *testing.B) {
	testValues := []int64{
		0, 1, 999, 1000, 1500, 999999, 1000000, 1500000,
		999999999, 1000000000, 1500000000, 999999999999,
		1000000000000, 1500000000000,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, val := range testValues {
			_ = HumanBytes(val)
		}
	}
}

func BenchmarkHumanBytes2(b *testing.B) {
	testValues := []uint64{
		0, 1, 1023, 1024, 1536, 1048575, 1048576, 1572864,
		1073741823, 1073741824, 1610612736, 4294967296,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, val := range testValues {
			_ = HumanBytes2(val)
		}
	}
}
