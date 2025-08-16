package bfloat16

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestBfloat16(t *testing.T) {
	cases := []struct {
		name  string
		input uint16
		want  uint32
	}{
		// Zero cases
		{"positive zero", 0x0000, 0x0},
		{"negative zero", 0x8000, 0x80000000},

		// Normal numbers
		{"one", 0x3F80, 0x3F800000},
		{"negative one", 0xBF80, 0xBF800000},
		{"two", 0x4000, 0x40000000},
		{"half", 0x3F00, 0x3F000000},
		{"quarter", 0x3E80, 0x3E800000},
		{"max finite", 0x7F7F, 0x7F7F0000},
		{"min positive normal", 0x0080, 0x00800000},

		// Infinity cases
		{"positive infinity", 0x7F80, 0x7F800000},
		{"negative infinity", 0xFF80, 0xFF800000},

		// NaN cases
		{"NaN", 0x7FC0, 0x7FC00000},
		{"NaN with payload", 0x7FC1, 0x7FC10000},

		// Subnormal cases
		{"min positive subnormal", 0x0001, 0x00010000},
		{"max subnormal", 0x007F, 0x007F0000},

		// Powers of 2
		{"2^10", 0x4480, 0x44800000},
		{"2^-10", 0x3A80, 0x3A800000},
		{"2^20", 0x4B80, 0x4B800000},

		// Common approximations in BF16
		{"pi approximation", 0x4049, 0x40490000},
		{"e approximation", 0x402E, 0x402E0000},
		{"sqrt(2) approximation", 0x3FB5, 0x3FB50000},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			t.Run("Float32s", func(t *testing.T) {
				got := Float32s([]uint16{tt.input})[0]
				if diff := cmp.Diff(tt.want, math.Float32bits(got)); diff != "" {
					t.Errorf("Float32s mismatch (-want +got):\n%s", diff)
				}
			})

			t.Run("FromFloat32s", func(t *testing.T) {
				got := FromFloat32s([]float32{math.Float32frombits(tt.want)})
				if diff := cmp.Diff([]uint16{tt.input}, got); diff != "" {
					t.Errorf("FromFloat32s mismatch (-want +got):\n%s", diff)
				}
			})
		})
	}
}

func BenchmarkBfloat16(b *testing.B) {
	f32s := make([]float32, 1_000_000)
	for i := range f32s {
		f32s[i] = rand.Float32()
	}
	for b.Loop() {
		Float32s(FromFloat32s(f32s))
	}
}
