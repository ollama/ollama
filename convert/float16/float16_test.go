package float16

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestFloat16(t *testing.T) {
	cases := []struct {
		name  string
		input uint16
		want  uint32
	}{
		// Zero cases
		{"positive zero", 0x0000, 0x0},
		{"negative zero", 0x8000, 0x80000000},

		// Normal numbers
		{"one", 0x3C00, 0x3F800000},
		{"negative one", 0xBC00, 0xBF800000},
		{"two", 0x4000, 0x40000000},
		{"half", 0x3800, 0x3F000000},
		{"max normal", 0x7BFF, 0x477fe000},
		{"min positive normal", 0x0400, 0x38800000},

		// Infinity cases
		{"positive infinity", 0x7C00, 0x7F800000},
		{"negative infinity", 0xFC00, 0xFF800000},

		// NaN cases
		{"NaN", 0x7C01, 0x7f802000},
		{"NaN with payload", 0x7E00, 0x7FC00000},

		// Subnormal cases
		{"min positive subnormal", 0x0001, 0x33800000},
		{"max subnormal", 0x03FF, 0x387fc000},

		// Common values
		{"pi approximation", 0x4248, 0x40490000},
		{"e approximation", 0x416F, 0x402de000},
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

func BenchmarkFloat16(b *testing.B) {
	f32s := make([]float32, 1_000_000)
	for i := range f32s {
		f32s[i] = rand.Float32()
	}
	for b.Loop() {
		Float32s(FromFloat32s(f32s))
	}
}
