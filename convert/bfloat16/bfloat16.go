package bfloat16

import "math"

// FromFloat32s converts a slice of float32 values to a slice of bfloat16 values, represented as uint16s.
func FromFloat32s(f32s []float32) (u16s []uint16) {
	u16s = make([]uint16, len(f32s))
	for i := range f32s {
		u16s[i] = uint16(math.Float32bits(f32s[i]) >> 16)
	}
	return u16s
}

// Float32s converts a slice of bfloat16 values, represented as uint16s, back to a slice of float32 values.
func Float32s(u16s []uint16) (f32s []float32) {
	f32s = make([]float32, len(u16s))
	for i := range u16s {
		f32s[i] = math.Float32frombits(uint32(u16s[i]) << 16)
	}
	return f32s
}
