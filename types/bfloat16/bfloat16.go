// Vendored code from https://github.com/d4l3k/go-bfloat16
// unsafe pointer replaced by "math"
package bfloat16

import "math"

type BF16 uint16

func FromBytes(buf []byte) BF16 {
	return BF16(uint16(buf[0]) + uint16(buf[1])<<8)
}

func ToBytes(b BF16) []byte {
	return []byte{byte(b & 0xFF), byte(b >> 8)}
}

func Decode(buf []byte) []BF16 {
	var out []BF16
	for i := 0; i < len(buf); i += 2 {
		out = append(out, FromBytes(buf[i:]))
	}
	return out
}

func Encode(f []BF16) []byte {
	var out []byte
	for _, a := range f {
		out = append(out, ToBytes(a)...)
	}
	return out
}

func DecodeFloat32(buf []byte) []float32 {
	var out []float32
	for i := 0; i < len(buf); i += 2 {
		out = append(out, ToFloat32(FromBytes(buf[i:])))
	}
	return out
}

func EncodeFloat32(f []float32) []byte {
	var out []byte
	for _, a := range f {
		out = append(out, ToBytes(FromFloat32(a))...)
	}
	return out
}

func ToFloat32(b BF16) float32 {
	u32 := uint32(b) << 16
	return math.Float32frombits(u32)
}

func FromFloat32(f float32) BF16 {
	u32 := math.Float32bits(f)
	return BF16(u32 >> 16)
}
