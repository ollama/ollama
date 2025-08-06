package float16

import (
	"math"
)

func FromFloat32s(f32s []float32) (u16s []uint16) {
	u16s = make([]uint16, len(f32s))
	for i := range f32s {
		bits := math.Float32bits(f32s[i])
		sign := (bits >> 31) & 0x1
		exponent := (bits >> 23) & 0xFF
		mantissa := bits & 0x7FFFFF
		if exponent == 0xFF {
			if mantissa == 0 {
				// Infinity
				u16s[i] = uint16((sign << 15) | 0x7C00)
			} else {
				// NaN
				u16s[i] = uint16((sign << 15) | 0x7C00 | (mantissa >> 13))
			}
		} else if exponent == 0 && mantissa == 0 {
			// Zero
			u16s[i] = uint16(sign << 15)
		} else {
			// Convert exponent from FP32 bias (127) to FP16 bias (15)
			exponent := int(exponent) - 127 + 15
			if exponent >= 31 {
				// Overflow to infinity
				u16s[i] = uint16((sign << 15) | 0x7C00)
			} else if exponent <= 0 {
				// Underflow - create subnormal or zero
				if exponent < -10 {
					u16s[i] = uint16(sign << 15) // Zero
				} else {
					// Subnormal number
					mantissa = (mantissa | 0x800000) >> uint(-exponent+1)
					u16s[i] = uint16((sign << 15) | (mantissa >> 13))
				}
			} else {
				// Normal number - truncate mantissa from 23 to 10 bits
				u16s[i] = uint16((sign << 15) | (uint32(exponent) << 10) | (mantissa >> 13))
			}
		}
	}

	return u16s
}

func Float32s(u16s []uint16) (f32s []float32) {
	f32s = make([]float32, len(u16s))
	for i := range u16s {
		sign := (u16s[i] >> 15) & 0x1
		exponent := (u16s[i] >> 10) & 0x1F
		mantissa := u16s[i] & 0x3FF

		var u32 uint32
		switch exponent {
		case 0:
			if mantissa == 0 {
				// Zero
				u32 = uint32(sign) << 31
			} else {
				// Subnormal - convert to normal
				// Find leading 1 bit
				shift := 0
				temp := mantissa
				for temp&0x400 == 0 {
					temp <<= 1
					shift++
				}

				exponent := 127 - 15 + 1 - shift
				mantissa := (uint32(temp&0x3FF) << 13)

				u32 = (uint32(sign) << 31) | (uint32(exponent) << 23) | mantissa
			}
		case 0x1F:
			if mantissa == 0 {
				// Infinity
				u32 = (uint32(sign) << 31) | 0x7F800000
			} else {
				// NaN
				u32 = (uint32(sign) << 31) | 0x7F800000 | (uint32(mantissa) << 13)
			}
		default:
			// Normal number
			exponent := uint32(exponent) - 15 + 127
			mantissa := uint32(mantissa) << 13

			u32 = (uint32(sign) << 31) | (exponent << 23) | mantissa
		}

		f32s[i] = math.Float32frombits(u32)
	}
	return f32s
}
