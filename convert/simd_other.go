//go:build !amd64

package convert

import (
	"math"

	"github.com/x448/float16"
)

func convertF16ToF32(dst []float32, src []uint16) {
	for i, v := range src {
		dst[i] = float16.Frombits(v).Float32()
	}
}

func convertF32ToF16(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = float16.Fromfloat32(v).Bits()
	}
}

func convertBF16ToF32(dst []float32, src []uint16) {
	for i, v := range src {
		dst[i] = math.Float32frombits(uint32(v) << 16)
	}
}

func convertF32ToBF16(dst []uint16, src []float32) {
	for i, v := range src {
		dst[i] = uint16(math.Float32bits(v) >> 16)
	}
}
