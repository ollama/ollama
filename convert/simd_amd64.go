//go:build amd64

package convert

import (
	"math"

	"github.com/x448/float16"
	"golang.org/x/sys/cpu"
)

var useAVX2 = cpu.X86.HasAVX2

//go:noescape
func f16ToF32AVX(dst *float32, src *uint16, chunks int)

//go:noescape
func f32ToF16AVX(dst *uint16, src *float32, chunks int)

//go:noescape
func bf16ToF32AVX(dst *float32, src *uint16, chunks int)

//go:noescape
func f32ToBF16AVX(dst *uint16, src *float32, chunks int)

func convertF16ToF32(dst []float32, src []uint16) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		f16ToF32AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = float16.Frombits(src[i]).Float32()
	}
}

func convertF32ToF16(dst []uint16, src []float32) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		f32ToF16AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = float16.Fromfloat32(src[i]).Bits()
	}
}

func convertBF16ToF32(dst []float32, src []uint16) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		bf16ToF32AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = math.Float32frombits(uint32(src[i]) << 16)
	}
}

func convertF32ToBF16(dst []uint16, src []float32) {
	n := len(src)
	if n == 0 {
		return
	}
	bulk := 0
	if useAVX2 && n >= 8 {
		bulk = (n / 8) * 8
		f32ToBF16AVX(&dst[0], &src[0], n/8)
	}
	for i := bulk; i < n; i++ {
		dst[i] = uint16(math.Float32bits(src[i]) >> 16)
	}
}
