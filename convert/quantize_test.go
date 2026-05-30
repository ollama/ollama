package convert

import (
	"math"
	"testing"
)

var benchDst []byte

func makeF32Data(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%2000-1000) * 0.001
	}
	return out
}

func BenchmarkQuantizeQ8_0(b *testing.B) {
	src := makeF32Data(benchElems)
	b.SetBytes(int64(benchElems * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		benchDst = quantizeQ8_0(src)
	}
}

func BenchmarkQuantizeQ4_K(b *testing.B) {
	n := (benchElems / 256) * 256
	src := makeF32Data(n)
	b.SetBytes(int64(n * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		benchDst = quantizeQ4_K(src)
	}
}

func TestQuantizeQ8_0_Correctness(t *testing.T) {
	src := []float32{
		1.0, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.125,
		2.0, -2.0, 1.5, -1.5, 0.75, -0.75, 0.375, -0.375,
		3.0, -3.0, 2.5, -2.5, 1.25, -1.25, 0.625, -0.625,
		4.0, -4.0, 3.5, -3.5, 1.75, -1.75, 0.875, -0.875,
	}

	out := quantizeQ8_0(src)
	if len(out) != 34 {
		t.Fatalf("expected 34 bytes, got %d", len(out))
	}

	dBits := uint16(out[0]) | uint16(out[1])<<8
	d := float16Decode(dBits)
	if d <= 0 {
		t.Fatalf("scale should be positive, got %f", d)
	}

	// max abs is 4.0, so d ≈ 4.0/127 ≈ 0.0315
	expectedD := float32(4.0) / 127.0
	if math.Abs(float64(d-expectedD)) > 0.01 {
		t.Errorf("scale d=%f, expected ~%f", d, expectedD)
	}

	// first quant should be round(1.0 / d) = round(1.0 * 127 / 4.0) = round(31.75) = 32
	q0 := int8(out[2])
	expected := int8(math.RoundToEven(float64(src[0]) / float64(d)))
	if q0 != expected {
		t.Errorf("qs[0]=%d, expected %d", q0, expected)
	}
}

func float16Decode(bits uint16) float32 {
	sign := uint32(bits>>15) << 31
	exp := int((bits >> 10) & 0x1F)
	mant := uint32(bits & 0x3FF)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= ^uint32(0x400)
		fallthrough
	default:
		return math.Float32frombits(sign | uint32(exp+112)<<23 | mant<<13)
	case 31:
		return math.Float32frombits(sign | 0x7F800000 | mant<<13)
	}
}

func TestQuantizeQ4_K_Correctness(t *testing.T) {
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(i-128) * 0.01
	}

	out := quantizeQ4_K(src)
	if len(out) != blockQ4K {
		t.Fatalf("expected %d bytes, got %d", blockQ4K, len(out))
	}

	// dequantize and check error is small
	dBits := uint16(out[0]) | uint16(out[1])<<8
	dminBits := uint16(out[2]) | uint16(out[3])<<8
	d := float16Decode(dBits)
	dmin := float16Decode(dminBits)

	scales := out[4 : 4+kScaleSz]
	qs := out[4+kScaleSz:]

	var maxErr float32
	for j := range 8 {
		var sc, m uint8
		getScaleMinK4(j, scales, &sc, &m)
		blockD := d * float32(sc)
		blockM := dmin * float32(m)

		for ii := range 32 {
			idx := 32*j + ii
			group := idx / 64
			pos := idx % 64
			var q uint8
			if pos < 32 {
				q = qs[group*32+pos] & 0xF
			} else {
				q = qs[group*32+pos-32] >> 4
			}
			reconstructed := blockD*float32(q) - blockM
			err := float32Abs(reconstructed - src[idx])
			if err > maxErr {
				maxErr = err
			}
		}
	}

	if maxErr > 0.1 {
		t.Errorf("max quantization error too large: %f (expected < 0.1)", maxErr)
	}
}
