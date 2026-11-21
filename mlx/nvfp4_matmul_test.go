package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestGatherNVFP4QMM(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !hasMetalTensorOps() || metalMatmul() == metalMatmulUnsupported {
			t.Skip("requires Metal tensor ops and a known grouped reduction")
		}
		for _, shape := range []struct{ rows, experts, width, channels int }{
			{515, 128, 2304, 2304},
			{77, 7, 96, 67},     // partial output tile and partially occupied runs
			{583, 129, 64, 37},  // prefix scan spans more than 128 experts
			{4096, 1024, 32, 9}, // largest supported prefix scan
		} {
			Scoped(func() {
				rows, expertCount, width, channels := shape.rows, shape.experts, shape.width, shape.channels
				activation := make([]float32, rows*width)
				experts := make([]uint32, rows)
				for row := range rows {
					// Include empty experts, multiple full tiles and a partial tail.
					if row >= 65 {
						experts[row] = uint32(2 + 2*((row-65)*((expertCount-1)/2)/(rows-65)))
					}
					for col := range width {
						activation[row*width+col] = float32((row*61+col*109)%997-498) * .25
					}
				}
				// The leading product is 4,128,768. The following 1/16 terms are
				// below half an FP32 ULP individually, while their block sum is
				// large enough to change the result if the reduction is reordered.
				activation[0] = 1536
				for col := 1; col < 16; col++ {
					activation[col] = 0
				}
				for col := 16; col < width; col++ {
					activation[col] = .125
				}
				packed := make([]uint32, expertCount*width/8)
				scales := make([]uint8, expertCount*width/16)
				for i := range packed {
					packed[i] = uint32(i) * 2654435761
				}
				for i := range width / 8 {
					packed[i] = 0x11111111
				}
				packed[0] = 0x11111117
				for i := range scales {
					scales[i] = uint8(i % 127)
					if i%3 == 0 {
						scales[i] |= 128
					}
				}
				for i := range width / 16 {
					scales[i] = 0x38
				}
				scales[0] = 0x7e
				// Reconstructed NVFP4 input values have a BF16-exact significand.
				values := FromValues(activation, rows, 1, 1, width).AsType(DTypeBFloat16)
				ids := FromValues(experts, rows, 1)
				w := Add(FromValues(packed, expertCount, 1, width/8), Arange(0, float64(channels), 1, DTypeUint32).Reshape(1, channels, 1))
				s := BroadcastTo(FromValues(scales, expertCount, 1, width/16), int32(expertCount), int32(channels), int32(width/16))
				result := GatherNVFP4QMM(values, w, s, ids)
				if result == nil {
					t.Fatal("qualified grouped routes did not dispatch")
				}
				ref := GatherQMM(values.AsType(DTypeFloat32), w, s, nil, nil, ids, true, 16, 4, "nvfp4", nil, true)
				actual, want := result.Flatten(0, -1).Floats(), ref.Flatten(0, -1).Floats()
				control := float32(4128768)
				if metalMatmul() == metalMatmulTensor {
					// The accelerator adds each 16-product subtotal as a group.
					control += float32(width/16 - 1)
				}
				if want[0] != control {
					t.Fatalf("native rounding control = %v, want %v", want[0], control)
				}
				for i, v := range actual {
					if math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Fatalf("shape %+v, route %d, channel %d: got %08x, want %08x", shape, i/channels, i%channels, math.Float32bits(v), math.Float32bits(want[i]))
					}
				}
				if GatherNVFP4QMM(values, w, s, ids.Reshape(1, -1)) != nil {
					t.Fatal("ungrouped route shape dispatched")
				}
				if GatherNVFP4QMM(values.AsType(DTypeFloat32), w, s, ids) != nil {
					t.Fatal("FP32 activations were silently rounded to BF16")
				}
				few := values.Slice(Slice(0, 4), Slice(), Slice(), Slice())
				if GatherNVFP4QMM(few, w, s, ids.Slice(Slice(0, 4), Slice())) != nil {
					t.Fatal("short route list selected the sorted matrix reduction")
				}
			})
		}
	})
}

func TestGatherNVFP4QMMExtremes(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !hasMetalTensorOps() || metalMatmul() == metalMatmulUnsupported {
			t.Skip("requires Metal tensor ops and a known grouped reduction")
		}
		Scoped(func() {
			const rows, width, channels = 16, 64, 9
			// Include the smallest normal BF16 input of either sign. Paired
			// with a small scale, its dot can underflow to signed zero.
			patterns := []uint32{
				0, 0x00010000, 0x80800000, 0x7f7f0000, 0x00800000,
				0x7fc00000, 0x7f800000, 0xff800000, 0x35b00000, 0xb5100000,
			}
			data := make([]float32, rows*width)
			for i := range data {
				data[i] = math.Float32frombits(patterns[(i/width)%len(patterns)])
			}
			encodings := []uint8{0x7e, 0x38, 0x01, 0x37, 0x7f, 0xff, 0, 1, 0x80}
			blocks := make([]uint8, channels*width/16)
			for i := range blocks {
				blocks[i] = encodings[i/(width/16)]
			}
			w := BroadcastTo(FromValues([]uint32{0x7f129817}, 1, 1, 1), 1, channels, width/8)
			s := FromValues(blocks, 1, channels, width/16)
			ids := Zeros(DTypeInt32, rows, 1)
			x := FromValues(data, rows, 1, 1, width).AsType(DTypeBFloat16)
			y := GatherNVFP4QMM(x, w, s, ids)
			if y == nil {
				t.Fatal("valid grouped matmul did not dispatch")
			}
			ref := GatherQMM(x.AsType(DTypeFloat32), w, s, nil, nil, ids, true, 16, 4, "nvfp4", nil, true)
			actual, want := y.Flatten(0, -1).Floats(), ref.Flatten(0, -1).Floats()
			for i, v := range actual {
				if math.Float32bits(v) != math.Float32bits(want[i]) {
					t.Fatalf("row %d, channel %d: got %08x, want %08x", i/channels, i%channels, math.Float32bits(v), math.Float32bits(want[i]))
				}
			}
		})
	})
}

// Cancellation in a later tile can produce a subnormal even when every
// product in that tile is normal. Only the scalar reduction flushes that result.
func TestGatherNVFP4QMMCancellation(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !hasMetalTensorOps() || metalMatmul() == metalMatmulUnsupported {
			t.Skip("requires Metal tensor ops and a known grouped reduction")
		}
		Scoped(func() {
			const rows, width = 16, 64
			data := make([]float32, width)
			data[0], data[1], data[32] = 0x1p-126, 0x1p-126, -0x1p-116
			x := BroadcastTo(FromValues(data, 1, 1, 1, width).AsType(DTypeBFloat16), rows, 1, 1, width)
			// The first tile adds 1.5 * 2^-126. The second subtracts 2^-126.
			words := []uint32{0x12, 0, 0, 0, 1, 0, 0, 0}
			w := FromValues(words, 1, 1, width/8)
			s := FromValues([]uint8{0x38, 0, 1, 0}, 1, 1, width/16)
			ids := Zeros(DTypeInt32, rows, 1)
			got := GatherNVFP4QMM(x, w, s, ids)
			want := GatherQMM(x.AsType(DTypeFloat32), w, s, nil, nil, ids, true, 16, 4, "nvfp4", nil, true)
			if got == nil {
				t.Fatal("valid grouped matmul did not dispatch")
			}
			expected := float32(0)
			if metalMatmul() == metalMatmulTensor {
				expected = 0x1p-127
			}
			for i, v := range want.Flatten(0, -1).Floats() {
				if v != expected {
					t.Fatalf("native cancellation row %d: got %g, want %g", i, v, expected)
				}
			}
			if got.View(DTypeUint32).Equal(want.View(DTypeUint32)).AsType(DTypeInt32).Reshape(-1).SumAxis(0, false).Int() != rows {
				t.Fatal("grouped cancellation differs from native FP32")
			}
		})
	})
}
