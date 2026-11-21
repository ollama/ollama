package mlx

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func TestGatherNVFP4QMV(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !isM5MatrixDevice(kernelDevice()) {
			t.Skip("requires qualified Metal NVFP4 matvec")
		}
		for _, width := range []int{2304, 1152} {
			Scoped(func() {
				// Across rows, experts appear once, twice, three times, or four times.
				ids := FromValues([]uint32{0, 1, 2, 3, 4, 5, 6, 7, 1, 8, 9, 0, 2, 10, 11, 12, 13, 14, 0, 2, 15, 16, 17, 18, 19, 20, 21, 22, 0, 23, 24, 25}, 4, 8)
				n := 1
				if width == 1152 {
					n = 8
				}
				data := make([]float32, 4*n*width)
				for i := range data {
					data[i] = float32((i*337)%1981-990) * 0.25
				}
				x := FromValues(data, 4, n, 1, width)
				words := make([]uint32, 128*width/8)
				for i := range words {
					words[i] = uint32(i) * 2654435761
				}
				w := Add(FromValues(words, 128, 1, width/8), Arange(0, 2304, 1, DTypeUint32).Reshape(1, 2304, 1))
				// Each output row uses a single E4M3FN encoding across its K
				// blocks. All 256 encodings reach the output independently.
				blocks := make([]uint8, 128*2304)
				for i := range blocks {
					blocks[i] = uint8(i)
				}
				s := BroadcastTo(FromValues(blocks, 128, 2304, 1), 128, 2304, int32(width/16))
				for _, routes := range []*Array{ids, Zeros(DTypeUint32, 4, 8)} {
					y := GatherNVFP4QMV(x, w, s, routes)
					if y == nil {
						t.Fatalf("width %d did not dispatch", width)
					}
					reference := GatherQMM(x, w, s, nil, nil, routes, true, 16, 4, "nvfp4", nil, false)
					actual, expected := y.Flatten(0, -1).Floats(), reference.Flatten(0, -1).Floats()
					for i, v := range actual {
						if math.Float32bits(v) != math.Float32bits(expected[i]) {
							t.Fatalf("width %d, output %d (scale %#02x): got %08x, expected %08x", width, i, uint8(i%2304), math.Float32bits(v), math.Float32bits(expected[i]))
						}
					}
				}
				invalid := Add(ids, FromValue(128).AsType(DTypeUint32))
				for i, v := range GatherNVFP4QMV(x, w, s, invalid).Flatten(0, -1).Floats() {
					if !math.IsNaN(float64(v)) {
						t.Fatalf("invalid route output %d = %g", i, v)
					}
				}
				if GatherNVFP4QMV(x.Slice(Slice(0, 1), Slice(), Slice(), Slice()), w, s, ids) != nil {
					t.Fatal("mismatched route-row count dispatched")
				}
			})
		}
	})
}

func TestGatherNVFP4QMVRectangular(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !isM5MatrixDevice(kernelDevice()) {
			t.Skip("requires qualified Metal NVFP4 matvec")
		}
		Scoped(func() {
			const rows, topK, experts, width, channels = 3, 4, 7, 96, 67
			ids := FromValues([]int32{0, 1, 4, 6, 1, 1, 2, 0, 1, 1, 2, 4}, rows, topK)
			words := make([]uint32, experts*channels*width/8)
			blocks := make([]uint8, experts*channels*width/16)
			for i := range words {
				words[i] = uint32(i) * 2654435761
			}
			for i := range blocks {
				blocks[i] = uint8((i * 19) % 127)
			}
			w := FromValues(words, experts, channels, width/8)
			s := FromValues(blocks, experts, channels, width/16)
			for _, inputRoutes := range []int{1, topK} {
				data := make([]float32, rows*inputRoutes*width)
				for i := range data {
					data[i] = float32((i*17)%57-21) / float32(1+i%23)
				}
				x := FromValues(data, rows, inputRoutes, 1, width)
				got := GatherNVFP4QMV(x, w, s, ids)
				if got == nil {
					t.Fatalf("input routes %d did not dispatch", inputRoutes)
				}
				ref := GatherQMM(x, w, s, nil, nil, ids, true, 16, 4, "nvfp4", nil, false)
				actual, expected := got.Flatten(0, -1).Floats(), ref.Flatten(0, -1).Floats()
				for i, y := range actual {
					if math.Float32bits(y) != math.Float32bits(expected[i]) {
						t.Fatalf("input routes %d, row %d, channel %d: got %08x, want %08x", inputRoutes, i/channels, i%channels, math.Float32bits(y), math.Float32bits(expected[i]))
					}
				}
			}
			// Native GatherQMM uses a different reduction at this alignment.
			if GatherNVFP4QMV(Zeros(DTypeFloat32, 1, 1, 1, 512), Zeros(DTypeUint32, 1, 8, 64),
				Zeros(DTypeUint8, 1, 8, 32), Zeros(DTypeUint32, 1, 1)) != nil {
				t.Fatal("aligned native QMV dispatched")
			}
		})
	})
}
