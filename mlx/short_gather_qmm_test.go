package mlx

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

// shortGatherRuns reports whether ShortGatherQMM runs rows on the device
// rather than leaving them to the stock GatherQMM.
func shortGatherRuns(mode string, rows int, perRoute bool, architecture string) bool {
	_, ok := tuneShortGather(mode, rows, perRoute, architecture)
	return ok
}

// shortGatherBank returns random packed weights and block scales without NaN
// encodings for an expert bank of either format.
func shortGatherBank(random *rand.Rand, mode string, experts, channels, width int) (weight, scales *Array) {
	perWord, group, scale := 8, 16, func() uint8 { return uint8(40 + random.IntN(24)) }
	if mode == "mxfp8" {
		perWord, group, scale = 4, 32, func() uint8 { return uint8(120 + random.IntN(12)) }
	}
	words := make([]uint32, experts*channels*width/perWord)
	for i := range words {
		words[i] = random.Uint32()
		if mode == "mxfp8" {
			// Clear one exponent bit of each E4M3 byte so none is NaN.
			words[i] &^= 0x40404040
		}
	}
	blocks := make([]uint8, experts*channels*width/group)
	for i := range blocks {
		blocks[i] = scale()
	}
	return FromValues(words, experts, channels, width/perWord), FromValues(blocks, experts, channels, width/group)
}

// shortGatherRoutes picks topK distinct experts per row. Spread routes never
// share an expert, random ones share by chance and hot ones draw from a pool
// of ten, so every row repeats experts and long runs split.
func shortGatherRoutes(random *rand.Rand, pattern string, rows, topK, experts int) *Array {
	ids := make([]int32, 0, rows*topK)
	for row := range rows {
		switch pattern {
		case "spread":
			for k := range topK {
				ids = append(ids, int32((row*topK+k)%experts))
			}
		case "random":
			for _, e := range random.Perm(experts)[:topK] {
				ids = append(ids, int32(e))
			}
		case "hot":
			for _, e := range random.Perm(10)[:topK] {
				ids = append(ids, int32(e*experts/10))
			}
		}
	}
	return FromValues(ids, rows, topK)
}

func TestShortGatherQMM(t *testing.T) {
	for _, mode := range []string{"nvfp4", "mxfp8"} {
		t.Run(mode, func(t *testing.T) {
			withMLXThread(t, func(t *mlxthreadtest.T) {
				if !MetalIsAvailable() {
					t.Skip("requires Metal")
				}
				group, bits := 16, 4
				quantize := func(x *Array) *Array {
					values, _ := QuantizeNVFP4Activation4Over6(x, 1536)
					return values
				}
				if mode == "mxfp8" {
					group, bits = 32, 8
					quantize = func(x *Array) *Array { return MXFP8Quantize(x, DTypeFloat32) }
				}
				random := rand.New(rand.NewPCG(5, 9))
				const experts, topK, channels = 128, 8, 256
				// Inkling's gate/up product reads one vector per row, its down
				// product one per route.
				for _, shape := range []struct{ width, vectors int }{{2304, 1}, {1152, topK}} {
					weight, scales := shortGatherBank(random, mode, experts, channels, shape.width)
					for rows := 1; rows <= 16; rows++ {
						// The documented single-row fallback before M3 keeps the stock gather.
						if arch := kernelArchitecture(); !shortGatherRuns(mode, rows, shape.vectors != 1, arch) {
							continue
						}
						for _, pattern := range []string{"spread", "random", "hot"} {
							values := make([]float32, rows*shape.vectors*shape.width)
							for i := range values {
								values[i] = float32(random.NormFloat64())
							}
							raw := FromValues(values, rows, shape.vectors, 1, shape.width)
							ids := shortGatherRoutes(random, pattern, rows, topK, experts)
							// The products of quantized activations and their sums
							// within a lane's 8 values are exact in FP32, so on M5
							// each route matches GatherQMM bit for bit. Other
							// activations may round differently in a few places.
							generation, _ := appleGPU(kernelArchitecture())
							for _, x := range []*Array{quantize(raw.AsType(DTypeBFloat16)), raw} {
								got := ShortGatherQMM(x, weight, scales, ids, mode)
								if got == nil {
									t.Fatalf("K=%d rows=%d: kernel did not run", shape.width, rows)
								}
								want := GatherQMM(x, weight, scales, nil, nil, ids, true, group, bits, mode, nil, false)
								label := fmt.Sprintf("K=%d rows=%d %s", shape.width, rows, pattern)
								var err error
								if x != raw && generation >= 17 {
									err = requireExact(label, got, want)
								} else {
									err = requireClose(label, got, want, 1e-5*float64(want.Abs().Flatten(0, -1).MaxAxis(0, false).Floats()[0]))
								}
								if err != nil {
									t.Fatal(err)
								}
							}
						}
					}
				}
			})
		})
	}
}

func TestShortGatherQMMEdges(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		if !MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		random := rand.New(rand.NewPCG(7, 3))
		const experts, channels, width = 8, 64, 512
		weight, scales := shortGatherBank(random, "nvfp4", experts, channels, width)
		x := FromValues(make([]float32, 3*width), 3, 1, 1, width)
		ids := FromValues([]int32{0, 1, 2, 9, 1, -1}, 3, 2)
		got := ShortGatherQMM(x, weight, scales, ids, "nvfp4")
		if got == nil {
			t.Fatal("kernel did not run")
		}
		out := got.Floats()
		for route := range 6 {
			for c := range channels {
				invalid := route == 3 || route == 5
				if v := out[route*channels+c]; math.IsNaN(float64(v)) != invalid {
					t.Fatalf("route %d channel %d = %g", route, c, v)
				}
			}
		}
		many := FromValues(make([]float32, 17*width), 17, 1, 1, width)
		if ShortGatherQMM(many, weight, scales, Zeros(DTypeInt32, 17, 8), "nvfp4") != nil {
			t.Error("accepted 136 routes")
		}
		if ShortGatherQMM(x.AsType(DTypeBFloat16), weight, scales, ids, "nvfp4") != nil {
			t.Error("accepted BF16 activations")
		}
	})
}

// TestShortGatherNVFP4SwiGLU checks the epilogue gather against the gather
// followed by NVFP4SwiGLUQuantize; both quantize the same BF16 values.
func TestShortGatherNVFP4SwiGLU(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		arch := kernelArchitecture()
		if tuning, _ := tuneShortGather("nvfp4", 2, false, arch); !MetalIsAvailable() || !tuning.epilogue {
			t.Skip("requires the Metal epilogue gather (M5 and later)")
		}
		random := rand.New(rand.NewPCG(13, 21))
		const experts, topK, channels, width, maxProduct = 32, 8, 2304, 2304, 1536
		weight, scales := shortGatherBank(random, "nvfp4", experts, channels, width)
		maxima := patternArray(DTypeFloat32, []int{experts}, 3, 0.25, 7, 17)
		denominator := FromValue(float32(2688))
		for _, rows := range []int{1, 2, 3, 4, 8, 16} {
			// The documented single-row fallback before M3 keeps the stock gather.
			if !shortGatherRuns("nvfp4", rows, false, arch) {
				continue
			}
			for _, pattern := range []string{"spread", "random", "hot"} {
				values := make([]float32, rows*width)
				for i := range values {
					values[i] = float32(random.NormFloat64())
				}
				x, xmax := QuantizeNVFP4Activation4Over6(FromValues(values, rows, 1, 1, width).AsType(DTypeBFloat16), maxProduct)
				ids := shortGatherRoutes(random, pattern, rows, topK, experts)
				scores := patternArray(DTypeFloat32, []int{rows, topK}, 0.5, 0.05, 5, 11)
				label := fmt.Sprintf("rows=%d %s", rows, pattern)
				activated := ShortGatherNVFP4SwiGLU(x, weight, scales, ids, xmax, maxima, denominator, scores, topK, 4, 25, maxProduct)
				if activated == nil {
					t.Fatalf("%s: kernel did not run", label)
				}
				got, gotMax := QuantizeNVFP4Activation4Over6(activated, maxProduct)
				inner := ShortGatherQMM(x, weight, scales, ids, "nvfp4").Reshape(rows*topK, channels)
				want, wantMax := NVFP4SwiGLUQuantize(inner, xmax, maxima, denominator, ids.Reshape(-1), scores.Reshape(-1), topK, 4, 25, maxProduct, DTypeFloat32)
				if err := requireExact(label, got, want); err != nil {
					t.Fatal(err)
				}
				if err := requireExact(label+" maxima", gotMax.Reshape(-1), wantMax); err != nil {
					t.Fatal(err)
				}
			}
		}
	})
}
