package mlx

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

func BenchmarkNVFP4BlocksEager(b *testing.B) {
	benchmarkNVFP4Blocks(b, false, func(blocks *Array) (*Array, *Array) {
		return quantizeNVFP4Blocks(blocks, FromValue(float32(1536)), FromValue(float32(1.0/1536)))
	})
}

func BenchmarkNVFP4BlocksCompiled(b *testing.B) {
	benchmarkNVFP4Blocks(b, true, func(blocks *Array) (*Array, *Array) {
		result := compiledNVFP4Blocks(blocks, FromValue(float32(1536)), FromValue(float32(1.0/1536)))
		return result[0], result[1]
	})
}

func BenchmarkNVFP4BlocksKernel(b *testing.B) {
	benchmarkNVFP4Blocks(b, false, func(blocks *Array) (*Array, *Array) {
		values, maxima := fastNVFP4Activation(blocks.Reshape(blocks.Dim(0), -1), 1536)
		if values == nil {
			panic("NVFP4 activation unavailable")
		}
		return values, maxima
	})
}

func benchmarkNVFP4Blocks(b *testing.B, metalOnly bool, quantize func(*Array) (*Array, *Array)) {
	worker := mlxTestThread(b)
	for _, rows := range []int{8, 512} {
		b.Run(fmt.Sprintf("rows%d", rows), func(b *testing.B) {
			var supported bool
			if err := worker.Do(b.Context(), func() error {
				metal := MetalIsAvailable()
				supported = metal
				if !metal && !metalOnly {
					values, _ := fastNVFP4Activation(Zeros(DTypeBFloat16, 1, 2304), 1536)
					supported = values != nil
				}
				if !supported {
					return nil
				}
				if metal {
					EnableCompile()
				}
				Scoped(func() {
					const width = 2304
					data := make([]float32, rows*width)
					random := rand.New(rand.NewPCG(41, 723))
					for i := range data {
						data[i] = float32(random.NormFloat64())
					}
					blocks := FromValues(data, rows, width/16, 16).AsType(DTypeBFloat16)
					Eval(blocks)
					run := func() { Eval(quantize(blocks)) }
					Scoped(run)
					b.ResetTimer()
					for range b.N {
						Scoped(run)
					}
					b.StopTimer()
				})
				return nil
			}); err != nil {
				b.Fatal(err)
			}
			if !supported {
				b.Skip("requires a qualified activation backend")
			}
		})
	}
}

func requireNVFP4Activation(t *mlxthreadtest.T) {
	t.Helper()
	values, _ := fastNVFP4Activation(Zeros(DTypeBFloat16, 1, 2304), 1536)
	if values == nil {
		if MetalIsAvailable() {
			t.Fatal("Metal declined an eligible NVFP4 activation")
		}
		t.Skip("requires a qualified NVFP4 activation device")
	}
}

func TestNVFP4ActivationMatchesEagerAcrossShapes(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		requireNVFP4Activation(t)
		for _, input := range []*Array{
			Zeros(DTypeBFloat16, 1, 17),
			Zeros(DTypeBFloat16, 1, 4112),
			Zeros(DTypeFloat32, 1, 2304),
			BroadcastTo(FromValue(float32(0)).AsType(DTypeBFloat16), 1<<22, 1152),
		} {
			if values, maxima := fastNVFP4Activation(input, 1536); values != nil || maxima != nil {
				t.Fatalf("unexpected NVFP4 specialization for %v %v", input.Dims(), input.DType())
			}
		}
		random := rand.New(rand.NewPCG(41, 723))
		for _, tc := range [][4]int{{1, 1, 2304, 1536}, {8, 8, 1152, 1536}, {67, 1, 2304, 1536}, {13, 8, 1152, 1536}, {2, 1, 16, Nvfp4MaxProduct}, {3, 2, 528, Nvfp4MaxProduct}, {1, 2, 4096, 1536}} {
			Scoped(func() {
				rows, routes, width, maxProduct := tc[0], tc[1], tc[2], tc[3]
				data := make([]float32, rows*routes*width)
				for i := range data {
					// Include empty blocks, equal magnitudes and random rounding decisions.
					switch (i / 16) % 3 {
					case 1:
						data[i] = float32(i%7-3) * 0.5
					case 2:
						data[i] = float32(random.NormFloat64())
					}
				}
				x := FromValues(data, rows, routes, 1, width).AsType(DTypeBFloat16)
				want, wantMax := quantizeNVFP4Blocks(x.Reshape(-1, width/16, 16), FromValue(float32(maxProduct)), FromValue(1/float32(maxProduct)))
				got, gotMax := fastNVFP4Activation(x, maxProduct)
				if got == nil {
					t.Fatalf("Metal activation unavailable for shape %v", tc)
				}
				if !slices.Equal(got.Dims(), x.Dims()) || !slices.Equal(gotMax.Dims(), []int{rows, routes, 1, 1}) ||
					!slices.Equal(got.Floats(), want.Floats()) || !slices.Equal(gotMax.Floats(), wantMax.Floats()) {
					t.Errorf("shape %v: Metal activation or row maxima differ from eager", tc)
				}
			})
		}
	})
}

func TestNVFP4ActivationFallback(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		// FP32 always takes the graph path, including on Metal.
		x := Arange(-64, 64, 1, DTypeFloat32).Reshape(2, 64)
		for _, maxProduct := range []int{1536, Nvfp4MaxProduct} {
			got, gotMax := QuantizeNVFP4Activation4Over6(x, maxProduct)
			want, wantMax := quantizeNVFP4Blocks(x.Reshape(2, 4, 16), FromValue(float32(maxProduct)), FromValue(1/float32(maxProduct)))
			if !slices.Equal(got.Dims(), x.Dims()) || !slices.Equal(gotMax.Dims(), []int{2, 1}) ||
				!slices.Equal(got.Floats(), want.Floats()) || !slices.Equal(gotMax.Floats(), wantMax.Floats()) {
				t.Errorf("graph fallback differs for max product %d", maxProduct)
			}
		}
		if CUDAIsAvailable() {
			_, arch := kernelDevice()
			values, maxima := fastNVFP4Activation(x.AsType(DTypeBFloat16), 1536)
			if (values != nil) != (arch == "sm_120") || (maxima != nil) != (arch == "sm_120") {
				t.Fatalf("unexpected activation dispatch for %s", arch)
			}
		}
	})
}
