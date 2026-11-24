package inkling2

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestDynamicShortConvolutionRetainsHistoryAcrossDecode(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			convolution := shortConv{Weight: mlx.FromValues([]float32{1, 2}, 2, 1), heads: 1}
			taps := nn.NewLinear(mlx.FromValues([]float32{0, 1}, 2, 1), nil)
			state := cache.NewRecurrentCache(1, 1, 1, 1, 1)
			defer state.Free()
			input := mlx.FromValues([]float32{1, 2, 3}, 1, 3, 1)
			prefill := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 3), SeqQueryLens: []int32{3}}
			full := convolution.forward(input, taps.Forward(input), prefill, nil).Floats()
			for index, want := range []float64{2, 8, 16} {
				chunk := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1), SeqOffsets: []int32{int32(index)}, SeqQueryLens: []int32{1}}
				token := input.Slice(mlx.Slice(), mlx.Slice(index, index+1), mlx.Slice())
				got := convolution.forward(token, taps.Forward(token), chunk, state).Floats()[0]
				if math.Abs(float64(got)-want/math.Sqrt(5)) > 1e-6 || got != full[index] {
					t.Errorf("token %d: decoded %g, prefill %g, want %g", index, got, full[index], want/math.Sqrt(5))
				}
			}
		})
	})
}

// A verification forward with snapshot splits matches token-by-token decode,
// and each split's snapshot restores the state decode had after that token.
// Both convolution routes, the kernel and the graph, must keep this contract.
func TestShortConvolutionSnapshotsEachSplit(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("the convolution kernel needs Metal")
		}
		mlx.Scoped(func() {
			const channels, taps, prefix, tokens = 64, 4, 3, 5
			random := rand.New(rand.NewPCG(5, 8))
			values := func(shape ...int) *mlx.Array {
				count := 1
				for _, size := range shape {
					count *= size
				}
				v := make([]float32, count)
				for i := range v {
					v[i] = float32(random.NormFloat64())
				}
				return mlx.FromValues(v, shape...).AsType(mlx.DTypeBFloat16)
			}
			conv := shortConv{Weight: values(taps, channels)}
			history, x := values(1, prefix, channels), values(1, tokens, channels)
			step := func(c *cache.RecurrentCache, in *mlx.Array, offset int) *mlx.Array {
				n := in.Dim(1)
				return conv.forward(in, nil, &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, n), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(n)}}, c)
			}
			state := func(c *cache.RecurrentCache) []float32 {
				return c.Get(&batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1)}, mlx.DTypeBFloat16).ConvState().AsType(mlx.DTypeFloat32).Floats()
			}

			decoded := cache.NewRecurrentCache(taps-1, channels, 1, 1, 1)
			defer decoded.Free()
			step(decoded, history, 0)
			var wantY []float32
			var wantStates [][]float32
			for i := range tokens {
				y := step(decoded, x.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()), prefix+i)
				wantY = append(wantY, y.AsType(mlx.DTypeFloat32).Floats()...)
				wantStates = append(wantStates, state(decoded))
			}

			verified := cache.NewRecurrentCache(taps-1, channels, 1, 1, 1)
			defer verified.Free()
			step(verified, history, 0)
			verified.PrepareSnapshots([]int{prefix + 1, prefix + 2, prefix + 3, prefix + 4})
			if got := step(verified, x, prefix).AsType(mlx.DTypeFloat32).Floats(); !slices.Equal(got, wantY) {
				t.Errorf("verification outputs differ from decode")
			}
			if got := state(verified); !slices.Equal(got, wantStates[tokens-1]) {
				t.Errorf("final state differs from decode")
			}
			snapshots := verified.TakeSnapshots()
			defer func() {
				for _, s := range snapshots {
					s.Close()
				}
			}()
			if len(snapshots) != tokens-1 {
				t.Fatalf("%d snapshots, want %d", len(snapshots), tokens-1)
			}
			for i, s := range snapshots {
				if !verified.Restore(s, prefix+1+i) {
					t.Fatalf("snapshot %d did not restore", i)
				}
				if got := state(verified); !slices.Equal(got, wantStates[i]) {
					t.Errorf("snapshot %d state differs from decode after token %d", i, i)
				}
			}
		})
	})
}

// forwardAdd adds a fixed convolution to the residual in the kernel's launch.
// Its sum and the state it stores match the separate convolution and FP32 sum
// bit for bit, for decode, verification and prefill widths.
func TestShortConvolutionAddMatchesSeparateSum(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("the convolution kernel needs Metal")
		}
		const channels, taps, prefix = 2304, 4, 5
		rng := rand.New(rand.NewPCG(19, 7))
		mlx.Scoped(func() {
			conv := shortConv{Weight: draftTestValues(rng, 0.5, taps, channels)}
			for _, tokens := range []int{1, 4, 8, 17} {
				for _, dtypes := range [][2]mlx.DType{
					{mlx.DTypeFloat32, mlx.DTypeFloat32},
					{mlx.DTypeFloat32, mlx.DTypeBFloat16},
					{mlx.DTypeBFloat16, mlx.DTypeBFloat16},
				} {
					mlx.Scoped(func() {
						name := fmt.Sprintf("%d tokens, %v residual, %v sum", tokens, dtypes[0], dtypes[1])
						history := draftTestValues(rng, 1, 1, prefix, channels)
						x := draftTestValues(rng, 1, 1, tokens, channels)
						residual := draftTestValues(rng, 2, 1, tokens, channels).AsType(dtypes[0])
						if sum, _ := mlx.ShortConvolutionAdd(mlx.Zeros(x.DType(), 1, taps-1, channels), x, conv.Weight, residual, 0.5, dtypes[1]); sum == nil {
							t.Fatalf("%s: the fused convolution add declined", name)
						}
						fusedCache, separateCache := cache.NewRecurrentCache(taps-1, channels, 1, 1, 1), cache.NewRecurrentCache(taps-1, channels, 1, 1, 1)
						defer fusedCache.Free()
						defer separateCache.Free()
						run := func(c *cache.RecurrentCache, fused bool) (sum, state *mlx.Array) {
							conv.forward(history, nil, &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, prefix), SeqOffsets: []int32{0}, SeqQueryLens: []int32{prefix}}, c)
							b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, tokens), SeqOffsets: []int32{prefix}, SeqQueryLens: []int32{int32(tokens)}}
							if fused {
								sum = conv.forwardAdd(x, residual, dtypes[1], b, c)
							} else {
								sum = sumFP32(residual, conv.forward(x, nil, b, c), dtypes[1])
							}
							return sum, c.Get(&batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 1)}, x.DType()).ConvState()
						}
						gotSum, gotState := run(fusedCache, true)
						wantSum, wantState := run(separateCache, false)
						requireIdentical(t, name, gotSum, wantSum)
						requireIdentical(t, name+" state", gotState, wantState)
					})
				}
			}
		})
	})
}

func TestMetalConvTapsMatchEager(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		mlx.EnableCompile()
		random := rand.New(rand.NewPCG(71, 219))
		makeValues := func(shape ...int) *mlx.Array {
			count := 1
			for _, size := range shape {
				count *= size
			}
			values := make([]float32, count)
			for i := range values {
				values[i] = float32(random.NormFloat64())
			}
			return mlx.FromValues(values, shape...).AsType(mlx.DTypeBFloat16)
		}
		const rows, channels, taps, heads = 2, 128, 4, 8
		for _, length := range []int{1, 8, 17, 64} {
			mlx.Scoped(func() {
				x := makeValues(rows, length, channels)
				previous := makeValues(rows, taps-1, channels)
				window := mlx.Concatenate([]*mlx.Array{previous, x}, 1)
				for _, dynamic := range []bool{false, true} {
					var inputs []*mlx.Array
					if dynamic {
						weight := makeValues(taps, channels)
						shift := makeValues(rows, length, heads, taps).AsType(mlx.DTypeFloat32)
						inputs = convTapArgs(window, x, weight, shift)
					} else {
						inputs = convTapArgs(window, x, makeValues(channels, taps).Transpose(1, 0), nil)
					}
					eager := convTaps(inputs...)[0].AsType(mlx.DTypeFloat32).Floats()
					compiled := compiledConvTaps(inputs...)[0].AsType(mlx.DTypeFloat32).Floats()
					if !slices.Equal(compiled, eager) {
						t.Errorf("length %d dynamic %t: compiled convolution changed the BF16 result", length, dynamic)
					}
				}
			})
		}
	})
}
