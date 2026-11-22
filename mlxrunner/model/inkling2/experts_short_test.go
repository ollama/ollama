package inkling2

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// shortExpertBank returns an Inkling-shaped NVFP4 or MXFP8 bank of random
// weights without NaN encodings.
func shortExpertBank(random *rand.Rand, nvfp4 bool, experts, outputs, width int) expertBank {
	perWord, group, scale := 4, 32, func() uint8 { return uint8(118 + random.IntN(6)) }
	if nvfp4 {
		perWord, group, scale = 8, 16, func() uint8 { return uint8(48 + random.IntN(16)) }
	}
	words := make([]uint32, experts*outputs*width/perWord)
	for i := range words {
		words[i] = random.Uint32()
		if !nvfp4 {
			// Clear one exponent bit of each E4M3 byte so none is NaN.
			words[i] &^= 0x40404040
		}
	}
	blocks := make([]uint8, experts*outputs*width/group)
	for i := range blocks {
		blocks[i] = scale()
	}
	bank := expertBank{
		Weight:      mlx.FromValues(words, experts, outputs, width/perWord),
		BlockScales: mlx.FromValues(blocks, experts, outputs, width/group),
	}
	if nvfp4 {
		maxima := make([]float32, experts)
		for i := range maxima {
			maxima[i] = 0.05 + 0.15*random.Float32()
		}
		bank.Maxima, bank.Denominator = mlx.FromValues(maxima, experts), mlx.FromValues([]float32{2688}, 1)
	}
	return bank
}

// stockShortExperts is combineQuantized's short path with GatherQMM reading
// every route's expert.
func stockShortExperts(e routedExperts, flat, indices, weights *mlx.Array) *mlx.Array {
	rows, width := flat.Dim(0), flat.Dim(1)
	routes := rows * e.topK
	nvfp4 := e.GateUp.Maxima != nil
	gather := func(b expertBank, x *mlx.Array) *mlx.Array {
		if nvfp4 {
			return mlx.GatherQMM(x, b.Weight, b.BlockScales, nil, nil, indices, true, 16, 4, "nvfp4", nil, false)
		}
		return mlx.GatherQMM(x, b.Weight, b.BlockScales, nil, nil, indices, true, 32, 8, "mxfp8", nil, false)
	}
	x := flat.ExpandDims(1).ExpandDims(1)
	var values, xmax, down, downMax *mlx.Array
	if nvfp4 {
		values, xmax = mlx.QuantizeNVFP4Activation4Over6(x, nvfp4ActivationDenominator)
	} else {
		values = mlx.MXFP8Quantize(x, mlx.DTypeFloat32)
	}
	inner := gather(e.GateUp, values).Reshape(routes, -1)
	if nvfp4 {
		down, downMax = mlx.NVFP4SwiGLUQuantize(inner, xmax, e.GateUp.Maxima, e.GateUp.Denominator, indices, weights,
			e.topK, e.gateCap, e.upCap, nvfp4ActivationDenominator, mlx.DTypeFloat32)
	} else {
		down = mlx.MXFP8SwiGLUQuantize(inner, weights, e.gateCap, e.upCap, mlx.DTypeFloat32)
	}
	inner = gather(e.Down, down.Reshape(rows, e.topK, 1, -1)).Reshape(routes, width)
	return mlx.ExpertSum(inner, downMax, e.Down.Maxima, e.Down.Denominator, indices, nil, e.topK, nvfp4ActivationDenominator)
}

// Short forwards share each selected expert's weight reads across the rows
// that select it. Each route reduces as GatherQMM does, so on M5 the routed
// result matches the per-route gathers bit for bit; elsewhere it may differ
// by an FP32 rounding that a quantizer can carry to a nearby code.
func TestShortExpertsMatchStockGather(t *testing.T) {
	for _, nvfp4 := range []bool{true, false} {
		name := "mxfp8"
		if nvfp4 {
			name = "nvfp4"
		}
		t.Run(name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				if !mlx.MetalIsAvailable() {
					t.Skip("requires Metal")
				}
				mlx.Scoped(func() {
					const experts, topK, hidden, width = 32, 8, 2304, 1152
					random := rand.New(rand.NewPCG(11, uint64(len(name))))
					e := routedExperts{
						GateUp: shortExpertBank(random, nvfp4, experts, 2*width, hidden),
						Down:   shortExpertBank(random, nvfp4, experts, hidden, width),
						topK:   topK, gateCap: 4, upCap: 25,
					}
					mode := name
					epilogue := 0
					for rows := 1; rows <= 16; rows++ {
						input := make([]float32, rows*hidden)
						for i := range input {
							input[i] = float32(random.NormFloat64())
						}
						ids := make([]uint32, 0, rows*topK)
						gains := make([]float32, 0, rows*topK)
						for range rows {
							for _, expert := range random.Perm(experts)[:topK] {
								ids = append(ids, uint32(expert))
								gains = append(gains, random.Float32())
							}
						}
						flat := mlx.FromValues(input, rows, hidden).AsType(mlx.DTypeBFloat16)
						indices := mlx.FromValues(ids, rows, topK)
						weights := mlx.FromValues(gains, rows, topK)
						for _, probe := range []struct {
							bank  expertBank
							shape []int
							name  string
						}{{e.GateUp, []int{rows, 1, 1, hidden}, "gate/up"}, {e.Down, []int{rows, topK, 1, width}, "down"}} {
							// Before M3 a single row keeps the stock gather, which is faster there.
							x := mlx.Zeros(mlx.DTypeFloat32, probe.shape...)
							if mlx.ShortGatherQMM(x, probe.bank.Weight, probe.bank.BlockScales, indices, mode) == nil && rows > 1 {
								t.Fatalf("rows %d: short %s gather did not run", rows, probe.name)
							}
						}
						if nvfp4 && rows > 1 {
							// combineQuantized's gate/up gather finishes the SwiGLU itself
							// on devices that enable the epilogue.
							values, xmax := mlx.QuantizeNVFP4Activation4Over6(flat.ExpandDims(1).ExpandDims(1), nvfp4ActivationDenominator)
							if mlx.ShortGatherNVFP4SwiGLU(values.AsType(mlx.DTypeFloat32), e.GateUp.Weight, e.GateUp.BlockScales, indices, xmax,
								e.GateUp.Maxima, e.GateUp.Denominator, weights, topK, e.gateCap, e.upCap, nvfp4ActivationDenominator) != nil {
								epilogue++
							}
						}
						got := e.combine(flat, indices, weights, false).AsType(mlx.DTypeFloat32).Floats()
						want := stockShortExperts(e, flat, indices, weights).AsType(mlx.DTypeFloat32).Floats()
						var diff, scale float64
						same := 0
						for i := range want {
							if math.Float32bits(got[i]) == math.Float32bits(want[i]) {
								same++
							}
							diff += math.Abs(float64(got[i] - want[i]))
							scale += math.Abs(float64(want[i]))
						}
						if same != len(want) {
							t.Logf("rows %d: %d of %d outputs match bit for bit", rows, same, len(want))
						}
						if !(diff <= scale/1000) {
							t.Fatalf("rows %d: mean difference %g of mean magnitude %g", rows, diff/float64(len(want)), scale/float64(len(want)))
						}
					}
					if nvfp4 {
						// The epilogue's device gate does not depend on the row count
						// past one row, so it runs for every multi-row forward or none.
						t.Logf("the SwiGLU epilogue ran for %d of 15 multi-row forwards", epilogue)
						if epilogue != 0 && epilogue != 15 {
							t.Fatalf("the SwiGLU epilogue ran for only %d of 15 multi-row forwards", epilogue)
						}
					}
				})
			})
		})
	}
}
