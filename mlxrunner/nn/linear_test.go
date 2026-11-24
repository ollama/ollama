package nn

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestQuantizedLinearMXFP4MatchesDequantizedWeight(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		weightVals := make([]float32, 3*32)
		for i := range weightVals {
			weightVals[i] = float32((i%11)-5) / 7
		}
		inputVals := make([]float32, 2*32)
		for i := range inputVals {
			inputVals[i] = float32((i%7)-3) / 5
		}

		weight := mlx.FromValues(weightVals, 3, 32).AsType(mlx.DTypeBFloat16)
		input := mlx.FromValues(inputVals, 2, 32).AsType(mlx.DTypeBFloat16)
		mlx.Eval(weight, input)

		ql := NewQuantizedLinear(weight, nil, 32, 4, "mxfp4")
		if ql.QBiases != nil {
			t.Fatalf("mxfp4 qbiases = %v, want nil", ql.QBiases)
		}

		dequantizedWeight := mlx.Dequantize(ql.Weight, ql.Scales, ql.QBiases, 32, 4, "mxfp4", nil)
		mlx.Eval(dequantizedWeight)

		qOut := ql.Forward(input).AsType(mlx.DTypeFloat32)
		dOut := NewLinear(dequantizedWeight, nil).Forward(input).AsType(mlx.DTypeFloat32)
		mlx.Eval(qOut, dOut)

		got := qOut.Floats()
		want := dOut.Floats()
		if len(got) != len(want) {
			t.Fatalf("output length = %d, want %d", len(got), len(want))
		}

		for i := range got {
			if !approxEqual(got[i], want[i], 1e-3) {
				t.Fatalf("output[%d] = %.6f, want %.6f", i, got[i], want[i])
			}
		}
	})
}

// A dense nvfp4 projection carries the checkpoint's global scale through
// QuantizedMatmul, which applies it to the output in a single fused kernel.
// The dequantized weights are the reference.
func TestQuantizedLinearGlobalScaleMatchesDequantized(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("nvfp4 quantized_matmul requires a GPU backend")
		}
		const rows, cols, group = 64, 64, 16

		weightValues := make([]float32, rows*cols)
		for i := range weightValues {
			weightValues[i] = float32((i%23)-11) * 0.011
		}
		weight := mlx.FromValues(weightValues, rows, cols).AsType(mlx.DTypeBFloat16)
		packed, scales, _ := mlx.Quantize(weight, group, 4, "nvfp4")
		mlx.Eval(packed, scales)

		globalScale := mlx.FromValues([]float32{0.375}, 1)
		linear := &QuantizedLinear{
			Weight: packed, Scales: scales, GlobalScale: globalScale,
			GroupSize: group, Bits: 4, Mode: "nvfp4",
		}

		xValues := make([]float32, cols)
		for i := range xValues {
			xValues[i] = float32(i%7-3) / 8
		}
		x := mlx.FromValues(xValues, 1, cols).AsType(mlx.DTypeBFloat16)

		got := linear.Forward(x).AsType(mlx.DTypeFloat32)
		dense := mlx.Dequantize(packed, scales, nil, group, 4, "nvfp4", globalScale)
		want := mlx.Matmul(x.AsType(mlx.DTypeFloat32), mlx.Transpose(dense.AsType(mlx.DTypeFloat32), 1, 0))
		mlx.Eval(got, want)

		gotValues, wantValues := got.Floats(), want.Floats()
		if len(gotValues) != len(wantValues) {
			t.Fatalf("output length = %d, want %d", len(gotValues), len(wantValues))
		}
		for i := range gotValues {
			if math.IsNaN(float64(gotValues[i])) || math.IsInf(float64(gotValues[i]), 0) {
				t.Fatalf("output[%d] = %v, want finite", i, gotValues[i])
			}
			delta := math.Abs(float64(gotValues[i] - wantValues[i]))
			tolerance := 0.02 * math.Max(math.Abs(float64(wantValues[i])), 1)
			if delta > tolerance {
				t.Fatalf("output[%d] = %v, want %v (delta %v > %v)", i, gotValues[i], wantValues[i], delta, tolerance)
			}
		}
	})
}

func TestSwiGLU(t *testing.T) {
	for _, tt := range []struct {
		name       string
		quantized  bool
		withScales bool
		withBias   bool
	}{
		{name: "plain"},
		{name: "quantized", quantized: true},
		{name: "quantized with global scales", quantized: true, withScales: true},
		{name: "quantized with global scales and bias", quantized: true, withScales: true, withBias: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				if tt.quantized && !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
					t.Skip("nvfp4 quantized_matmul requires a GPU backend")
				}
				const width, group = 64, 16

				weightValues := make([]float32, width*width)
				for i := range weightValues {
					weightValues[i] = float32((i%23)-11) * 0.011
				}
				weight := mlx.FromValues(weightValues, width, width).AsType(mlx.DTypeBFloat16)
				var bias *mlx.Array
				if tt.withBias {
					biasValues := make([]float32, width)
					for i := range biasValues {
						biasValues[i] = float32(i%5-2) / 4
					}
					bias = mlx.FromValues(biasValues, width).AsType(mlx.DTypeBFloat16)
				}

				var gate, up LinearLayer
				if tt.quantized {
					gateQuantized := NewQuantizedLinear(weight, bias, group, 4, "nvfp4")
					upQuantized := NewQuantizedLinear(weight, bias, group, 4, "nvfp4")
					if tt.withScales {
						gateQuantized.GlobalScale = mlx.FromValue(float32(mlx.Nvfp4MaxProduct) * 0.75)
						upQuantized.GlobalScale = mlx.FromValue(float32(mlx.Nvfp4MaxProduct) * 1.25)
					}
					gate, up = gateQuantized, upQuantized
				} else {
					gate, up = NewLinear(weight, bias), NewLinear(weight, bias)
				}

				xValues := make([]float32, width)
				for i := range xValues {
					xValues[i] = float32(i%7-3) / 8
				}
				x := mlx.FromValues(xValues, 1, 1, width).AsType(mlx.DTypeBFloat16)
				got := SwiGLU(gate, up, x).AsType(mlx.DTypeFloat32)
				want := mlx.SwiGLU(gate.Forward(x), up.Forward(x)).AsType(mlx.DTypeFloat32)
				mlx.Eval(got, want)
				if gotValues, wantValues := got.Floats(), want.Floats(); !slices.Equal(gotValues, wantValues) {
					t.Fatalf("SwiGLU() = %v, want %v", gotValues, wantValues)
				}
			})
		})
	}
}

// Batched tokens share one weight matrix, including quantization scales and
// the layer bias. Compare each row with an independent request so flattening
// cannot change row order or apply a scale along the request dimension.
func TestQuantizedLinearBatch(t *testing.T) {
	for _, mode := range []string{"affine", "mxfp4", "mxfp8", "nvfp4"} {
		for _, rows := range []int{2, 4, 8} {
			t.Run(fmt.Sprintf("%s/rows%d", mode, rows), func(t *testing.T) {
				mlxtest.Run(t, func(t *mlxtest.T) {
					if mode != "affine" && !mlx.GPUIsAvailable() {
						t.Skip("FP quantized matmul requires a GPU backend")
					}
					const inputs, outputs = 512, 64
					group, bits := 32, 4
					if mode == "mxfp8" {
						bits = 8
					}
					if mode == "nvfp4" {
						group = 16
					}
					w := make([]float32, outputs*inputs)
					for i := range w {
						w[i] = float32(i%23-11) / 32
					}
					bias := make([]float32, outputs)
					for i := range bias {
						bias[i] = float32(i%5-2) / 4
					}
					linear := NewQuantizedLinear(mlx.FromValues(w, outputs, inputs).AsType(mlx.DTypeBFloat16), mlx.FromValues(bias, outputs), group, bits, mode)
					scales := make([]float32, outputs)
					for i := range scales {
						scales[i] = float32(i%3+1) / 4
					}
					linear.GlobalScale = mlx.FromValues(scales, outputs)
					for _, tokens := range []int{1, 3} {
						values := make([]float32, rows*tokens*inputs)
						for i := range values {
							values[i] = float32(i%13-6) / 16
						}
						x := mlx.FromValues(values, rows, tokens, inputs).AsType(mlx.DTypeBFloat16)
						got := linear.Forward(x)
						if want := []int{rows, tokens, outputs}; !slices.Equal(got.Dims(), want) {
							t.Fatalf("shape = %v, want %v", got.Dims(), want)
						}
						for row := range rows {
							want := linear.Forward(x.Slice(mlx.Slice(row, row+1), mlx.Slice(), mlx.Slice())).AsType(mlx.DTypeFloat32).Floats()
							actual := got.Slice(mlx.Slice(row, row+1), mlx.Slice(), mlx.Slice()).AsType(mlx.DTypeFloat32).Floats()
							for i := range want {
								if !approxEqual(actual[i], want[i], 0.02) {
									t.Fatalf("tokens %d row %d output %d = %g, want %g", tokens, row, i, actual[i], want[i])
								}
							}
						}
					}
				})
			})
		}
	}
}
