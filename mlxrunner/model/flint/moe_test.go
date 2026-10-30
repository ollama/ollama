package flint

import (
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/nn"
)

func TestPackedNVFP4ExpertRequiresGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		path := "model.layers.2.mlp.experts.gate_proj.weight"
		tensors := map[string]*mlx.Array{
			path:            mlx.FromValues(make([]uint32, 2*1*8), 2, 1, 8),
			path + "_scale": mlx.FromValues(make([]uint8, 2*1*4), 2, 1, 4),
		}
		cfg := &Config{NumExperts: 2, QuantGroupSize: 16, QuantBits: 4, QuantMode: "nvfp4"}
		_, err := loadPackedExpertProjection(tensors, path, 1, 64, cfg)
		if err == nil || !strings.Contains(err.Error(), "missing NVFP4 global scale") {
			t.Fatalf("loadPackedExpertProjection() error = %v, want missing global scale", err)
		}
	})
}

func TestFlintExpertResidualCompilePreservesBF16Rounding(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.EnableCompile()
		for _, tokens := range []int{1, 3, 16} {
			const topK, width = 6, 32
			selectedValues := make([]float32, tokens*topK*width)
			scoreValues := make([]float32, tokens*topK)
			sharedValues := make([]float32, tokens*width)
			residualValues := make([]float32, tokens*width)
			for i := range selectedValues {
				selectedValues[i] = float32((i*17)%79-39) / 37
			}
			for i := range scoreValues {
				scoreValues[i] = float32((i*7)%31+1) / 91
			}
			for i := range sharedValues {
				sharedValues[i] = float32((i*11)%71-35) / 29
			}
			for i := range residualValues {
				residualValues[i] = float32((i*13)%67-33) / 31
			}
			selected := mlx.FromValues(selectedValues, tokens, topK, width).AsType(mlx.DTypeBFloat16)
			scores := mlx.FromValues(scoreValues, tokens, topK)
			shared := mlx.FromValues(sharedValues, tokens, width).AsType(mlx.DTypeBFloat16)
			residual := mlx.FromValues(residualValues, tokens, width).AsType(mlx.DTypeBFloat16)
			want := mlx.Add(residual, mlx.Add(flintExpertWeightedSum(selected, scores), shared))
			got := flintExpertWeightedSumAdd(selected, scores, shared, residual)[0]
			want = mlx.Contiguous(want.AsType(mlx.DTypeFloat32), false)
			got = mlx.Contiguous(got.AsType(mlx.DTypeFloat32), false)
			mlx.Eval(want, got)
			wantValues := want.Floats()
			for i, actual := range got.Floats() {
				if actual != wantValues[i] {
					t.Fatalf("tokens=%d output[%d]=%g, previous path=%g", tokens, i, actual, wantValues[i])
				}
			}
		}
	})
}

func TestSparseMoERoutingAndSharedExpert(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		weight := func(values []float32, dims ...int) *mlx.Array {
			return mlx.FromValues(values, dims...).AsType(mlx.DTypeBFloat16)
		}
		shared := &DenseMLP{
			GateProj: nn.NewLinear(weight([]float32{1, 0}, 1, 2), nil),
			UpProj:   nn.NewLinear(weight([]float32{1, 0}, 1, 2), nil),
			DownProj: nn.NewLinear(weight([]float32{1, 0}, 2, 1), nil),
		}
		moe := &SparseMoE{
			Router: nn.NewLinear(weight([]float32{0, 0, 0, 0, -100, 0}, 3, 2), nil),
			Shared: shared,
			Experts: &Experts{
				Gate: &ExpertProjection{Weight: weight([]float32{1, 0, 1, 0, 1, 0}, 3, 1, 2)},
				Up:   &ExpertProjection{Weight: weight([]float32{1, 0, 1, 0, 1, 0}, 3, 1, 2)},
				Down: &ExpertProjection{Weight: weight([]float32{1, 0, 2, 0, 0, 0}, 3, 2, 1)},
			},
		}
		cfg := &Config{HiddenSize: 2, NumExperts: 3, NumExpertsPerTok: 2}
		routerCalled := false
		moe.route = func(inputs ...*mlx.Array) []*mlx.Array {
			routerCalled = true
			return flintRouter(cfg.NumExpertsPerTok)(inputs...)
		}
		input := weight([]float32{1, 1}, 1, 1, 2)
		out := moe.Forward(input, cfg)
		mlx.Eval(out)
		if !routerCalled {
			t.Fatal("compiled router was skipped")
		}
		got := out.AsType(mlx.DTypeFloat32).Floats()
		want := float64(2.5 / (1 + math.Exp(-1)))
		if len(got) != 2 || math.Abs(float64(got[0])-want) > 0.04 || math.Abs(float64(got[1])) > 0.01 {
			t.Fatalf("routed and shared expert output = %v, want approximately [%g, 0]", got, want)
		}
		original := flintExpertWeightedSumAdd
		defer func() { flintExpertWeightedSumAdd = original }()
		addCalled := false
		flintExpertWeightedSumAdd = func(inputs ...*mlx.Array) []*mlx.Array {
			addCalled = true
			return original(inputs...)
		}
		withResidual := moe.ForwardAdd(input, input, cfg)
		mlx.Eval(withResidual)
		if !addCalled {
			t.Fatal("compiled routed/shared/residual sum was skipped")
		}
		withResidualValues := withResidual.AsType(mlx.DTypeFloat32).Floats()
		if len(withResidualValues) != 2 || math.Abs(float64(withResidualValues[0])-want-1) > 0.04 || math.Abs(float64(withResidualValues[1])-1) > 0.01 {
			t.Fatalf("routed and shared expert output with residual = %v, want approximately [%g, 1]", withResidualValues, want+1)
		}
	})
}

func TestExpertsSortedPrefillMatchesUnsorted(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const tokens, width, experts, topK = 256, 32, 4, 2
		cfg := &Config{HiddenSize: width, NumExpertsPerTok: topK}
		projection := func(offset int) *ExpertProjection {
			values := make([]float32, experts*width*width)
			for i := range values {
				values[i] = float32((i*7+offset)%31-15) * 0.003
			}
			return &ExpertProjection{Weight: mlx.FromValues(values, experts, width, width).AsType(mlx.DTypeBFloat16)}
		}
		bank := &Experts{Gate: projection(1), Up: projection(5), Down: projection(9)}
		inputs := make([]float32, tokens*width)
		for i := range inputs {
			inputs[i] = float32((i*3)%23-11) * 0.02
		}
		routes := make([]int32, tokens*topK)
		for i := range routes {
			routes[i] = int32((i*3 + i/5) % experts)
		}
		x := mlx.FromValues(inputs, tokens, width).AsType(mlx.DTypeBFloat16)
		indices := mlx.FromValues(routes, tokens, topK)
		expanded := mlx.Reshape(x, tokens, 1, 1, width)
		gate := bank.Gate.Forward(expanded, indices)
		up := bank.Up.Forward(expanded, indices)
		want := bank.Down.Forward(flintSwiGLU(gate, up), indices)
		want = mlx.Contiguous(mlx.Reshape(want, tokens, topK, width).AsType(mlx.DTypeFloat32), false)
		got := mlx.Contiguous(bank.Forward(x, indices, cfg).AsType(mlx.DTypeFloat32), false)
		mlx.Eval(got, want)
		wantValues := want.Floats()
		for i, value := range got.Floats() {
			if delta := math.Abs(float64(value - wantValues[i])); delta > 1e-3 {
				t.Fatalf("sorted expert output[%d] = %g, unsorted = %g", i, value, wantValues[i])
			}
		}
	})
}

func TestSparseMoEPackedManifest(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		weight := func(dims ...int) *mlx.Array {
			count := 1
			for _, dim := range dims {
				count *= dim
			}
			return mlx.FromValues(make([]float32, count), dims...).AsType(mlx.DTypeBFloat16)
		}
		prefix := "model.layers.2.mlp"
		tensors := map[string]*mlx.Array{
			prefix + ".gate.weight":                    weight(2, 2),
			prefix + ".shared_expert.gate_proj.weight": weight(1, 2),
			prefix + ".shared_expert.up_proj.weight":   weight(1, 2),
			prefix + ".shared_expert.down_proj.weight": weight(2, 1),
			prefix + ".experts.gate_proj.weight":       weight(2, 1, 2),
			prefix + ".experts.up_proj.weight":         weight(2, 1, 2),
			prefix + ".experts.down_proj.weight":       weight(2, 2, 1),
		}
		cfg := &Config{HiddenSize: 2, MoeIntermediateSize: 1, NumExperts: 2}
		got, err := loadSparseMoE(model.NewLinearFactory(tensors, 0, 0, "", nil), tensors, "model.layers.2", cfg)
		if err != nil {
			t.Fatal(err)
		}
		if got.Experts.Gate.Weight != tensors[prefix+".experts.gate_proj.weight"] {
			t.Fatal("packed expert weights were not retained")
		}
	})
}

func TestSparseMoEPackedQuantizedMatchesBF16(t *testing.T) {
	for _, quant := range []string{"int4", "int8", "nvfp4", "mxfp4", "mxfp8"} {
		t.Run(quant, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				const experts, width = 2, 64
				groupSize, bits, mode := model.QuantizationParams(quant)
				cfg := &Config{
					HiddenSize: width, MoeIntermediateSize: width, SharedExpertIntermediateSize: width,
					NumExperts: experts, NumExpertsPerTok: 2, QuantGroupSize: groupSize, QuantBits: bits, QuantMode: mode,
				}
				prefix := "model.layers.2.mlp"
				full := map[string]*mlx.Array{
					prefix + ".gate.weight":                    mlx.Zeros(mlx.DTypeBFloat16, experts, width),
					prefix + ".shared_expert.gate_proj.weight": mlx.Zeros(mlx.DTypeBFloat16, width, width),
					prefix + ".shared_expert.up_proj.weight":   mlx.Zeros(mlx.DTypeBFloat16, width, width),
					prefix + ".shared_expert.down_proj.weight": mlx.Zeros(mlx.DTypeBFloat16, width, width),
				}
				quantized := make(map[string]*mlx.Array, len(full)+9)
				for key, value := range full {
					quantized[key] = value
				}
				for _, projection := range []string{"gate_proj", "up_proj", "down_proj"} {
					values := make([]float32, experts*width*width)
					for i := range values {
						values[i] = 0.5 + float32((i+i/4096*7)%13)*0.1
					}
					key := prefix + ".experts." + projection + ".weight"
					w := mlx.FromValues(values, experts, width, width).AsType(mlx.DTypeBFloat16)
					full[key] = w
					quantized[key], quantized[key+"_scale"], quantized[key+"_qbias"] = mlx.Quantize(w, groupSize, bits, mode)
					if mode == "nvfp4" {
						quantized[key+".global_scale"] = mlx.FromValue(float32(1))
					}
				}
				loadedFull, err := loadSparseMoE(model.NewLinearFactory(full, 0, 0, "", nil), full, "model.layers.2", cfg)
				if err != nil {
					t.Fatal(err)
				}
				loadedQuantized, err := loadSparseMoE(model.NewLinearFactory(quantized, groupSize, bits, mode, nil), quantized, "model.layers.2", cfg)
				if err != nil {
					t.Fatal(err)
				}
				if loadedQuantized.Experts.Gate.Scales == nil || loadedQuantized.Experts.Up.Scales == nil || loadedQuantized.Experts.Down.Scales == nil {
					t.Fatal("quantized expert banks were dequantized or not loaded")
				}
				if (loadedQuantized.Experts.Gate.GlobalScale != nil) != (mode == "nvfp4") {
					t.Fatalf("%s expert global scale was not loaded as expected", quant)
				}
				input := make([]float32, width)
				for i := range input {
					input[i] = 0.1 + float32(i%7)*0.02
				}
				x := mlx.FromValues(input, 1, 1, width).AsType(mlx.DTypeBFloat16)
				ref := mlx.Contiguous(loadedFull.Forward(x, cfg).AsType(mlx.DTypeFloat32), false)
				got := mlx.Contiguous(loadedQuantized.Forward(x, cfg).AsType(mlx.DTypeFloat32), false)
				mlx.Eval(ref, got)
				var errorSquared, refSquared float64
				for i, actual := range got.Floats() {
					want := ref.Floats()[i]
					delta := float64(actual - want)
					errorSquared += delta * delta
					refSquared += float64(want) * float64(want)
				}
				tolerance := 0.05
				if bits == 4 {
					tolerance = 0.20
				}
				if refSquared < 1 || math.Sqrt(errorSquared/refSquared) > tolerance {
					t.Fatalf("%s expert output relative RMS = %g, reference energy = %g", quant, math.Sqrt(errorSquared/refSquared), refSquared)
				}
			})
		})
	}
}
