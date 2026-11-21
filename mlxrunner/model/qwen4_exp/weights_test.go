package qwen4_exp

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestSparseMoEGlobalScales(t *testing.T) {
	const experts, hidden, intermediate = 3, 64, 32
	for _, tc := range []struct {
		name, prefix, weightSuffix, scaleSuffix string
		gate, down                              []float32
	}{
		{"unscaled", "model.language_model.layers.0.mlp", "", ".global_scale", nil, nil},
		{"per expert", "model.language_model.layers.0.mlp", "", ".global_scale", []float32{0.5, 2, 4}, []float32{2, 0.25, 0.5}},
		{"scalar MTP", "mtp.layers.0.mlp", ".weight", "_scale_2", []float32{0.5}, []float32{2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				values := func(n int, factor float32) []float32 {
					v := make([]float32, n)
					for i := range v {
						v[i] = float32((i*37)%101-50) * factor
					}
					return v
				}
				tensors := map[string]*mlx.Array{}
				for name, dims := range map[string][]int{
					"gate": {experts, hidden}, "shared_expert_gate": {1, hidden},
					"shared_expert.gate_proj": {intermediate, hidden},
					"shared_expert.up_proj":   {intermediate, hidden},
					"shared_expert.down_proj": {hidden, intermediate},
				} {
					tensors[tc.prefix+"."+name+".weight"] = mlx.FromValues(values(dims[0]*dims[1], 0.001), dims...)
				}
				dense := make([]*mlx.Array, 2)
				for i, p := range []struct {
					name       string
					rows, cols int
					global     []float32
				}{
					{"gate_up_proj", 2 * intermediate, hidden, tc.gate},
					{"down_proj", hidden, intermediate, tc.down},
				} {
					key := tc.prefix + ".experts." + p.name + tc.weightSuffix
					w, s, _ := mlx.Quantize(mlx.FromValues(values(experts*p.rows*p.cols, 0.001), experts, p.rows, p.cols), 16, 4, "nvfp4")
					tensors[key], tensors[key+"_scale"] = w, s
					dense[i] = mlx.Dequantize(w, s, nil, 16, 4, "nvfp4", nil).AsType(mlx.DTypeFloat32)
					if p.global != nil {
						gs := mlx.FromValues(p.global, len(p.global))
						tensors[key+tc.scaleSuffix] = gs
						dense[i] = mlx.Mul(dense[i], mlx.Reshape(gs, int32(len(p.global)), 1, 1))
					}
				}
				m := &Model{Config: &Config{HiddenSize: hidden, NumExpertsPerTok: 2}}
				m.quantGroup, m.quantBits, m.quantMode = 16, 4, "nvfp4"
				moe, err := m.loadSparseMoE(model.NewLinearFactory(tensors, 16, 4, "nvfp4", nil), tensors, tc.prefix)
				if err != nil {
					t.Fatal(err)
				}
				reference := *moe
				reference.GateUp = &expertProjection{Weight: mlx.Transpose(dense[0], 0, 2, 1)}
				reference.Down = &expertProjection{Weight: mlx.Transpose(dense[1], 0, 2, 1)}
				for _, tokens := range []int{2, 64} {
					x := mlx.FromValues(values(tokens*hidden, 0.02), 1, tokens, hidden)
					got, want := moe.Forward(x, m.Config), reference.Forward(x, m.Config)
					mlx.Eval(got, want)
					gotValues, wantValues := got.Floats(), want.Floats()
					var peak, errorMax float64
					for j, v := range wantValues {
						peak = math.Max(peak, math.Abs(float64(v)))
						errorMax = math.Max(errorMax, math.Abs(float64(gotValues[j]-v)))
					}
					if math.IsNaN(errorMax) || errorMax > 0.005*peak {
						t.Fatalf("%d tokens: max error %g, reference peak %g", tokens, errorMax, peak)
					}
				}
			})
		})
	}
}
