package cohere2_moe

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestLoadPerExpertGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const experts, rows, cols, groupSize = 4, 32, 64, 16
		cfg := &Config{
			NumExperts:     experts,
			QuantGroupSize: groupSize,
			QuantBits:      4,
			QuantMode:      "nvfp4",
		}
		tensors := make(map[string]*mlx.Array)
		for e := range experts {
			base := fmt.Sprintf("model.layers.1.mlp.experts.%d.gate_proj.weight", e)
			tensors[base] = mlx.Zeros(mlx.DTypeUint32, rows, cols/8)
			tensors[base+"_scale"] = mlx.Zeros(mlx.DTypeUint8, rows, cols/groupSize)
			tensors[base+".global_scale"] = mlx.FromValues([]float32{float32(e + 1)}, 1)
		}

		got, err := readExperts(tensors, cfg, true, "model.layers.1", "gate_proj")
		if err != nil || got == nil {
			t.Fatalf("readExperts = %v, %v", got, err)
		}
		if got.Scales == nil || got.GlobalScale == nil {
			t.Fatal("per-expert weights did not use the quantized global-scale path")
		} else if dims := got.GlobalScale.Dims(); len(dims) != 1 || dims[0] != experts {
			t.Fatalf("global scale dims = %v, want [%d]", dims, experts)
		}
		if dims := got.Weight.Dims(); len(dims) != 3 || dims[0] != experts {
			t.Fatalf("weight dims = %v, want expert dimension %d", dims, experts)
		}

		// Reading released the tensors; rebuild them with one global scale missing.
		for e := range experts {
			base := fmt.Sprintf("model.layers.1.mlp.experts.%d.gate_proj.weight", e)
			tensors[base] = mlx.Zeros(mlx.DTypeUint32, rows, cols/8)
			tensors[base+"_scale"] = mlx.Zeros(mlx.DTypeUint8, rows, cols/groupSize)
			if e < experts-1 {
				tensors[base+".global_scale"] = mlx.FromValues([]float32{float32(e + 1)}, 1)
			}
		}
		if got, err := readExperts(tensors, cfg, true, "model.layers.1", "gate_proj"); err == nil {
			t.Fatalf("readExperts accepted an incomplete global-scale set: %v", got)
		}
	})
}
