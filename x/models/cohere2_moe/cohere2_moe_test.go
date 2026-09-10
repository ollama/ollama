package cohere2_moe

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
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

		got := loadStackedExperts(tensors, cfg, true, "model.layers.1", "gate_proj")
		if got == nil {
			t.Fatal("loadStackedExperts returned nil")
		}
		if !mlx.MetalIsAvailable() {
			if got.Scales != nil {
				t.Fatal("per-expert weights did not use the dense fallback off Metal")
			}
		} else if got.Scales == nil || got.GlobalScales == nil {
			t.Fatal("per-expert weights did not use the quantized global-scale path")
		} else if dims := got.GlobalScales.Dims(); len(dims) != 1 || dims[0] != experts {
			t.Fatalf("global scale dims = %v, want [%d]", dims, experts)
		}
		if dims := got.Weight.Dims(); len(dims) != 3 || dims[0] != experts {
			t.Fatalf("weight dims = %v, want expert dimension %d", dims, experts)
		}

		delete(tensors, "model.layers.1.mlp.experts.3.gate_proj.weight.global_scale")
		if got := loadStackedExperts(tensors, cfg, true, "model.layers.1", "gate_proj"); got != nil {
			t.Fatal("loadStackedExperts accepted an incomplete global-scale set")
		}
	})
}
