package deepseek_v41

import (
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestVisionTowerOddPatchGrid(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		zeros := func(shape ...int) *mlx.Array { return mlx.Zeros(mlx.DTypeFloat32, shape...) }
		ones := func(n int) *mlx.Array {
			values := make([]float32, n)
			for i := range values {
				values[i] = 1
			}
			return mlx.FromValues(values, n)
		}
		tensors := map[string]*mlx.Array{
			"vision.patch_embed.proj.weight": zeros(8, 3),
			"vision.patch_embed.proj.bias":   zeros(8),
			"vision.norm.weight":             ones(8),
			"aligner.w1.weight":              zeros(4, 32),
			"aligner.w1.bias":                zeros(4),
			"aligner.w2.weight":              zeros(4, 4),
			"aligner.w2.bias":                zeros(4),
			"image_start":                    zeros(4), "image_end": zeros(4), "image_newline": zeros(4),
		}
		prefix := "vision.blocks.0"
		for _, name := range []string{"norm1", "norm2"} {
			tensors[prefix+"."+name+".weight"] = ones(8)
		}
		for name, shape := range map[string][]int{
			"attn.wqkv.weight": {24, 8}, "attn.wqkv.bias": {24},
			"attn.wo.weight": {8, 8}, "attn.wo.bias": {8},
			"mlp.w1.weight": {16, 8}, "mlp.w2.weight": {8, 8},
		} {
			tensors[fmt.Sprintf("%s.%s", prefix, name)] = zeros(shape...)
		}
		cfg := &Config{HiddenSize: 4, Vision: visionConfig{ModelType: "deepseek_v41_vision", NumHiddenLayers: 1, HiddenSize: 8, NumAttentionHeads: 1, IntermediateSize: 8, PatchSize: 1, RoPETheta: 10000, DownsampleRatio: 2}}
		tower, err := loadVision(tensors, cfg)
		if err != nil {
			t.Fatal(err)
		}
		out := tower.forward(zeros(9, 3), imageGeometry{patchRows: 3, patchCols: 3, textRows: 2, textCols: 2})
		if got := out.Dims(); len(got) != 2 || got[0] != 4 || got[1] != 4 {
			t.Fatalf("vision output shape = %v", got)
		}
		mlx.Eval(out)
		for _, value := range out.Floats() {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("nonfinite vision output: %v", value)
			}
		}
	})
}
