package deepseek_v41

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestProjectionUsesPublishedMXFP8ScaleCompanion(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		values := make([]float32, 2*32)
		for i := range values {
			values[i] = .25
		}
		packed, scales, _ := mlx.Quantize(mlx.FromValues(values, 2, 32), 32, 8, "mxfp8")
		linear, err := makeProjection(map[string]*mlx.Array{"test.weight": packed, "test.weight_scale": scales},
			map[string]*model.TensorQuantInfo{"test.weight": {QuantType: "MXFP8", GroupSize: 32}}, "test")
		if err != nil {
			t.Fatal(err)
		}
		result := linear.Forward(mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 32), 1))
		mlx.Eval(result)
		compareFloats(t, "projection", result.Floats(), []float32{8, 8}, .02)
	})
}
