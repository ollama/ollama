package model

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestLoadExpertProjectionValidatesPackedShapes(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const key = "experts.down_proj.weight"
		tensors := map[string]*mlx.Array{key: mlx.Zeros(mlx.DTypeFloat32, 8, 16)}
		if _, err := LoadExpertProjection(tensors, "experts.down_proj", 16, 4, "nvfp4", nil); err == nil || !strings.Contains(err.Error(), "nonempty") {
			t.Fatalf("rank-two weights returned %v", err)
		}

		tensors[key] = mlx.Zeros(mlx.DTypeUint32, 2, 8, 2)
		if _, err := LoadExpertProjection(tensors, "experts.down_proj", 16, 4, "nvfp4", nil); err == nil || !strings.Contains(err.Error(), "missing its quantization scales") {
			t.Fatalf("missing scales returned %v", err)
		}

		tensors[key+"_scale"] = mlx.Zeros(mlx.DTypeUint8, 2, 8, 2)
		if _, err := LoadExpertProjection(tensors, "experts.down_proj", 16, 4, "nvfp4", nil); err == nil || !strings.Contains(err.Error(), "incompatible") {
			t.Fatalf("wrong scale width returned %v", err)
		}

		tensors[key+"_scale"] = mlx.Zeros(mlx.DTypeUint8, 2, 8, 1)
		tensors[key+".global_scale"] = mlx.FromValue(float32(2))
		tensors["experts.down_proj.global_scale"] = mlx.FromValue(float32(3))
		p, err := LoadExpertProjection(tensors, "experts.down_proj", 16, 4, "nvfp4", nil)
		if err != nil {
			t.Fatal(err)
		}
		for _, got := range p.GlobalScale.Floats() {
			if want := float32(2 * mlx.Nvfp4MaxProduct); got != want {
				t.Fatalf("global scale = %g, want %g", got, want)
			}
		}
	})
}
