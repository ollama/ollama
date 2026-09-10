package glm4_moe_lite

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestConfigNestedRopeTheta(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(`{"rope_parameters":{"rope_theta":1000000}}`), &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.normalize()
	if cfg.RopeTheta != 1000000 {
		t.Fatalf("rope theta = %v, want 1000000", cfg.RopeTheta)
	}
}

func TestSanitizeMLAWeightsGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const heads, headDim, latentDim, groupSize = 2, 8, 16, 16
		packed := make([]uint32, heads*headDim*latentDim/8)
		for i := range packed {
			for j := range 8 {
				packed[i] |= uint32((i*8+j)%15+1) << (4 * j)
			}
		}
		blockScales := make([]uint8, heads*headDim*latentDim/groupSize)
		for i := range blockScales {
			blockScales[i] = 0x38
		}

		weight := mlx.FromValues(packed, heads*headDim, latentDim/8)
		scales := mlx.FromValues(blockScales, heads*headDim, latentDim/groupSize)
		globalScale := mlx.FromValues([]float32{0.5}, 1)
		cfg := &Config{
			NumAttentionHeads: heads,
			QKNopeHeadDim:     headDim / 2,
			VHeadDim:          headDim / 2,
			KVLoraRank:        latentDim,
			QuantGroupSize:    groupSize,
			QuantBits:         4,
			QuantMode:         "nvfp4",
		}

		gotEmbed, gotUnembed := sanitizeMLAWeights(map[string]*mlx.Array{
			"layer.self_attn.kv_b_proj.weight":              weight,
			"layer.self_attn.kv_b_proj.weight_scale":        scales,
			"layer.self_attn.kv_b_proj.weight.global_scale": globalScale,
		}, "layer", cfg)

		unscaledEmbed, unscaledUnembed := sanitizeMLAWeights(map[string]*mlx.Array{
			"layer.self_attn.kv_b_proj.weight":       weight,
			"layer.self_attn.kv_b_proj.weight_scale": scales,
		}, "layer", cfg)
		gotEmbed = gotEmbed.AsType(mlx.DTypeFloat32)
		gotUnembed = gotUnembed.AsType(mlx.DTypeFloat32)
		wantEmbed := mlx.MulScalar(unscaledEmbed.AsType(mlx.DTypeFloat32), 0.5)
		wantUnembed := mlx.MulScalar(unscaledUnembed.AsType(mlx.DTypeFloat32), 0.5)
		mlx.Eval(gotEmbed, gotUnembed, wantEmbed, wantUnembed)

		for _, pair := range [][2][]float32{
			{gotEmbed.Floats(), wantEmbed.Floats()},
			{gotUnembed.Floats(), wantUnembed.Floats()},
		} {
			for i := range pair[0] {
				if math.Abs(float64(pair[0][i]-pair[1][i])) > 1e-6 {
					t.Fatalf("value[%d] = %v, want %v", i, pair[0][i], pair[1][i])
				}
			}
		}
	})
}

func TestConfigTopLevelRopeThetaTakesPrecedence(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(`{"rope_theta":10000,"rope_parameters":{"rope_theta":1000000}}`), &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.normalize()
	if cfg.RopeTheta != 10000 {
		t.Fatalf("rope theta = %v, want 10000", cfg.RopeTheta)
	}
}
