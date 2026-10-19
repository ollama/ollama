package qwen3

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestConfigureRope(t *testing.T) {
	const headDim = 128
	baseScale := float32(1 / math.Sqrt(headDim))
	// mlx-lm's YarnRoPE for head_dim 128, rope_theta 1e6, factor 4 and
	// original_max_position_embeddings 8192, as Prism's Bonsai 1.7B checkpoints
	// configure it.
	const yarn = `"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192`
	yarnFreqs := map[int]float64{0: 1, 10: 8.659643, 20: 86.42848, 25: 341.04068, 30: 1522.6879, 35: 7643.8125, 40: 22493.652, 63: 3223368.8}
	yarnScale := baseScale * float32(1.138629436111989*1.138629436111989)

	for _, tt := range []struct {
		name      string
		config    string
		wantTheta float32
		wantYarn  bool
		wantScale float32
	}{
		{"default", `{"rope_theta": 5000000}`, 5000000, false, baseScale},
		{"rope_scaling yarn", `{"rope_theta": 1000000, "rope_scaling": {` + yarn + `}}`, 1000000, true, yarnScale},
		{"legacy type key", `{"rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192}}`, 1000000, true, yarnScale},
		{"rope_parameters yarn", `{"rope_parameters": {"rope_theta": 1000000, ` + yarn + `}}`, 1000000, true, yarnScale},
		{"rope_parameters default", `{"rope_parameters": {"rope_type": "default", "rope_theta": 5000000}}`, 5000000, false, baseScale},
		{"top-level rope_theta wins", `{"rope_theta": 2000000, "rope_parameters": {"rope_theta": 5000000}}`, 2000000, false, baseScale},
		{"rope_scaling wins", `{"rope_scaling": {"rope_type": "default"}, "rope_parameters": {` + yarn + `}}`, 1000000, false, baseScale},
		{"explicit attention_factor", `{"rope_scaling": {` + yarn + `, "attention_factor": 1.0}}`, 1000000, true, baseScale},
		{"mscale ratio", `{"rope_scaling": {` + yarn + `, "mscale": 1.0, "mscale_all_dim": 1.0}}`, 1000000, true, baseScale},
		// DeepSeek-R1-0528-Qwen3-8B sets attn_factor to cancel the mscale.
		{"attn_factor canceling the mscale", `{"rope_scaling": {` + yarn + `, "attn_factor": 0.8782488562869419}}`, 1000000, true, baseScale},
		{"attn_factor scaling the mscale", `{"rope_scaling": {` + yarn + `, "attn_factor": 0.5}}`, 1000000, true, baseScale * float32(0.5*1.138629436111989*0.5*1.138629436111989)},
		{"attn_factor scaling the mscale ratio", `{"rope_scaling": {` + yarn + `, "mscale": 1.0, "mscale_all_dim": 0.5, "attn_factor": 0.5}}`, 1000000, true, baseScale * float32(0.5*1.138629436111989/1.0693147180559945*0.5*1.138629436111989/1.0693147180559945)},
		{"explicit attention_factor ignores attn_factor", `{"rope_scaling": {` + yarn + `, "attention_factor": 1.0, "attn_factor": 0.8782488562869419}}`, 1000000, true, baseScale},
		{"yarn without factor", `{"rope_scaling": {"rope_type": "yarn", "original_max_position_embeddings": 8192}}`, 1000000, false, baseScale},
		{"unsupported scaling", `{"rope_scaling": {"rope_type": "linear", "factor": 2.0}}`, 1000000, false, baseScale},
	} {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			cfg := Config{HeadDim: headDim}
			if err := json.Unmarshal([]byte(tt.config), &cfg); err != nil {
				t.Fatal(err)
			}
			cfg.configureRope()

			if cfg.RopeTheta != tt.wantTheta {
				t.Errorf("RopeTheta = %v, want %v", cfg.RopeTheta, tt.wantTheta)
			}
			if diff := math.Abs(float64(cfg.Scale-tt.wantScale)) / float64(tt.wantScale); diff > 1e-6 {
				t.Errorf("Scale = %v, want %v", cfg.Scale, tt.wantScale)
			}
			if !tt.wantYarn {
				if cfg.RopeFreqs != nil {
					t.Fatalf("RopeFreqs = %v, want default RoPE", cfg.RopeFreqs)
				}
				return
			}

			if cfg.RopeFreqs == nil {
				t.Fatal("RopeFreqs not set for YaRN")
			}
			freqs := cfg.RopeFreqs.Floats()
			if len(freqs) != headDim/2 {
				t.Fatalf("len(RopeFreqs) = %d, want %d", len(freqs), headDim/2)
			}
			for i, want := range yarnFreqs {
				if diff := math.Abs(float64(freqs[i])-want) / want; diff > 1e-5 {
					t.Errorf("RopeFreqs[%d] = %v, want %v as mlx-lm computes it", i, freqs[i], want)
				}
			}
		})
	}
}
