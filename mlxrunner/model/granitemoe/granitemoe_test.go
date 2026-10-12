package granitemoe

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestConfigRealValues(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		raw := `{
			"hidden_size": 2048,
			"intermediate_size": 1536,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"num_hidden_layers": 40,
			"tie_word_embeddings": false,
			"vocab_size": 100352,
			"rms_norm_eps": 1e-05,
			"rope_theta": 30000000,
			"rope_scaling": null,
			"max_position_embeddings": 131072,
			"num_local_experts": 56,
			"num_experts_per_tok": 4,
			"attention_multiplier": 0.015625,
			"embedding_multiplier": 1.0,
			"logits_scaling": 1.0,
			"residual_multiplier": 1.0
		}`
		var cfg Config
		if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
			t.Fatalf("unmarshal config: %v", err)
		}

		// Defaults.
		if cfg.HiddenSize != 2048 {
			t.Errorf("hidden_size = %d, want 2048", cfg.HiddenSize)
		}
		if cfg.NumAttentionHeads != 32 {
			t.Errorf("num_attention_heads = %d, want 32", cfg.NumAttentionHeads)
		}
		if cfg.NumKeyValueHeads != 8 {
			t.Errorf("num_key_value_heads = %d, want 8", cfg.NumKeyValueHeads)
		}
		if cfg.NumLocalExperts != 56 {
			t.Errorf("num_local_experts = %d, want 56", cfg.NumLocalExperts)
		}
		if cfg.NumExpertsPerTok != 4 {
			t.Errorf("num_experts_per_tok = %d, want 4", cfg.NumExpertsPerTok)
		}
		if cfg.AttentionMultiplier != 0.015625 {
			t.Errorf("attention_multiplier = %f, want 0.015625", cfg.AttentionMultiplier)
		}
		if cfg.EmbeddingMultiplier != 1.0 {
			t.Errorf("embedding_multiplier = %f, want 1.0", cfg.EmbeddingMultiplier)
		}
		if cfg.ResidualMultiplier != 1.0 {
			t.Errorf("residual_multiplier = %f, want 1.0", cfg.ResidualMultiplier)
		}
		if cfg.LogitsScaling != 1.0 {
			t.Errorf("logits_scaling = %f, want 1.0", cfg.LogitsScaling)
		}

		// Computed: Scale = AttentionMultiplier.
		cfg.Scale = cfg.AttentionMultiplier
		if cfg.Scale != 0.015625 {
			t.Errorf("Scale = %f, want 0.015625", cfg.Scale)
		}
	})
}

func TestConfigDefaultExperts(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Config with num_local_experts and num_experts_per_tok omitted.
		raw := `{
			"hidden_size": 2048,
			"intermediate_size": 1536,
			"num_attention_heads": 32,
			"num_key_value_heads": 8,
			"num_hidden_layers": 40,
			"vocab_size": 100352,
			"rms_norm_eps": 1e-05,
			"rope_theta": 10000
		}`
		var cfg Config
		if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
			t.Fatalf("unmarshal config: %v", err)
		}

		// Apply the same defaults that newModel() applies.
		if cfg.NumLocalExperts <= 0 {
			cfg.NumLocalExperts = 8
		}
		if cfg.NumExpertsPerTok <= 0 {
			cfg.NumExpertsPerTok = 2
		}

		// Defaults should apply when fields are omitted.
		if cfg.NumLocalExperts != 8 {
			t.Errorf("NumLocalExperts default = %d, want 8", cfg.NumLocalExperts)
		}
		if cfg.NumExpertsPerTok != 2 {
			t.Errorf("NumExpertsPerTok default = %d, want 2", cfg.NumExpertsPerTok)
		}
	})
}

func TestSplitLastAxisHalves(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Create a [2, 3, 6] tensor and split along the last axis.
		// Row-major layout: [2][3][6] = 6 rows of 6 elements each.
		data := make([]float32, 2*3*6)
		for i := range data {
			data[i] = float32(i)
		}
		a := mlx.FromValues(data, 2, 3, 6)
		mlx.Eval(a)

		lo, hi := splitLastAxisHalves(a)
		mlx.Eval(lo, hi)

		// Floats() on a strided view returns memory-order data.
		// Make contiguous to get logical row-major order.
		loCont := mlx.Contiguous(lo, false)
		hiCont := mlx.Contiguous(hi, false)
		mlx.Eval(loCont, hiCont)

		loVals := loCont.Floats()
		hiVals := hiCont.Floats()

		// lo should be [2, 3, 3], hi should be [2, 3, 3].
		if len(loVals) != 18 {
			t.Errorf("lo len = %d, want 18", len(loVals))
		}
		if len(hiVals) != 18 {
			t.Errorf("hi len = %d, want 18", len(hiVals))
		}

		// Row-major [2,3,6]: lo slices [0:3] on last axis, hi slices [3:6].
		// Expected lo:  [0,1,2, 6,7,8, 12,13,14, 18,19,20, 24,25,26, 30,31,32]
		// Expected hi:  [3,4,5, 9,10,11, 15,16,17, 21,22,23, 27,28,29, 33,34,35]
		expectedLo := []float32{0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26, 30, 31, 32}
		expectedHi := []float32{3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29, 33, 34, 35}
		for i := 0; i < 18; i++ {
			if loVals[i] != expectedLo[i] {
				t.Errorf("lo[%d] = %f, want %f", i, loVals[i], expectedLo[i])
			}
			if hiVals[i] != expectedHi[i] {
				t.Errorf("hi[%d] = %f, want %f", i, hiVals[i], expectedHi[i])
			}
		}
	})
}

func TestFuseGateUpProjectionsGlobalScales(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const experts, rows, cols = 4, 16, 32
		weight := mlx.FromValues(make([]float32, experts*rows*cols), experts, rows, cols)
		mlx.Eval(weight)
		scales := mlx.FromValues(make([]float32, experts*rows), experts, rows)
		mlx.Eval(scales)

		// Shared global scale (per-expert bank) — fusion should succeed.
		globalScales := mlx.FromValues([]float32{1.0, 2.0, 3.0, 4.0}, experts)
		mlx.Eval(globalScales)

		gate := &stackedExpertWeights{
			Weight:       weight.Clone(),
			Scales:       scales.Clone(),
			GlobalScales: globalScales,
			Bits:         4,
			GroupSize:    64,
			Mode:         "nvfp4",
		}
		up := &stackedExpertWeights{
			Weight:       weight.Clone(),
			Scales:       scales.Clone(),
			GlobalScales: globalScales,
			Bits:         4,
			GroupSize:    64,
			Mode:         "nvfp4",
		}

		fused := fuseGateUpProjections(gate, up)
		if fused == nil {
			t.Fatal("fuseGateUpProjections returned nil for matching GlobalScales")
		}
		if fused.Weight == nil {
			t.Fatal("fused weight is nil")
		}
		if fused.GlobalScales == nil {
			t.Fatal("fused result has nil GlobalScales")
		}
		fusedWC := mlx.Contiguous(fused.Weight, false)
		mlx.Eval(fusedWC)
		if fusedWC.Dim(0) != experts || fusedWC.Dim(1) != rows*2 {
			t.Errorf("fused weight dims = %v, want [%d, %d, %d]", fusedWC.Dims(), experts, rows*2, cols)
		}
		if fused.Mode != "nvfp4" {
			t.Errorf("fused.Mode = %q, want %q (dropped Mode makes gather_qmm fail with \"invalid quantization mode ''\")", fused.Mode, "nvfp4")
		}
		if fused.GroupSize != 64 {
			t.Errorf("fused.GroupSize = %d, want 64", fused.GroupSize)
		}
		if fused.Bits != 4 {
			t.Errorf("fused.Bits = %d, want 4", fused.Bits)
		}

		// Mismatched global scales — fusion should be refused (nil).
		wrongGlobalScales := mlx.FromValues([]float32{4.0, 3.0, 2.0, 1.0}, experts)
		mlx.Eval(wrongGlobalScales)

		up2 := &stackedExpertWeights{
			Weight:       weight.Clone(),
			Scales:       scales.Clone(),
			GlobalScales: wrongGlobalScales,
			Bits:         4,
			GroupSize:    64,
			Mode:         "nvfp4",
		}

		refused := fuseGateUpProjections(gate, up2)
		if refused != nil {
			t.Fatal("fuseGateUpProjections should return nil for mismatched GlobalScales")
		}
	})
}
