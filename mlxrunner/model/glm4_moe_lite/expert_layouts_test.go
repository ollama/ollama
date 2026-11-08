package glm4_moe_lite

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx/mlxtest"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

// TestExpertLayouts pins how LoadWeights resolves the routed-expert layouts
// the importer can produce for GLM4-MoE-Lite, so a loader migration shows its
// intent as a diff. No published MLX artifact of this architecture exists.
func TestExpertLayouts(t *testing.T) {
	const (
		experts = 4
		hidden  = 64
		inter   = 128
		layer   = "model.layers.0"
		group   = layer + ".mlp.experts"
	)
	stacked := func(s *fx.Set, base, suffix string, gate, up, down fx.Encoding) {
		s.Weight(base+".gate_proj"+suffix, gate, experts, inter, hidden)
		s.Weight(base+".up_proj"+suffix, up, experts, inter, hidden)
		s.Weight(base+".down_proj"+suffix, down, experts, hidden, inter)
	}
	perExpert := func(s *fx.Set, enc fx.Encoding, gs []float32) {
		s.PerExpert(group, "gate_proj", enc, experts, inter, hidden, gs)
		s.PerExpert(group, "up_proj", enc, experts, inter, hidden, gs)
		s.PerExpert(group, "down_proj", enc, experts, hidden, inter, gs)
	}
	gs := []float32{0.5, 1, 2, 4}

	bf16 := map[string]string{
		"gate": "dense BF16[4 128 64]",
		"up":   "dense BF16[4 128 64]",
		"down": "dense BF16[4 64 128]",
	}
	nvfp4Scaled := map[string]string{
		"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
		"up":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
		"down": "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
	}
	int4 := map[string]string{
		"gate": "affine/64/4 U32[4 128 8] s=BF16[4 128 1] qb=BF16[4 128 1]",
		"up":   "affine/64/4 U32[4 128 8] s=BF16[4 128 1] qb=BF16[4 128 1]",
		"down": "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
	}

	for _, tc := range []struct {
		name     string
		defaults fx.Encoding
		build    func(*fx.Set)
		err      string
		want     map[string]string
	}{
		{
			name:     "L3 bf16",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.BF16, fx.BF16, fx.BF16) },
			want:     bf16,
		},
		{
			name:     "L3 nvfp4 with global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				stacked(s, group, ".weight", fx.NVFP4, fx.NVFP4, fx.NVFP4)
				for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
					s.GlobalScale(group+"."+p+".weight", gs...)
				}
			},
			want: nvfp4Scaled,
		},
		{
			name:     "L3 mxfp8",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8) },
			want: map[string]string{
				"gate": "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"up":   "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"down": "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L3 int4",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.Int4, fx.Int4, fx.Int4) },
			want:     int4,
		},
		{
			name:     "era A switch_mlp int4",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { stacked(s, layer+".mlp.switch_mlp", ".weight", fx.Int4, fx.Int4, fx.Int4) },
			want:     int4,
		},
		{
			name:     "experts wins over switch_mlp",
			defaults: fx.Int4,
			build: func(s *fx.Set) {
				stacked(s, group, ".weight", fx.BF16, fx.BF16, fx.BF16)
				stacked(s, layer+".mlp.switch_mlp", ".weight", fx.Int4, fx.Int4, fx.Int4)
			},
			want: bf16,
		},
		{
			// All three projections must be quantized; otherwise every bank is dequantized.
			name:     "L3 nvfp4 gate+up with bf16 down",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.NVFP4, fx.NVFP4, fx.BF16) },
			want:     bf16,
		},
		{
			name:     "L4 nvfp4 per-expert with scalar global scales",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { perExpert(s, fx.NVFP4, gs) },
			want:     nvfp4Scaled,
		},
		{
			name:     "L4 bf16 per-expert",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { perExpert(s, fx.BF16, nil) },
			want:     bf16,
		},
		{
			name:     "L4 per-expert with one global scale missing",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				perExpert(s, fx.NVFP4, gs)
				s.Delete(group + ".3.gate_proj.weight.global_scale")
			},
			err: "3 of 4 experts have global scales",
		},
		{
			name:     "bare stacked bf16",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { stacked(s, group, "", fx.BF16, fx.BF16, fx.BF16) },
			want:     bf16,
		},
		{
			// Each projection is resolved on its own, so a mixed layout loads.
			name:     "stacked gate+up with per-expert down",
			defaults: fx.BF16,
			build: func(s *fx.Set) {
				s.Weight(group+".gate_proj.weight", fx.BF16, experts, inter, hidden)
				s.Weight(group+".up_proj.weight", fx.BF16, experts, inter, hidden)
				s.PerExpert(group, "down_proj", fx.BF16, experts, hidden, inter, nil)
			},
			want: bf16,
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			s.Weight(layer+".mlp.gate.weight", fx.BF16, experts, hidden)
			s.Weight(layer+".mlp.shared_experts.gate_proj.weight", fx.BF16, inter, hidden)
			s.Weight(layer+".mlp.shared_experts.up_proj.weight", fx.BF16, inter, hidden)
			s.Weight(layer+".mlp.shared_experts.down_proj.weight", fx.BF16, hidden, inter)
			tc.build(s)

			cfg := &Config{
				HiddenSize: hidden, NumHiddenLayers: 1, MoEIntermediateSize: inter,
				NRoutedExperts: experts, NSharedExperts: 1, NumExpertsPerTok: 2,
				RMSNormEps: 1e-5, TensorQuant: s.Quant(),
			}
			cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = fx.Defaults(tc.defaults)
			m := &Model{Config: cfg, Layers: make([]Block, 1)}
			err := m.LoadWeights(s.Tensors())
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("LoadWeights error = %v, want %q", err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatalf("LoadWeights: %v", err)
			}
			block, ok := m.Layers[0].(*MoEBlock)
			if !ok {
				t.Fatalf("layer 0 = %T, want *MoEBlock", m.Layers[0])
			}
			for _, d := range fx.Compare(glm4Banks(block.MoE.SwitchMLP), tc.want) {
				t.Error(d)
			}
		})
	}
}

// glm4Banks describes the banks SwitchMLP.Forward reads; dense banks stay in
// [experts, out, in] and are transposed per call.
func glm4Banks(s *SwitchMLP) map[string]string {
	if !s.UseQuantized {
		return map[string]string{
			"gate": fx.Bank{Weight: s.GateWeight}.String(),
			"up":   fx.Bank{Weight: s.UpWeight}.String(),
			"down": fx.Bank{Weight: s.DownWeight}.String(),
		}
	}
	return map[string]string{
		"gate": fx.Bank{
			Weight: s.GateWeightQ, Scales: s.GateScales, Biases: s.GateBiases, GlobalScale: s.GateGlobalScales,
			GroupSize: s.GateGroupSize, Bits: s.GateBits, Mode: s.GateMode,
		}.String(),
		"up": fx.Bank{
			Weight: s.UpWeightQ, Scales: s.UpScales, Biases: s.UpBiases, GlobalScale: s.UpGlobalScales,
			GroupSize: s.UpGroupSize, Bits: s.UpBits, Mode: s.UpMode,
		}.String(),
		"down": fx.Bank{
			Weight: s.DownWeightQ, Scales: s.DownScales, Biases: s.DownBiases, GlobalScale: s.DownGlobalScales,
			GroupSize: s.DownGroupSize, Bits: s.DownBits, Mode: s.DownMode,
		}.String(),
	}
}
