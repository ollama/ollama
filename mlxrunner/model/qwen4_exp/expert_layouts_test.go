package qwen4_exp

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

// TestExpertLayouts pins how loadSparseMoE resolves every routed-expert
// layout a Qwen 4 experimental package can carry, so a loader migration shows
// its intent as a diff.
func TestExpertLayouts(t *testing.T) {
	const (
		experts = 4
		hidden  = 64
		inter   = 128
		prefix  = "model.language_model.layers.0.mlp"
		group   = prefix + ".experts"
	)
	fused := func(s *fx.Set, suffix string, gateUp, down fx.Encoding) {
		s.Weight(group+".gate_up_proj"+suffix, gateUp, experts, 2*inter, hidden)
		s.Weight(group+".down_proj"+suffix, down, experts, hidden, inter)
	}
	scale := func(s *fx.Set, suffix string) {
		s.GlobalScale(group+".gate_up_proj"+suffix, 0.5, 1, 2, 4)
		s.GlobalScale(group+".down_proj"+suffix, 0.5, 1, 2, 4)
	}

	nvfp4 := map[string]string{
		"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4]",
		"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8]",
	}
	nvfp4Scaled := map[string]string{
		"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4] gs=F32[4]",
		"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
	}
	bf16 := map[string]string{
		"gate_up": "dense BF16[4 64 256]",
		"down":    "dense BF16[4 128 64]",
	}

	for _, tc := range []struct {
		name     string
		defaults fx.Encoding
		build    func(*fx.Set)
		err      string
		want     map[string]string
	}{
		{
			name:     "L1 bf16 bare",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { fused(s, "", fx.BF16, fx.BF16) },
			want:     bf16,
		},
		{
			name:     "L1 nvfp4 bare",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { fused(s, "", fx.NVFP4, fx.NVFP4) },
			want:     nvfp4,
		},
		{
			name:     "L1 nvfp4 bare with global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, "", fx.NVFP4, fx.NVFP4)
				scale(s, "")
			},
			want: nvfp4Scaled,
		},
		{
			name:     "L1 mxfp8 bare",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { fused(s, "", fx.MXFP8, fx.MXFP8) },
			want: map[string]string{
				"gate_up": "mxfp8/32/8 U32[4 256 16] s=U8[4 256 2]",
				"down":    "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L1 int4 bare",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { fused(s, "", fx.Int4, fx.Int4) },
			want: map[string]string{
				"gate_up": "affine/64/4 U32[4 256 8] s=BF16[4 256 1] qb=BF16[4 256 1]",
				"down":    "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
			},
		},
		{
			name:     "L2 nvfp4 with global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, ".weight", fx.NVFP4, fx.NVFP4)
				scale(s, ".weight")
			},
			want: nvfp4Scaled,
		},
		{
			name:     "L2 nvfp4 with scalar ModelOpt scale_2",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, ".weight", fx.NVFP4, fx.NVFP4)
				s.Raw(group+".gate_up_proj.weight_scale_2", mlx.FromValue(float32(0.5)))
				s.Raw(group+".down_proj.weight_scale_2", mlx.FromValue(float32(2)))
			},
			want: nvfp4Scaled,
		},
		{
			// No package ships both spellings; the shared reader prefers ".weight".
			name:     ".weight wins over bare name",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, "", fx.BF16, fx.BF16)
				fused(s, ".weight", fx.NVFP4, fx.NVFP4)
			},
			want: nvfp4,
		},
		{
			// An FP8 checkpoint stored exactly carries its scale as a global scale.
			name:     "mxfp8 bank with a global scale",
			defaults: fx.MXFP8,
			build: func(s *fx.Set) {
				fused(s, "", fx.MXFP8, fx.MXFP8)
				scale(s, "")
			},
			want: map[string]string{
				"gate_up": "mxfp8/32/8 U32[4 256 16] s=U8[4 256 2] gs=F32[4]",
				"down":    "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4] gs=F32[4]",
			},
		},
		{
			name:     "int4 bank with a global scale",
			defaults: fx.Int4,
			build: func(s *fx.Set) {
				fused(s, "", fx.Int4, fx.Int4)
				scale(s, "")
			},
			err: `has a global scale but uses mode="affine"`,
		},
		{
			name:     "bf16 bank with a global scale",
			defaults: fx.BF16,
			build: func(s *fx.Set) {
				fused(s, "", fx.BF16, fx.BF16)
				scale(s, "")
			},
			err: `has a global scale but uses mode=""`,
		},
		{
			// Separate banks are fused at load into the gate_up bank the forward runs.
			name:     "L3 separate banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				s.Weight(group+".gate_proj.weight", fx.NVFP4, experts, inter, hidden)
				s.Weight(group+".up_proj.weight", fx.NVFP4, experts, inter, hidden)
				s.Weight(group+".down_proj.weight", fx.NVFP4, experts, hidden, inter)
			},
			want: nvfp4,
		},
		{
			name:     "L4 per-expert",
			defaults: fx.BF16,
			build: func(s *fx.Set) {
				s.PerExpert(group, "gate_proj", fx.BF16, experts, inter, hidden, nil)
				s.PerExpert(group, "up_proj", fx.BF16, experts, inter, hidden, nil)
				s.PerExpert(group, "down_proj", fx.BF16, experts, hidden, inter, nil)
			},
			want: bf16,
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			s.Weight(prefix+".gate.weight", fx.BF16, experts, hidden)
			s.Weight(prefix+".shared_expert_gate.weight", fx.BF16, 1, hidden)
			s.Weight(prefix+".shared_expert.gate_proj.weight", fx.BF16, inter, hidden)
			s.Weight(prefix+".shared_expert.up_proj.weight", fx.BF16, inter, hidden)
			s.Weight(prefix+".shared_expert.down_proj.weight", fx.BF16, hidden, inter)
			tc.build(s)

			m := &Model{Config: &Config{HiddenSize: hidden, NumExperts: experts, NumExpertsPerTok: 2}}
			m.quantGroup, m.quantBits, m.quantMode = fx.Defaults(tc.defaults)
			m.tensorQuant = s.Quant()
			tensors := s.Tensors()
			linears := model.NewLinearFactory(tensors, m.quantGroup, m.quantBits, m.quantMode, m.tensorQuant)
			moe, err := m.loadSparseMoE(linears, tensors, prefix)
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("loadSparseMoE error = %v, want %q", err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatalf("loadSparseMoE: %v", err)
			}
			got := map[string]string{
				"gate_up": qwen4Bank(moe.GateUp).String(),
				"down":    qwen4Bank(moe.Down).String(),
			}
			for _, d := range fx.Compare(got, tc.want) {
				t.Error(d)
			}
		})
	}
}

func qwen4Bank(p *expertProjection) fx.Bank {
	return fx.Bank{
		Weight: p.Weight, Scales: p.Scales, Biases: p.Biases, GlobalScale: p.GlobalScale,
		GroupSize: p.GroupSize, Bits: p.Bits, Mode: p.Mode,
	}
}
