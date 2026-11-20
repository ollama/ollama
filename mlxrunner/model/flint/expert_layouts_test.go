package flint

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/model"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

// TestExpertLayouts pins how loadSparseMoE treats every routed-expert layout
// a Flint package can carry, so a loader migration shows its intent as a diff.
func TestExpertLayouts(t *testing.T) {
	const (
		experts = 4
		hidden  = 64
		inter   = 128
		layer   = "model.layers.0"
		group   = layer + ".mlp.experts"
	)
	stacked := func(s *fx.Set, enc fx.Encoding, projs ...string) {
		for _, p := range projs {
			out, in := inter, hidden
			if p == "down_proj" {
				out, in = hidden, inter
			}
			s.Weight(group+"."+p+".weight", enc, experts, out, in)
		}
	}
	perExpert := func(s *fx.Set, enc fx.Encoding, gs []float32) {
		s.PerExpert(group, "gate_proj", enc, experts, inter, hidden, gs)
		s.PerExpert(group, "up_proj", enc, experts, inter, hidden, gs)
		s.PerExpert(group, "down_proj", enc, experts, hidden, inter, gs)
	}
	perExpertGS := []float32{0.5, 1, 2, 4}

	bf16 := map[string]string{
		"gate": "dense BF16[4 128 64]",
		"up":   "dense BF16[4 128 64]",
		"down": "dense BF16[4 64 128]",
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
			build:    func(s *fx.Set) { stacked(s, fx.BF16, "gate_proj", "up_proj", "down_proj") },
			want:     bf16,
		},
		{
			name:     "L3 int4 with qbias",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { stacked(s, fx.Int4, "gate_proj", "up_proj", "down_proj") },
			want: map[string]string{
				"gate": "affine/64/4 U32[4 128 8] s=BF16[4 128 1] qb=BF16[4 128 1]",
				"up":   "affine/64/4 U32[4 128 8] s=BF16[4 128 1] qb=BF16[4 128 1]",
				"down": "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
			},
		},
		{
			name:     "L3 mxfp8",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { stacked(s, fx.MXFP8, "gate_proj", "up_proj", "down_proj") },
			want: map[string]string{
				"gate": "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"up":   "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"down": "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L3 nvfp4 with per-expert global scale bank",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				stacked(s, fx.NVFP4, "gate_proj", "up_proj", "down_proj")
				for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
					s.GlobalScale(group+"."+p+".weight", perExpertGS...)
				}
			},
			want: map[string]string{
				"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"up":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down": "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
			},
		},
		{
			name:     "L3 nvfp4 with scalar global scale",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				stacked(s, fx.NVFP4, "gate_proj", "up_proj", "down_proj")
				for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
					s.ScalarGlobalScale(group+"."+p+".weight", 0.5)
				}
			},
			want: map[string]string{
				"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"up":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down": "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
			},
		},
		{
			name:     "L3 scale expert count mismatch",
			defaults: fx.MXFP8,
			build: func(s *fx.Set) {
				stacked(s, fx.MXFP8, "gate_proj", "up_proj", "down_proj")
				s.Raw(group+".up_proj.weight.scale", mlx.Zeros(mlx.DTypeUint8, experts-1, inter, hidden/32))
			},
			err: "unsupported expert quantization",
		},
		{
			name:     "L3 nvfp4 without global scale",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { stacked(s, fx.NVFP4, "gate_proj", "up_proj", "down_proj") },
			err:      "missing NVFP4 global scale",
		},
		{
			name:     "L3 nvfp4 gate+up with bf16 down",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				stacked(s, fx.NVFP4, "gate_proj", "up_proj")
				stacked(s, fx.BF16, "down_proj")
				s.GlobalScale(group+".gate_proj.weight", perExpertGS...)
				s.GlobalScale(group+".up_proj.weight", perExpertGS...)
			},
			want: map[string]string{
				"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"up":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down": "dense BF16[4 64 128]",
			},
		},
		{
			name:     "L3 f32 dense rejected",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { stacked(s, fx.F32, "gate_proj", "up_proj", "down_proj") },
			err:      "invalid unquantized expert shape or dtype",
		},
		{
			name:     "L3 missing down projection",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { stacked(s, fx.BF16, "gate_proj", "up_proj") },
			err:      "missing " + group + ".down_proj",
		},
		{
			name:     "L4 per-expert bf16",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { perExpert(s, fx.BF16, nil) },
			want:     bf16,
		},
		{
			// Per-expert quantized experts are stacked at load like the stacked form.
			name:     "L4 per-expert nvfp4",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { perExpert(s, fx.NVFP4, perExpertGS) },
			want: map[string]string{
				"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"up":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down": "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
			},
		},
		{
			name:     "bare stacked names",
			defaults: fx.BF16,
			build: func(s *fx.Set) {
				s.Weight(group+".gate_proj", fx.BF16, experts, inter, hidden)
				s.Weight(group+".up_proj", fx.BF16, experts, inter, hidden)
				s.Weight(group+".down_proj", fx.BF16, experts, hidden, inter)
			},
			want: bf16,
		},
		{
			name:     "L5 switch_mlp",
			defaults: fx.BF16,
			build: func(s *fx.Set) {
				for _, p := range []string{"gate_proj", "up_proj"} {
					s.Weight(layer+".mlp.switch_mlp."+p+".weight", fx.BF16, experts, inter, hidden)
				}
				s.Weight(layer+".mlp.switch_mlp.down_proj.weight", fx.BF16, experts, hidden, inter)
			},
			err: "missing " + group + ".gate_proj",
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			s.Weight(layer+".mlp.gate.weight", fx.BF16, experts, hidden)
			s.Weight(layer+".mlp.shared_expert.gate_proj.weight", fx.BF16, inter, hidden)
			s.Weight(layer+".mlp.shared_expert.up_proj.weight", fx.BF16, inter, hidden)
			s.Weight(layer+".mlp.shared_expert.down_proj.weight", fx.BF16, hidden, inter)
			tc.build(s)

			groupSize, bits, mode := fx.Defaults(tc.defaults)
			cfg := &Config{
				HiddenSize: hidden, MoeIntermediateSize: inter, SharedExpertIntermediateSize: inter,
				NumExperts: experts, NumExpertsPerTok: 2,
				QuantGroupSize: groupSize, QuantBits: bits, QuantMode: mode, TensorQuant: s.Quant(),
			}
			tensors := s.Tensors()
			moe, err := loadSparseMoE(model.NewLinearFactory(tensors, groupSize, bits, mode, cfg.TensorQuant), tensors, layer, cfg)
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("loadSparseMoE error = %v, want %q", err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatalf("loadSparseMoE: %v", err)
			}
			for _, d := range fx.Compare(flintBanks(moe.Experts), tc.want) {
				t.Error(d)
			}
		})
	}
}

func flintBanks(e *Experts) map[string]string {
	out := make(map[string]string)
	for name, p := range map[string]*ExpertProjection{"gate": e.Gate, "up": e.Up, "down": e.Down} {
		out[name] = fx.Bank{
			Weight: p.Weight, Scales: p.Scales, Biases: p.Biases, GlobalScale: p.GlobalScale,
			GroupSize: p.GroupSize, Bits: p.Bits, Mode: p.Mode,
		}.String()
	}
	return out
}
