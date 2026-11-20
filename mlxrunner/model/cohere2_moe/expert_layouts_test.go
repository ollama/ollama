package cohere2_moe

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx/mlxtest"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

// TestExpertLayouts pins how LoadWeights resolves every routed-expert layout
// a Cohere2 MoE package can carry, so a loader migration shows its intent as
// a diff.
func TestExpertLayouts(t *testing.T) {
	const (
		experts = 4
		hidden  = 64
		inter   = 128
		layer   = "model.layers.0"
		group   = layer + ".mlp.experts"
		switchG = layer + ".mlp.switch_mlp"
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
		"gate":   "dense BF16[4 64 128]",
		"up":     "dense BF16[4 64 128]",
		"down":   "dense BF16[4 128 64]",
		"shared": "*nn.Linear",
	}
	mxfp8 := map[string]string{
		"gate":   "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
		"up":     "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
		"down":   "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
		"shared": "*nn.Linear",
	}
	nvfp4Scaled := map[string]string{
		"gate":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
		"up":     "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
		"down":   "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
		"shared": "*nn.Linear",
	}

	for _, tc := range []struct {
		name          string
		defaults      fx.Encoding
		singularShare bool
		build         func(*fx.Set)
		err           string
		want          map[string]string
	}{
		{
			name:     "L5 switch_mlp bf16",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { stacked(s, switchG, ".weight", fx.BF16, fx.BF16, fx.BF16) },
			want:     bf16,
		},
		{
			name:     "L5 switch_mlp mxfp8",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { stacked(s, switchG, ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8) },
			want:     mxfp8,
		},
		{
			name:     "L5 switch_mlp nvfp4 gate+up with mxfp8 down",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { stacked(s, switchG, ".weight", fx.NVFP4, fx.NVFP4, fx.MXFP8) },
			want: map[string]string{
				"gate":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4]",
				"up":     "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4]",
				"down":   "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
				"shared": "*nn.Linear",
			},
		},
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
			// All three projections must be quantized; otherwise every bank is dequantized.
			name:     "L3 nvfp4 gate+up with bf16 down",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.NVFP4, fx.NVFP4, fx.BF16) },
			want:     bf16,
		},
		{
			name:     "experts wins over switch_mlp",
			defaults: fx.MXFP8,
			build: func(s *fx.Set) {
				stacked(s, group, ".weight", fx.BF16, fx.BF16, fx.BF16)
				stacked(s, switchG, ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8)
			},
			want: bf16,
		},
		{
			name:     "projections resolved from different layouts",
			defaults: fx.MXFP8,
			build: func(s *fx.Set) {
				s.Weight(group+".gate_proj.weight", fx.MXFP8, experts, inter, hidden)
				s.Weight(switchG+".up_proj.weight", fx.MXFP8, experts, inter, hidden)
				s.PerExpert(group, "down_proj", fx.MXFP8, experts, hidden, inter, nil)
			},
			want: mxfp8,
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
			name:          "singular shared_expert names",
			defaults:      fx.MXFP8,
			singularShare: true,
			build:         func(s *fx.Set) { stacked(s, switchG, ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8) },
			want:          mxfp8,
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			s.Weight("model.embed_tokens.weight", fx.BF16, 16, hidden)
			s.Weight("model.norm.weight", fx.BF16, hidden)
			s.Weight(layer+".input_layernorm.weight", fx.BF16, hidden)
			s.Weight(layer+".self_attn.q_proj.weight", fx.BF16, 64, hidden)
			s.Weight(layer+".self_attn.k_proj.weight", fx.BF16, 32, hidden)
			s.Weight(layer+".self_attn.v_proj.weight", fx.BF16, 32, hidden)
			s.Weight(layer+".self_attn.o_proj.weight", fx.BF16, hidden, 64)
			s.Weight(layer+".mlp.gate.weight", fx.BF16, experts, hidden)
			shared := layer + ".mlp.shared_experts"
			if tc.singularShare {
				shared = layer + ".mlp.shared_expert"
			}
			s.Weight(shared+".gate_proj.weight", fx.BF16, inter, hidden)
			s.Weight(shared+".up_proj.weight", fx.BF16, inter, hidden)
			s.Weight(shared+".down_proj.weight", fx.BF16, hidden, inter)
			tc.build(s)

			cfg := &Config{
				HiddenSize: hidden, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 32,
				LayerNormEps: 1e-5, LayerTypes: []string{"sliding_attention"}, MLPLayerTypes: []string{"sparse"},
				NumExperts: experts, NumExpertsPerTok: 2, NumSharedExperts: 1,
				TensorQuant: s.Quant(),
			}
			cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = fx.Defaults(tc.defaults)
			m := &Model{Config: cfg, Layers: make([]*Layer, 1)}
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
			moe, ok := m.Layers[0].MLP.(*SparseMoE)
			if !ok {
				t.Fatalf("layer 0 MLP = %T, want *SparseMoE", m.Layers[0].MLP)
			}
			for _, d := range fx.Compare(cohere2Banks(moe), tc.want) {
				t.Error(d)
			}
		})
	}
}

// cohere2Banks describes the banks SwitchMLP.Forward reads and the shared
// expert's projection type.
func cohere2Banks(moe *SparseMoE) map[string]string {
	s := moe.SwitchMLP
	out := map[string]string{"shared": "-"}
	if moe.SharedExpert != nil {
		out["shared"] = fmt.Sprintf("%T", moe.SharedExpert.GateProj)
	}
	if !s.UseQuantized {
		out["gate"] = fx.Bank{Weight: s.GateWeight}.String()
		out["up"] = fx.Bank{Weight: s.UpWeight}.String()
		out["down"] = fx.Bank{Weight: s.DownWeight}.String()
		return out
	}
	out["gate"] = fx.Bank{
		Weight: s.GateWeightQ, Scales: s.GateScales, Biases: s.GateBiases, GlobalScale: s.GateGlobalScales,
		GroupSize: s.GateGroupSize, Bits: s.GateBits, Mode: s.GateMode,
	}.String()
	out["up"] = fx.Bank{
		Weight: s.UpWeightQ, Scales: s.UpScales, Biases: s.UpBiases, GlobalScale: s.UpGlobalScales,
		GroupSize: s.UpGroupSize, Bits: s.UpBits, Mode: s.UpMode,
	}.String()
	out["down"] = fx.Bank{
		Weight: s.DownWeightQ, Scales: s.DownScales, Biases: s.DownBiases, GlobalScale: s.DownGlobalScales,
		GroupSize: s.DownGroupSize, Bits: s.DownBits, Mode: s.DownMode,
	}.String()
	return out
}
