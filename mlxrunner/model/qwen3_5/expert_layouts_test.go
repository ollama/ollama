package qwen3_5

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

// TestExpertLayouts pins how LoadWeights resolves every routed-expert layout a
// Qwen 3.5 MoE package can carry, in the main stack and the MTP head, so a
// loader migration shows its intent as a diff.
func TestExpertLayouts(t *testing.T) {
	const (
		experts = 4
		hidden  = 64
		inter   = 128
		layer   = "model.language_model.layers.0"
		group   = layer + ".mlp.experts"
		mtp     = "mtp.layers.0"
	)
	fused := func(s *fx.Set, group, suffix string, gateUp, down fx.Encoding) {
		s.Weight(group+".gate_up_proj"+suffix, gateUp, experts, 2*inter, hidden)
		s.Weight(group+".down_proj"+suffix, down, experts, hidden, inter)
	}
	separate := func(s *fx.Set, group string, gate, up, down fx.Encoding) {
		s.Weight(group+".gate_proj.weight", gate, experts, inter, hidden)
		s.Weight(group+".up_proj.weight", up, experts, inter, hidden)
		s.Weight(group+".down_proj.weight", down, experts, hidden, inter)
	}
	perExpert := func(s *fx.Set, group string, enc fx.Encoding, gs []float32) {
		s.PerExpert(group, "gate_proj", enc, experts, inter, hidden, gs)
		s.PerExpert(group, "up_proj", enc, experts, inter, hidden, gs)
		s.PerExpert(group, "down_proj", enc, experts, hidden, inter, gs)
	}
	scale := func(s *fx.Set, group string, projs ...string) {
		for _, p := range projs {
			s.GlobalScale(group+"."+p+".weight", 0.5, 1, 2, 4)
		}
	}
	gs := []float32{0.5, 1, 2, 4}

	denseFused := map[string]string{
		"gate_up": "dense BF16[4 64 256]",
		"down":    "dense BF16[4 128 64]",
	}
	nvfp4Fused := map[string]string{
		"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4]",
		"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8]",
	}
	nvfp4FusedScaled := map[string]string{
		"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4] gs=F32[4]",
		"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
	}
	mxfp8Fused := map[string]string{
		"gate_up": "mxfp8/32/8 U32[4 256 16] s=U8[4 256 2]",
		"down":    "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
	}
	int4Fused := map[string]string{
		"gate_up": "affine/64/4 U32[4 256 8] s=BF16[4 256 1] qb=BF16[4 256 1]",
		"down":    "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
	}

	for _, tc := range []struct {
		name     string
		defaults fx.Encoding
		build    func(*fx.Set)
		err      string
		want     map[string]string
	}{
		{
			name:     "L1 bf16 fused bare",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { fused(s, group, "", fx.BF16, fx.BF16) },
			want:     denseFused,
		},
		{
			name:     "L1 nvfp4 fused bare",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { fused(s, group, "", fx.NVFP4, fx.NVFP4) },
			want:     nvfp4Fused,
		},
		{
			name:     "L1 mxfp8 fused bare",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { fused(s, group, "", fx.MXFP8, fx.MXFP8) },
			want:     mxfp8Fused,
		},
		{
			name:     "L2 nvfp4 fused with global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, group, ".weight", fx.NVFP4, fx.NVFP4)
				scale(s, group, "gate_up_proj", "down_proj")
			},
			want: nvfp4FusedScaled,
		},
		{
			name:     "fused nvfp4 gate_up with bf16 down",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { fused(s, group, ".weight", fx.NVFP4, fx.BF16) },
			want: map[string]string{
				"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4]",
				"down":    "dense BF16[4 128 64]",
			},
		},
		{
			name:     "L3 bf16 separate fuses at load",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { separate(s, group, fx.BF16, fx.BF16, fx.BF16) },
			want:     denseFused,
		},
		{
			name:     "L3 mxfp8 separate fuses at load",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { separate(s, group, fx.MXFP8, fx.MXFP8, fx.MXFP8) },
			want:     mxfp8Fused,
		},
		{
			name:     "L3 nvfp4 separate with matching global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				separate(s, group, fx.NVFP4, fx.NVFP4, fx.NVFP4)
				scale(s, group, "gate_proj", "up_proj", "down_proj")
			},
			want: nvfp4FusedScaled,
		},
		{
			name:     "L3 nvfp4 separate with differing gate+up global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				separate(s, group, fx.NVFP4, fx.NVFP4, fx.NVFP4)
				scale(s, group, "gate_proj", "down_proj")
				s.GlobalScale(group+".up_proj.weight", 4, 2, 1, 0.5)
			},
			want: map[string]string{
				"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"up":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down": "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
			},
		},
		{
			name:     "L4 bf16 per-expert fuses at load",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { perExpert(s, group, fx.BF16, nil) },
			want:     denseFused,
		},
		{
			name:     "L4 nvfp4 per-expert with scalar global scales",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { perExpert(s, group, fx.NVFP4, gs) },
			want:     nvfp4FusedScaled,
		},
		{
			name:     "stacked gate with per-expert up and down",
			defaults: fx.BF16,
			build: func(s *fx.Set) {
				s.Weight(group+".gate_proj.weight", fx.BF16, experts, inter, hidden)
				s.PerExpert(group, "up_proj", fx.BF16, experts, inter, hidden, nil)
				s.PerExpert(group, "down_proj", fx.BF16, experts, hidden, inter, nil)
			},
			want: denseFused,
		},
		{
			name:     "era A switch_mlp int4 separate",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { separate(s, layer+".mlp.switch_mlp", fx.Int4, fx.Int4, fx.Int4) },
			want:     int4Fused,
		},
		{
			name:     "switch_mlp wins over experts",
			defaults: fx.Int4,
			build: func(s *fx.Set) {
				separate(s, layer+".mlp.switch_mlp", fx.Int4, fx.Int4, fx.Int4)
				separate(s, group, fx.BF16, fx.BF16, fx.BF16)
			},
			want: int4Fused,
		},
		{
			name:     "switch_mlp fused gate_up",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { fused(s, layer+".mlp.switch_mlp", ".weight", fx.BF16, fx.BF16) },
			want:     denseFused,
		},
		{
			name:     "MTP L4 bf16 per-expert beside L2 nvfp4 main experts",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, group, ".weight", fx.NVFP4, fx.NVFP4)
				scale(s, group, "gate_up_proj", "down_proj")
				layoutQwen35MTP(s, mtp, experts, hidden, inter)
				perExpert(s, mtp+".mlp.experts", fx.BF16, nil)
			},
			want: withMTP(nvfp4FusedScaled, denseFused),
		},
		{
			name:     "MTP L3 mxfp8 separate beside L2 nvfp4 main experts",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, group, ".weight", fx.NVFP4, fx.NVFP4)
				scale(s, group, "gate_up_proj", "down_proj")
				layoutQwen35MTP(s, mtp, experts, hidden, inter)
				separate(s, mtp+".mlp.experts", fx.MXFP8, fx.MXFP8, fx.MXFP8)
			},
			want: withMTP(nvfp4FusedScaled, mxfp8Fused),
		},
		{
			name:     "MTP L1 bf16 fused beside L1 nvfp4 main experts",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, group, "", fx.NVFP4, fx.NVFP4)
				layoutQwen35MTP(s, mtp, experts, hidden, inter)
				fused(s, mtp+".mlp.experts", "", fx.BF16, fx.BF16)
			},
			want: withMTP(nvfp4Fused, denseFused),
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			s.Weight("model.language_model.embed_tokens.weight", fx.BF16, 16, hidden)
			s.Weight("model.language_model.norm.weight", fx.BF16, hidden)
			layoutQwen35Layer(s, layer, experts, hidden, inter)
			tc.build(s)

			cfg := &Config{
				HiddenSize: hidden, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 32,
				RMSNormEps: 1e-6, TieWordEmbeddings: true, LayerTypes: []string{"full_attention"},
				NumExperts: experts, NumExpertsPerTok: 2, MoeIntermediateSize: inter, SharedExpertIntermediateSize: inter,
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
			got := qwen35Banks(t, m.Layers[0], "")
			if m.MTP != nil {
				for k, v := range qwen35Banks(t, m.MTP.Layer, "mtp ") {
					got[k] = v
				}
			}
			for _, d := range fx.Compare(got, tc.want) {
				t.Error(d)
			}
		})
	}
}

// layoutQwen35Layer stores every non-expert tensor of a full-attention MoE
// layer.
func layoutQwen35Layer(s *fx.Set, layer string, experts, hidden, inter int) {
	s.Weight(layer+".input_layernorm.weight", fx.BF16, hidden)
	s.Weight(layer+".post_attention_layernorm.weight", fx.BF16, hidden)
	s.Weight(layer+".self_attn.q_proj.weight", fx.BF16, 128, hidden)
	s.Weight(layer+".self_attn.k_proj.weight", fx.BF16, 32, hidden)
	s.Weight(layer+".self_attn.v_proj.weight", fx.BF16, 32, hidden)
	s.Weight(layer+".self_attn.o_proj.weight", fx.BF16, hidden, 64)
	s.Weight(layer+".self_attn.q_norm.weight", fx.BF16, 32)
	s.Weight(layer+".self_attn.k_norm.weight", fx.BF16, 32)
	s.Weight(layer+".mlp.gate.weight", fx.BF16, experts, hidden)
	s.Weight(layer+".mlp.shared_expert.gate_proj.weight", fx.BF16, inter, hidden)
	s.Weight(layer+".mlp.shared_expert.up_proj.weight", fx.BF16, inter, hidden)
	s.Weight(layer+".mlp.shared_expert.down_proj.weight", fx.BF16, hidden, inter)
	s.Weight(layer+".mlp.shared_expert_gate.weight", fx.BF16, 1, hidden)
}

// layoutQwen35MTP stores the MTP head around a single MoE layer; the caller
// adds its routed experts.
func layoutQwen35MTP(s *fx.Set, layer string, experts, hidden, inter int) {
	s.Weight("mtp.fc.weight", fx.BF16, hidden, 2*hidden)
	s.Weight("mtp.pre_fc_norm_embedding.weight", fx.BF16, hidden)
	s.Weight("mtp.pre_fc_norm_hidden.weight", fx.BF16, hidden)
	s.Weight("mtp.norm.weight", fx.BF16, hidden)
	layoutQwen35Layer(s, layer, experts, hidden, inter)
}

// qwen35Banks describes every bank SwitchMLP holds.
func qwen35Banks(t *mlxtest.T, layer *Layer, prefix string) map[string]string {
	moe, ok := layer.MLP.(*SparseMoE)
	if !ok {
		t.Fatalf("layer MLP = %T, want *SparseMoE", layer.MLP)
	}
	s := moe.SwitchMLP
	out := make(map[string]string)
	add := func(name string, q fx.Bank, dense *mlx.Array) {
		switch {
		case q.Weight != nil:
			out[prefix+name] = q.String()
		case dense != nil:
			out[prefix+name] = fx.Bank{Weight: dense}.String()
		}
	}
	add("gate_up", fx.Bank{
		Weight: s.GateUpWeightQ, Scales: s.GateUpScales, Biases: s.GateUpBias, GlobalScale: s.GateUpGlobalScales,
		GroupSize: s.GateUpGroupSize, Bits: s.GateUpBits, Mode: s.GateUpMode,
	}, s.GateUpWeight)
	add("gate", fx.Bank{
		Weight: s.GateWeightQ, Scales: s.GateScales, Biases: s.GateBiases, GlobalScale: s.GateGlobalScales,
		GroupSize: s.GateGroupSize, Bits: s.GateBits, Mode: s.GateMode,
	}, s.GateWeight)
	add("up", fx.Bank{
		Weight: s.UpWeightQ, Scales: s.UpScales, Biases: s.UpBiases, GlobalScale: s.UpGlobalScales,
		GroupSize: s.UpGroupSize, Bits: s.UpBits, Mode: s.UpMode,
	}, s.UpWeight)
	add("down", fx.Bank{
		Weight: s.DownWeightQ, Scales: s.DownScales, Biases: s.DownBiases, GlobalScale: s.DownGlobalScales,
		GroupSize: s.DownGroupSize, Bits: s.DownBits, Mode: s.DownMode,
	}, s.DownWeight)
	return out
}

func withMTP(main, mtp map[string]string) map[string]string {
	out := make(map[string]string, len(main)+len(mtp))
	for k, v := range main {
		out[k] = v
	}
	for k, v := range mtp {
		out["mtp "+k] = v
	}
	return out
}
