package gemma4

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

// TestExpertLayouts pins how LoadWeights resolves every routed-expert layout
// a Gemma 4 package can carry, so a loader migration shows its intent as a
// diff.
func TestExpertLayouts(t *testing.T) {
	const (
		experts = 4
		hidden  = 64
		inter   = 128
		prefix  = "model.language_model."
		layer   = prefix + "layers.0"
	)
	fused := func(s *fx.Set, base, suffix string, gateUp, down fx.Encoding) {
		s.Weight(base+".gate_up_proj"+suffix, gateUp, experts, 2*inter, hidden)
		s.Weight(base+".down_proj"+suffix, down, experts, hidden, inter)
	}
	separate := func(s *fx.Set, base, suffix string, gate, up, down fx.Encoding) {
		s.Weight(base+".gate_proj"+suffix, gate, experts, inter, hidden)
		s.Weight(base+".up_proj"+suffix, up, experts, inter, hidden)
		s.Weight(base+".down_proj"+suffix, down, experts, hidden, inter)
	}
	perExpert := func(s *fx.Set, group string, gate, up, down fx.Encoding, gs []float32) {
		s.PerExpert(group, "gate_proj", gate, experts, inter, hidden, gs)
		s.PerExpert(group, "up_proj", up, experts, inter, hidden, gs)
		s.PerExpert(group, "down_proj", down, experts, hidden, inter, gs)
	}
	gs := []float32{0.5, 1, 2, 4}

	denseFused := map[string]string{
		"gate_up": "dense BF16[4 64 256]",
		"down":    "dense BF16[4 128 64]",
	}
	denseSeparate := map[string]string{
		"gate": "dense BF16[4 64 128]",
		"up":   "dense BF16[4 64 128]",
		"down": "dense BF16[4 128 64]",
	}
	nvfp4Separate := map[string]string{
		"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
		"up":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
		"down": "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
	}
	nvfp4Fused := map[string]string{
		"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4]",
		"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8]",
	}
	const missing = "layer 0: missing MoE expert weights"

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
			build:    func(s *fx.Set) { fused(s, layer+".experts", "", fx.BF16, fx.BF16) },
			want:     denseFused,
		},
		{
			name:     "L1 nvfp4 gate_up with mxfp8 down bare",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { fused(s, layer+".experts", "", fx.NVFP4, fx.MXFP8) },
			want: map[string]string{
				"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4]",
				"down":    "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L1 int4 fused bare",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { fused(s, layer+".experts", "", fx.Int4, fx.Int4) },
			want: map[string]string{
				"gate_up": "affine/64/4 U32[4 256 8] s=BF16[4 256 1] qb=BF16[4 256 1]",
				"down":    "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
			},
		},
		{
			name:     "L2 mxfp8 fused",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { fused(s, layer+".experts", ".weight", fx.MXFP8, fx.MXFP8) },
			want: map[string]string{
				"gate_up": "mxfp8/32/8 U32[4 256 16] s=U8[4 256 2]",
				"down":    "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L2 nvfp4 fused with global scale bank and input scale",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				fused(s, layer+".experts", ".weight", fx.NVFP4, fx.NVFP4)
				for _, p := range []string{"gate_up_proj", "down_proj"} {
					key := layer + ".experts." + p + ".weight"
					s.GlobalScale(key, gs...)
					s.Raw(key+".input_scale", mlx.FromValue(float32(1)))
				}
			},
			want: map[string]string{
				"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4] gs=F32[4]",
				"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
			},
		},
		{
			name:     "era A moe.switch_mlp int4 fused",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { fused(s, layer+".moe.switch_mlp", ".weight", fx.Int4, fx.Int4) },
			want: map[string]string{
				"gate_up": "affine/64/4 U32[4 256 8] s=BF16[4 256 1] qb=BF16[4 256 1]",
				"down":    "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
			},
		},
		{
			name:     "moe. bf16 fused bare",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { fused(s, layer+".moe", "", fx.BF16, fx.BF16) },
			want:     denseFused,
		},
		{
			name:     "moe.experts fused stacked",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { fused(s, layer+".moe.experts", ".weight", fx.BF16, fx.BF16) },
			want:     denseFused,
		},
		{
			// Both banks go dense when either lacks scales; the quantized one is dequantized.
			name:     "fused nvfp4 gate_up with bf16 down",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { fused(s, layer+".experts", ".weight", fx.NVFP4, fx.BF16) },
			want: map[string]string{
				"gate_up": "dense BF16[4 64 256]",
				"down":    "dense BF16[4 128 64]",
			},
		},
		{
			name:     "bf16 separate bare experts.",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { separate(s, layer+".experts", "", fx.BF16, fx.BF16, fx.BF16) },
			want:     denseSeparate,
		},
		{
			name:     "bf16 separate bare moe.",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { separate(s, layer+".moe", "", fx.BF16, fx.BF16, fx.BF16) },
			want:     denseSeparate,
		},
		{
			name:     "int4 separate bare",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { separate(s, layer+".experts", "", fx.Int4, fx.Int4, fx.Int4) },
			want: map[string]string{
				"gate": "affine/64/4 U32[4 128 8] s=BF16[4 128 1] qb=BF16[4 128 1]",
				"up":   "affine/64/4 U32[4 128 8] s=BF16[4 128 1] qb=BF16[4 128 1]",
				"down": "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
			},
		},
		{
			name:     "L3 bf16 separate",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { separate(s, layer+".experts", ".weight", fx.BF16, fx.BF16, fx.BF16) },
			want:     denseSeparate,
		},
		{
			name:     "L3 mxfp8 separate",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { separate(s, layer+".experts", ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8) },
			want: map[string]string{
				"gate": "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"up":   "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"down": "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L3 nvfp4 separate with global scale bank",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				separate(s, layer+".experts", ".weight", fx.NVFP4, fx.NVFP4, fx.NVFP4)
				for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
					s.GlobalScale(layer+".experts."+p+".weight", gs...)
				}
			},
			want: nvfp4Separate,
		},
		{
			name:     "L4 nvfp4 per-expert with scalar global scales",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { perExpert(s, layer+".experts", fx.NVFP4, fx.NVFP4, fx.NVFP4, gs) },
			want:     nvfp4Separate,
		},
		{
			name:     "L4 nvfp4 per-expert under moe.experts",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { perExpert(s, layer+".moe.experts", fx.NVFP4, fx.NVFP4, fx.NVFP4, gs) },
			want:     nvfp4Separate,
		},
		{
			name:     "L4 mxfp8 per-expert",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { perExpert(s, layer+".experts", fx.MXFP8, fx.MXFP8, fx.MXFP8, nil) },
			want: map[string]string{
				"gate": "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"up":   "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"down": "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L4 bf16 per-expert",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { perExpert(s, layer+".experts", fx.BF16, fx.BF16, fx.BF16, nil) },
			want:     denseSeparate,
		},
		{
			// All three projections must be quantized; otherwise every bank is dequantized.
			name:     "L4 nvfp4 gate+up with bf16 down per-expert",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				s.PerExpert(layer+".experts", "gate_proj", fx.NVFP4, experts, inter, hidden, gs)
				s.PerExpert(layer+".experts", "up_proj", fx.NVFP4, experts, inter, hidden, gs)
				s.PerExpert(layer+".experts", "down_proj", fx.BF16, experts, hidden, inter, nil)
			},
			want: denseSeparate,
		},
		{
			name:     "L4 nvfp4 gate with mxfp8 up and down",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { perExpert(s, layer+".experts", fx.NVFP4, fx.MXFP8, fx.MXFP8, nil) },
			want: map[string]string{
				"gate": "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4]",
				"up":   "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"down": "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
			},
		},
		{
			name:     "L4 per-expert with one global scale missing",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				perExpert(s, layer+".experts", fx.NVFP4, fx.NVFP4, fx.NVFP4, gs)
				s.Delete(layer + ".experts.3.gate_proj.weight.global_scale")
			},
			err: "3 of 4 experts have global scales",
		},
		{
			name:     "era A per-expert fused gate_up under moe.experts",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				s.PerExpert(layer+".moe.experts", "gate_up_proj", fx.NVFP4, experts, 2*inter, hidden, nil)
				s.PerExpert(layer+".moe.experts", "down_proj", fx.NVFP4, experts, hidden, inter, nil)
			},
			// Stacked at load and kept fused, like the stacked form.
			want: nvfp4Fused,
		},
		{
			name:     "per-expert fused gate_up under experts",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				s.PerExpert(layer+".experts", "gate_up_proj", fx.NVFP4, experts, 2*inter, hidden, nil)
				s.PerExpert(layer+".experts", "down_proj", fx.NVFP4, experts, hidden, inter, nil)
			},
			want: nvfp4Fused,
		},
		{
			name:     "nothing under any prefix",
			defaults: fx.BF16,
			build:    func(s *fx.Set) {},
			err:      missing,
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			layoutGemma4Base(s, prefix, layer, experts, hidden, inter)
			tc.build(s)

			cfg := &TextConfig{
				HiddenSize: hidden, NumHiddenLayers: 1, IntermediateSize: inter,
				NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 32, GlobalHeadDim: 32,
				RMSNormEps: 1e-6, LayerTypes: []string{"sliding_attention"},
				EnableMoeBlock: true, NumExperts: experts, TopKExperts: 2, ExpertIntermediateSize: inter,
				TensorQuant: s.Quant(),
			}
			cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = fx.Defaults(tc.defaults)
			m := &Model{TextConfig: cfg, Layers: make([]*DecoderLayer, 1)}
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
			for _, d := range fx.Compare(gemma4Banks(m.Layers[0].MoE), tc.want) {
				t.Error(d)
			}
		})
	}
}

// layoutGemma4Base stores every non-expert tensor a one-layer MoE model needs.
func layoutGemma4Base(s *fx.Set, prefix, layer string, experts, hidden, inter int) {
	s.Weight(prefix+"embed_tokens.weight", fx.BF16, 16, hidden)
	s.Weight(prefix+"norm.weight", fx.BF16, hidden)
	for _, norm := range []string{
		"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
		"post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2",
	} {
		s.Weight(layer+"."+norm+".weight", fx.BF16, hidden)
	}
	s.Weight(layer+".self_attn.q_proj.weight", fx.BF16, 64, hidden)
	s.Weight(layer+".self_attn.k_proj.weight", fx.BF16, 32, hidden)
	s.Weight(layer+".self_attn.v_proj.weight", fx.BF16, 32, hidden)
	s.Weight(layer+".self_attn.o_proj.weight", fx.BF16, hidden, 64)
	s.Weight(layer+".self_attn.q_norm.weight", fx.BF16, 32)
	s.Weight(layer+".self_attn.k_norm.weight", fx.BF16, 32)
	s.Weight(layer+".mlp.gate_proj.weight", fx.BF16, inter, hidden)
	s.Weight(layer+".mlp.up_proj.weight", fx.BF16, inter, hidden)
	s.Weight(layer+".mlp.down_proj.weight", fx.BF16, hidden, inter)
	s.Weight(layer+".router.proj.weight", fx.BF16, experts, hidden)
	s.Weight(layer+".router.scale", fx.BF16, hidden)
	s.Weight(layer+".router.per_expert_scale", fx.BF16, experts)
}

// gemma4Banks describes the banks MoEBlock.Forward reads, with the quant mode
// it passes for each.
func gemma4Banks(m *MoEBlock) map[string]string {
	out := make(map[string]string)
	add := func(name string, b fx.Bank) {
		if b.Weight != nil {
			out[name] = b.String()
		}
	}
	if !m.UseQuantized {
		if m.UseFusedGateUp && m.GateUpWeight != nil {
			add("gate_up", fx.Bank{Weight: m.GateUpWeight})
		} else {
			add("gate", fx.Bank{Weight: m.GateWeight})
			add("up", fx.Bank{Weight: m.UpWeight})
		}
		add("down", fx.Bank{Weight: m.DownWeight})
		return out
	}
	if m.UseFusedGateUp {
		add("gate_up", fx.Bank{
			Weight: m.GateUpWeightQ, Scales: m.GateUpScales, Biases: m.GateUpBiases, GlobalScale: m.GateUpGlobalScales,
			GroupSize: m.GateUpGroupSize, Bits: m.GateUpBits, Mode: m.QuantMode,
		})
	} else {
		add("gate", fx.Bank{
			Weight: m.GateWeightQ, Scales: m.GateScales, Biases: m.GateBiases, GlobalScale: m.GateGlobalScales,
			GroupSize: m.GateGroupSize, Bits: m.GateBits, Mode: m.QuantMode,
		})
		upMode := m.UpQuantMode
		if upMode == "" {
			upMode = m.QuantMode
		}
		add("up", fx.Bank{
			Weight: m.UpWeightQ, Scales: m.UpScales, Biases: m.UpBiases, GlobalScale: m.UpGlobalScales,
			GroupSize: m.UpGroupSize, Bits: m.UpBits, Mode: upMode,
		})
	}
	downMode := m.DownQuantMode
	if downMode == "" {
		downMode = m.QuantMode
	}
	add("down", fx.Bank{
		Weight: m.DownWeightQ, Scales: m.DownScales, Biases: m.DownBiases, GlobalScale: m.DownGlobalScales,
		GroupSize: m.DownGroupSize, Bits: m.DownBits, Mode: downMode,
	})
	return out
}
