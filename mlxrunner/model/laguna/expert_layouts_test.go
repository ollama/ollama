package laguna

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

// TestExpertLayouts pins how LoadWeights resolves every routed-expert layout
// a Laguna package can carry, so a loader migration shows its intent as a
// diff.
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
	gs := []float32{0.5, 1, 2, 4}

	bf16Source := map[string]string{
		"gate":   "dense BF16[4 128 64]",
		"up":     "dense BF16[4 128 64]",
		"down":   "dense BF16[4 64 128]",
		"layout": "gate_up=source down=source",
	}
	mxfp8Fused := map[string]string{
		"gate_up": "mxfp8/32/8 U32[4 256 16] s=U8[4 256 2]",
		"down":    "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
		"layout":  "gate_up=gather down=gather",
	}
	nvfp4FusedScaled := map[string]string{
		"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4] gs=F32[4]",
		"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
		"layout":  "gate_up=gather down=gather",
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
			want:     bf16Source,
		},
		{
			name:     "L3 mxfp8 gate+up with bf16 down",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.MXFP8, fx.MXFP8, fx.BF16) },
			want: map[string]string{
				"gate_up": "mxfp8/32/8 U32[4 256 16] s=U8[4 256 2]",
				"down":    "dense BF16[4 64 128]",
				"layout":  "gate_up=gather down=source",
			},
		},
		{
			name:     "L3 mxfp8",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8) },
			want:     mxfp8Fused,
		},
		{
			name:     "L3 nvfp4 without global scale",
			defaults: fx.NVFP4,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.NVFP4, fx.NVFP4, fx.NVFP4) },
			want: map[string]string{
				"gate_up": "nvfp4/16/4 U32[4 256 8] s=U8[4 256 4]",
				"down":    "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8]",
				"layout":  "gate_up=gather down=gather",
			},
		},
		{
			name:     "L3 nvfp4 with matching global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				stacked(s, group, ".weight", fx.NVFP4, fx.NVFP4, fx.NVFP4)
				for _, p := range []string{"gate_proj", "up_proj", "down_proj"} {
					s.GlobalScale(group+"."+p+".weight", gs...)
				}
			},
			want: nvfp4FusedScaled,
		},
		{
			name:     "L3 nvfp4 with differing gate+up global scale banks",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				stacked(s, group, ".weight", fx.NVFP4, fx.NVFP4, fx.NVFP4)
				s.GlobalScale(group+".gate_proj.weight", gs...)
				s.GlobalScale(group+".up_proj.weight", 4, 2, 1, 0.5)
				s.GlobalScale(group+".down_proj.weight", gs...)
			},
			want: map[string]string{
				"gate":   "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"up":     "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down":   "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
				"layout": "gate_up=gather down=gather",
			},
		},
		{
			name:     "L3 int4 with qbias",
			defaults: fx.Int4,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.Int4, fx.Int4, fx.Int4) },
			want: map[string]string{
				"gate_up": "affine/64/4 U32[4 256 8] s=BF16[4 256 1] qb=BF16[4 256 1]",
				"down":    "affine/64/4 U32[4 64 16] s=BF16[4 64 2] qb=BF16[4 64 2]",
				"layout":  "gate_up=gather down=gather",
			},
		},
		{
			// bug: laguna.go:1083-1085 keeps the separate gate and up banks
			// alive beside the fused one, doubling their memory.
			name:     "L3 f32 dense",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { stacked(s, group, ".weight", fx.F32, fx.F32, fx.F32) },
			want: map[string]string{
				"gate_up": "dense F32[4 64 256]",
				"gate":    "dense F32[4 64 128]",
				"up":      "dense F32[4 64 128]",
				"down":    "dense F32[4 128 64]",
				"layout":  "gate_up=gather down=gather",
			},
		},
		{
			name:     "bare stacked bf16",
			defaults: fx.BF16,
			build:    func(s *fx.Set) { stacked(s, group, "", fx.BF16, fx.BF16, fx.BF16) },
			want:     bf16Source,
		},
		{
			name:     "L4 nvfp4 per-expert with scalar global scales",
			defaults: fx.NVFP4,
			build: func(s *fx.Set) {
				s.PerExpert(group, "gate_proj", fx.NVFP4, experts, inter, hidden, gs)
				s.PerExpert(group, "up_proj", fx.NVFP4, experts, inter, hidden, gs)
				s.PerExpert(group, "down_proj", fx.NVFP4, experts, hidden, inter, gs)
			},
			want: nvfp4FusedScaled,
		},
		{
			name:     "L4 bf16 per-expert",
			defaults: fx.BF16,
			build: func(s *fx.Set) {
				s.PerExpert(group, "gate_proj", fx.BF16, experts, inter, hidden, nil)
				s.PerExpert(group, "up_proj", fx.BF16, experts, inter, hidden, nil)
				s.PerExpert(group, "down_proj", fx.BF16, experts, hidden, inter, nil)
			},
			want: bf16Source,
		},
		{
			name:     "L5 switch_mlp mxfp8",
			defaults: fx.MXFP8,
			build:    func(s *fx.Set) { stacked(s, layer+".mlp.switch_mlp", ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8) },
			want:     mxfp8Fused,
		},
		{
			name:     "switch_mlp wins over experts",
			defaults: fx.MXFP8,
			build: func(s *fx.Set) {
				stacked(s, layer+".mlp.switch_mlp", ".weight", fx.MXFP8, fx.MXFP8, fx.MXFP8)
				stacked(s, group, ".weight", fx.BF16, fx.BF16, fx.BF16)
			},
			want: mxfp8Fused,
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
			want: bf16Source,
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			layoutLagunaBase(s, layer, experts, hidden, inter)
			tc.build(s)

			cfg := &Config{
				HiddenSize: hidden, IntermediateSize: inter, MoeIntermediateSize: inter, SharedExpertIntermediate: inter,
				NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 32, VocabSize: 16,
				NumExperts: experts, NumExpertsPerTok: 2, DecoderSparseStep: 1, MoeRoutedScalingFactor: 1,
				RMSNormEps: 1e-5, TieWordEmbeddings: true, TensorQuant: s.Quant(),
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
			for _, d := range fx.Compare(lagunaBanks(moe.SwitchMLP), tc.want) {
				t.Error(d)
			}
		})
	}
}

// layoutLagunaBase stores every non-expert tensor a one-layer MoE model needs.
func layoutLagunaBase(s *fx.Set, layer string, experts, hidden, inter int) {
	s.Weight("model.embed_tokens.weight", fx.BF16, 16, hidden)
	s.Weight("model.norm.weight", fx.BF16, hidden)
	s.Weight(layer+".input_layernorm.weight", fx.BF16, hidden)
	s.Weight(layer+".post_attention_layernorm.weight", fx.BF16, hidden)
	s.Weight(layer+".self_attn.q_proj.weight", fx.BF16, 64, hidden)
	s.Weight(layer+".self_attn.k_proj.weight", fx.BF16, 32, hidden)
	s.Weight(layer+".self_attn.v_proj.weight", fx.BF16, 32, hidden)
	s.Weight(layer+".self_attn.o_proj.weight", fx.BF16, hidden, 64)
	s.Weight(layer+".self_attn.g_proj.weight", fx.BF16, 2, hidden)
	s.Weight(layer+".self_attn.q_norm.weight", fx.BF16, 32)
	s.Weight(layer+".self_attn.k_norm.weight", fx.BF16, 32)
	s.Weight(layer+".mlp.gate.weight", fx.BF16, experts, hidden)
	s.Weight(layer+".mlp.experts.e_score_correction_bias", fx.BF16, experts)
	s.Weight(layer+".mlp.shared_expert.gate_proj.weight", fx.BF16, inter, hidden)
	s.Weight(layer+".mlp.shared_expert.up_proj.weight", fx.BF16, inter, hidden)
	s.Weight(layer+".mlp.shared_expert.down_proj.weight", fx.BF16, hidden, inter)
}

// lagunaBanks describes every bank SwitchMLP holds, including dense banks the
// forward pass no longer reads, and whether dense banks stay in source layout.
func lagunaBanks(s *SwitchMLP) map[string]string {
	out := make(map[string]string)
	add := func(name string, q fx.Bank, dense *mlx.Array) {
		switch {
		case q.Weight != nil:
			out[name] = q.String()
		case dense != nil:
			out[name] = fx.Bank{Weight: dense}.String()
		}
	}
	add("gate_up", fx.Bank{
		Weight: s.GateUpWeightQ, Scales: s.GateUpScales, Biases: s.GateUpBiases, GlobalScale: s.GateUpGlobalScale,
		GroupSize: s.GateUpGroupSize, Bits: s.GateUpBits, Mode: s.GateUpMode,
	}, s.GateUpWeight)
	add("gate", fx.Bank{
		Weight: s.GateWeightQ, Scales: s.GateScales, Biases: s.GateBiases, GlobalScale: s.GateGlobalScale,
		GroupSize: s.GateGroupSize, Bits: s.GateBits, Mode: s.GateMode,
	}, s.GateWeight)
	add("up", fx.Bank{
		Weight: s.UpWeightQ, Scales: s.UpScales, Biases: s.UpBiases, GlobalScale: s.UpGlobalScale,
		GroupSize: s.UpGroupSize, Bits: s.UpBits, Mode: s.UpMode,
	}, s.UpWeight)
	add("down", fx.Bank{
		Weight: s.DownWeightQ, Scales: s.DownScales, Biases: s.DownBiases, GlobalScale: s.DownGlobalScale,
		GroupSize: s.DownGroupSize, Bits: s.DownBits, Mode: s.DownMode,
	}, s.DownWeight)
	layout := func(source bool) string {
		if source {
			return "source"
		}
		return "gather"
	}
	out["layout"] = fmt.Sprintf("gate_up=%s down=%s", layout(s.GateUpWeightsSourceLayout), layout(s.DownWeightSourceLayout))
	return out
}
