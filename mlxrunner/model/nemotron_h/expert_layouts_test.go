package nemotron_h

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	fx "github.com/ollama/ollama/mlxrunner/model/expertfixture"
)

const (
	layoutExperts = 4
	layoutHidden  = 64
	layoutInter   = 128
	layoutShared  = 256
	layoutLatent  = 32
)

// TestExpertLayouts pins how LoadWeights resolves every routed-expert layout
// a NemotronH package can carry, including shared-expert folding, latent
// banks and MTP experts, so a loader migration shows its intent as a diff.
func TestExpertLayouts(t *testing.T) {
	const (
		layer = "backbone.layers.0"
		group = layer + ".mixer.experts"
		mtp   = "mtp.layers.0"
	)
	gs := []float32{0.5, 1, 2, 4}
	stacked := func(s *fx.Set, group string, up, down fx.Encoding, width int) {
		s.Weight(group+".up_proj.weight", up, layoutExperts, layoutInter, width)
		s.Weight(group+".down_proj.weight", down, layoutExperts, width, layoutInter)
	}
	perExpert := func(s *fx.Set, group string, enc fx.Encoding, scales []float32) {
		s.PerExpert(group, "up_proj", enc, layoutExperts, layoutInter, layoutHidden, scales)
		s.PerExpert(group, "down_proj", enc, layoutExperts, layoutHidden, layoutInter, scales)
	}

	nvfp4Folded := map[string]string{
		"up":     "nvfp4/16/4 U32[6 128 8] s=U8[6 128 4] gs=F32[6]",
		"down":   "nvfp4/16/4 U32[6 64 16] s=U8[6 64 8] gs=F32[6]",
		"folded": "2",
		"shared": "-",
	}
	mxfp8Folded := map[string]string{
		"up":     "mxfp8/32/8 U32[6 128 16] s=U8[6 128 2]",
		"down":   "mxfp8/32/8 U32[6 64 32] s=U8[6 64 4]",
		"folded": "2",
		"shared": "-",
	}
	bf16 := map[string]string{
		"up":     "dense BF16[4 64 128]",
		"down":   "dense BF16[4 128 64]",
		"folded": "0",
		"shared": "*nn.Linear",
	}

	for _, tc := range []struct {
		name     string
		defaults fx.Encoding
		shared   fx.Encoding
		sharedGS bool
		latent   bool
		build    func(*fx.Set)
		err      string
		want     map[string]string
	}{
		{
			name:     "L4 nvfp4 per-expert, nvfp4 shared folds",
			defaults: fx.NVFP4, shared: fx.NVFP4, sharedGS: true,
			build: func(s *fx.Set) { perExpert(s, group, fx.NVFP4, gs) },
			want:  nvfp4Folded,
		},
		{
			name:     "L4 mxfp8 per-expert, mxfp8 shared folds",
			defaults: fx.MXFP8, shared: fx.MXFP8,
			build: func(s *fx.Set) { perExpert(s, group, fx.MXFP8, nil) },
			want:  mxfp8Folded,
		},
		{
			name:     "L4 nvfp4 per-expert, bf16 shared stays separate",
			defaults: fx.NVFP4, shared: fx.BF16,
			build: func(s *fx.Set) { perExpert(s, group, fx.NVFP4, gs) },
			want: map[string]string{
				"up":     "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down":   "nvfp4/16/4 U32[4 64 16] s=U8[4 64 8] gs=F32[4]",
				"folded": "0",
				"shared": "*nn.Linear",
			},
		},
		{
			name:     "L4 bf16 per-expert",
			defaults: fx.BF16, shared: fx.BF16,
			build: func(s *fx.Set) { perExpert(s, group, fx.BF16, nil) },
			want:  bf16,
		},
		{
			name:     "L4 per-expert bare names",
			defaults: fx.BF16, shared: fx.BF16,
			build: func(s *fx.Set) {
				for e := range layoutExperts {
					s.Weight(fmt.Sprintf("%s.%d.up_proj", group, e), fx.BF16, layoutInter, layoutHidden)
					s.Weight(fmt.Sprintf("%s.%d.down_proj", group, e), fx.BF16, layoutHidden, layoutInter)
				}
			},
			want: bf16,
		},
		{
			name:     "L3 bf16 stacked",
			defaults: fx.BF16, shared: fx.BF16,
			build: func(s *fx.Set) { stacked(s, group, fx.BF16, fx.BF16, layoutHidden) },
			want:  bf16,
		},
		{
			name:     "L3 mxfp8 stacked, mxfp8 shared folds",
			defaults: fx.MXFP8, shared: fx.MXFP8,
			build: func(s *fx.Set) { stacked(s, group, fx.MXFP8, fx.MXFP8, layoutHidden) },
			want:  mxfp8Folded,
		},
		{
			name:     "L3 nvfp4 stacked with global scale bank, nvfp4 shared folds",
			defaults: fx.NVFP4, shared: fx.NVFP4, sharedGS: true,
			build: func(s *fx.Set) {
				stacked(s, group, fx.NVFP4, fx.NVFP4, layoutHidden)
				s.GlobalScale(group+".up_proj.weight", gs...)
				s.GlobalScale(group+".down_proj.weight", gs...)
			},
			want: nvfp4Folded,
		},
		{
			// The routed bank folds against the identity scale.
			name:     "L3 nvfp4 stacked without global scale, scaled shared folds",
			defaults: fx.NVFP4, shared: fx.NVFP4, sharedGS: true,
			build: func(s *fx.Set) { stacked(s, group, fx.NVFP4, fx.NVFP4, layoutHidden) },
			want:  nvfp4Folded,
		},
		{
			name:     "bare stacked nvfp4",
			defaults: fx.NVFP4, shared: fx.NVFP4,
			build: func(s *fx.Set) {
				s.Weight(group+".up_proj", fx.NVFP4, layoutExperts, layoutInter, layoutHidden)
				s.Weight(group+".down_proj", fx.NVFP4, layoutExperts, layoutHidden, layoutInter)
			},
			want: map[string]string{
				"up":     "nvfp4/16/4 U32[6 128 8] s=U8[6 128 4]",
				"down":   "nvfp4/16/4 U32[6 64 16] s=U8[6 64 8]",
				"folded": "2",
				"shared": "-",
			},
		},
		{
			// Folding needs both banks quantized.
			name:     "L3 nvfp4 up with bf16 down",
			defaults: fx.NVFP4, shared: fx.NVFP4, sharedGS: true,
			build: func(s *fx.Set) {
				stacked(s, group, fx.NVFP4, fx.BF16, layoutHidden)
				s.GlobalScale(group+".up_proj.weight", gs...)
			},
			want: map[string]string{
				"up":     "nvfp4/16/4 U32[4 128 8] s=U8[4 128 4] gs=F32[4]",
				"down":   "dense BF16[4 128 64]",
				"folded": "0",
				"shared": "*nn.QuantizedLinear",
			},
		},
		{
			name:     "L5 switch_mlp is not looked up",
			defaults: fx.BF16, shared: fx.BF16,
			build: func(s *fx.Set) { stacked(s, layer+".mixer.switch_mlp", fx.BF16, fx.BF16, layoutHidden) },
			err:   "layer 0: missing routed expert weights",
		},
		{
			name:     "L3 bf16 latent stacked",
			defaults: fx.BF16, shared: fx.BF16, latent: true,
			build: func(s *fx.Set) { stacked(s, group, fx.BF16, fx.BF16, layoutLatent) },
			want: map[string]string{
				"up":     "dense BF16[4 32 128]",
				"down":   "dense BF16[4 128 32]",
				"folded": "0",
				"shared": "*nn.Linear",
				"latent": "true",
			},
		},
		{
			name:     "L3 mxfp8 latent stacked",
			defaults: fx.MXFP8, shared: fx.MXFP8, latent: true,
			build: func(s *fx.Set) { stacked(s, group, fx.MXFP8, fx.MXFP8, layoutLatent) },
			want: map[string]string{
				"up":     "mxfp8/32/8 U32[4 128 8] s=U8[4 128 1]",
				"down":   "mxfp8/32/8 U32[4 32 32] s=U8[4 32 4]",
				"folded": "0",
				"shared": "*nn.QuantizedLinear",
				"latent": "true",
			},
		},
		{
			name:     "MTP L4 bf16 per-expert beside nvfp4 main experts",
			defaults: fx.NVFP4, shared: fx.NVFP4, sharedGS: true,
			build: func(s *fx.Set) {
				perExpert(s, group, fx.NVFP4, gs)
				layoutNemotronMTP(s, mtp)
				perExpert(s, mtp+".mixer.experts", fx.BF16, nil)
			},
			want: withMTP(nvfp4Folded, bf16),
		},
		{
			name:     "MTP L3 bf16 stacked beside bf16 main experts",
			defaults: fx.BF16, shared: fx.BF16,
			build: func(s *fx.Set) {
				stacked(s, group, fx.BF16, fx.BF16, layoutHidden)
				layoutNemotronMTP(s, mtp)
				stacked(s, mtp+".mixer.experts", fx.BF16, fx.BF16, layoutHidden)
			},
			want: withMTP(bf16, bf16),
		},
		{
			name:     "MTP L4 mxfp8 per-expert beside mxfp8 main experts",
			defaults: fx.MXFP8, shared: fx.MXFP8,
			build: func(s *fx.Set) {
				stacked(s, group, fx.MXFP8, fx.MXFP8, layoutHidden)
				layoutNemotronMTP(s, mtp)
				perExpert(s, mtp+".mixer.experts", fx.MXFP8, nil)
			},
			want: withMTP(mxfp8Folded, map[string]string{
				"up":     "mxfp8/32/8 U32[4 128 16] s=U8[4 128 2]",
				"down":   "mxfp8/32/8 U32[4 64 32] s=U8[4 64 4]",
				"folded": "0",
				"shared": "*nn.Linear",
			}),
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			layoutNemotronBase(s, layer, tc.shared, tc.sharedGS, tc.latent)
			tc.build(s)

			m, err := layoutNemotronLoad(s, tc.defaults, tc.latent)
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("LoadWeights error = %v, want %q", err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatalf("LoadWeights: %v", err)
			}
			got := nemotronBanks(m.Layers[0].MoE, "")
			if m.MTP != nil {
				for k, v := range nemotronBanks(m.MTP.Layers[0].MoE, "mtp ") {
					got[k] = v
				}
			}
			for _, d := range fx.Compare(got, tc.want) {
				t.Error(d)
			}
		})
	}
}

// TestExpertLayoutDenseGlobalScale pins what happens to a global scale shipped
// beside an unquantized expert bank: stacked banks multiply it into the
// weights, per-expert banks drop it.
func TestExpertLayoutDenseGlobalScale(t *testing.T) {
	const (
		layer = "backbone.layers.0"
		group = layer + ".mixer.experts"
	)
	gs := []float32{0.5, 1, 2, 4}
	for _, tc := range []struct {
		name    string
		build   func(*fx.Set)
		applied bool
	}{
		{
			name: "L3 stacked bf16 applies the scale",
			build: func(s *fx.Set) {
				s.Weight(group+".up_proj.weight", fx.BF16, layoutExperts, layoutInter, layoutHidden)
				s.Weight(group+".down_proj.weight", fx.BF16, layoutExperts, layoutHidden, layoutInter)
				s.GlobalScale(group+".up_proj.weight", gs...)
				s.GlobalScale(group+".down_proj.weight", gs...)
			},
			applied: true,
		},
		{
			// Per-expert dense banks apply their global scales, like stacked ones.
			name: "L4 per-expert bf16 applies the scale",
			build: func(s *fx.Set) {
				s.PerExpert(group, "up_proj", fx.BF16, layoutExperts, layoutInter, layoutHidden, gs)
				s.PerExpert(group, "down_proj", fx.BF16, layoutExperts, layoutHidden, layoutInter, gs)
			},
			applied: true,
		},
	} {
		mlxtest.RunSubtest(t, tc.name, func(t *mlxtest.T) {
			s := fx.New()
			layoutNemotronBase(s, layer, fx.BF16, false, false)
			tc.build(s)
			tensors := s.Tensors()
			var source *mlx.Array
			if w := tensors[group+".up_proj.weight"]; w != nil {
				source = w
			} else {
				parts := make([]*mlx.Array, layoutExperts)
				for e := range parts {
					parts[e] = tensors[fmt.Sprintf("%s.%d.up_proj.weight", group, e)]
				}
				source = mlx.Stack(parts, 0)
			}

			m, err := layoutNemotronLoad(s, fx.BF16, false)
			if err != nil {
				t.Fatalf("LoadWeights: %v", err)
			}
			want := source
			if tc.applied {
				want = mlx.Mul(source, mlx.FromValues(gs, layoutExperts, 1, 1).AsType(source.DType()))
			}
			want = mlx.Transpose(want, 0, 2, 1).AsType(mlx.DTypeFloat32)
			got := m.Layers[0].MoE.UpWeight.AsType(mlx.DTypeFloat32)
			mlx.Eval(got, want)
			gotValues, wantValues := got.Floats(), want.Floats()
			if len(gotValues) != len(wantValues) {
				t.Fatalf("up bank has %d values, want %d", len(gotValues), len(wantValues))
			}
			for i := range gotValues {
				if !(math.Abs(float64(gotValues[i]-wantValues[i])) <= 1e-6) {
					t.Fatalf("up[%d] = %v, want %v", i, gotValues[i], wantValues[i])
				}
			}
		})
	}
}

// layoutNemotronBase stores every non-expert tensor a one-layer MoE model
// needs, with the shared expert in the given encoding.
func layoutNemotronBase(s *fx.Set, layer string, shared fx.Encoding, sharedGS, latent bool) {
	s.Weight("backbone.embeddings.weight", fx.BF16, 16, layoutHidden)
	s.Weight("backbone.norm_f.weight", fx.BF16, layoutHidden)
	layoutNemotronMoELayer(s, layer, shared, sharedGS)
	if latent {
		s.Weight(layer+".mixer.fc1_latent_proj.weight", fx.BF16, layoutLatent, layoutHidden)
		s.Weight(layer+".mixer.fc2_latent_proj.weight", fx.BF16, layoutHidden, layoutLatent)
	}
}

func layoutNemotronMoELayer(s *fx.Set, layer string, shared fx.Encoding, sharedGS bool) {
	s.Weight(layer+".norm.weight", fx.BF16, layoutHidden)
	s.Weight(layer+".mixer.gate.weight", fx.BF16, layoutExperts, layoutHidden)
	s.Weight(layer+".mixer.shared_experts.up_proj.weight", shared, layoutShared, layoutHidden)
	s.Weight(layer+".mixer.shared_experts.down_proj.weight", shared, layoutHidden, layoutShared)
	if sharedGS {
		s.ScalarGlobalScale(layer+".mixer.shared_experts.up_proj.weight", 0.25)
		s.ScalarGlobalScale(layer+".mixer.shared_experts.down_proj.weight", 0.25)
	}
}

// layoutNemotronMTP stores a native MTP head whose single layer is an MoE
// layer with a BF16 shared expert; the caller adds its routed experts.
func layoutNemotronMTP(s *fx.Set, layer string) {
	s.Weight(layer+".enorm.weight", fx.BF16, layoutHidden)
	s.Weight(layer+".hnorm.weight", fx.BF16, layoutHidden)
	s.Weight(layer+".eh_proj.weight", fx.BF16, layoutHidden, 2*layoutHidden)
	s.Weight(layer+".final_layernorm.weight", fx.BF16, layoutHidden)
	layoutNemotronMoELayer(s, layer, fx.BF16, false)
}

func layoutNemotronLoad(s *fx.Set, defaults fx.Encoding, latent bool) (*Model, error) {
	cfg := &Config{
		HiddenSize: layoutHidden, NumHiddenLayers: 1, LayerTypes: []byte{'E'},
		LayerNormEpsilon: 1e-5, TieWordEmbeddings: true,
		NRoutedExperts: layoutExperts, NSharedExperts: 1, NumExpertsPerTok: 2,
		MoEIntermediateSize: layoutInter, MoESharedExpertIntermediateSize: layoutShared,
		ExpertGroupCount: 1, ExpertGroupUsedCount: 1,
		TensorQuant: s.Quant(),
	}
	if latent {
		cfg.MoELatentSize = layoutLatent
	}
	cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode = fx.Defaults(defaults)
	m := &Model{Config: cfg, Layers: make([]*Layer, 1)}
	return m, m.LoadWeights(s.Tensors())
}

// nemotronBanks describes the routed banks SparseMoE.Forward reads, whether
// the shared expert was folded into them, and which shared projections remain.
func nemotronBanks(m *SparseMoE, prefix string) map[string]string {
	out := map[string]string{
		prefix + "up": fx.Bank{
			Weight: m.UpWeightQ, Scales: m.UpScales, Biases: m.UpBiases, GlobalScale: m.UpGlobalScales,
			GroupSize: m.UpGroupSize, Bits: m.UpBits, Mode: m.UpMode,
		}.String(),
		prefix + "down": fx.Bank{
			Weight: m.DownWeightQ, Scales: m.DownScales, Biases: m.DownBiases, GlobalScale: m.DownGlobalScales,
			GroupSize: m.DownGroupSize, Bits: m.DownBits, Mode: m.DownMode,
		}.String(),
		prefix + "folded": fmt.Sprint(m.FoldedSharedExperts),
		prefix + "shared": "-",
	}
	if m.UpWeightQ == nil {
		out[prefix+"up"] = fx.Bank{Weight: m.UpWeight}.String()
	}
	if m.DownWeightQ == nil {
		out[prefix+"down"] = fx.Bank{Weight: m.DownWeight}.String()
	}
	if m.SharedUp != nil {
		out[prefix+"shared"] = fmt.Sprintf("%T", m.SharedUp)
	}
	if m.LatentIn != nil {
		out[prefix+"latent"] = "true"
	}
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
