package model

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

const testExperts = 3

// expertValues returns distinct values per expert so a misordered stack is visible.
func expertValues(e, out, in int) []float32 {
	v := make([]float32, out*in)
	for i := range v {
		v[i] = float32(e+1)*0.5 + float32(i%7)*0.03125
	}
	return v
}

// putExpert stores one expert's [out, in] weight under key, quantized with mode
// when mode is set, following the runner's companion names.
func putExpert(tensors map[string]*mlx.Array, key string, e int, mode string, global float32) {
	w := mlx.FromValues(expertValues(e, 32, 64), 32, 64).AsType(mlx.DTypeBFloat16)
	if mode == "" {
		tensors[key] = w
		return
	}
	gs, bits, m := QuantizationParams(mode)
	q, s, b := mlx.Quantize(w, gs, bits, m)
	tensors[key], tensors[key+"_scale"] = q, s
	if b != nil {
		tensors[key+"_qbias"] = b
	}
	if global != 0 {
		tensors[key+".global_scale"] = mlx.FromValues([]float32{global}, 1)
	}
}

func stackedFixture(key, mode string, withGlobal bool) map[string]*mlx.Array {
	per := map[string]*mlx.Array{}
	for e := range testExperts {
		g := float32(0)
		if withGlobal {
			g = float32(e+1) / 1024
		}
		putExpert(per, fmt.Sprintf("x.%d", e), e, mode, g)
	}
	stack := func(suffix string) *mlx.Array {
		parts := make([]*mlx.Array, testExperts)
		for e := range testExperts {
			parts[e] = per[fmt.Sprintf("x.%d%s", e, suffix)]
		}
		return mlx.Stack(parts, 0)
	}
	tensors := map[string]*mlx.Array{key: stack("")}
	if mode != "" {
		tensors[key+"_scale"] = stack("_scale")
		if per["x.0_qbias"] != nil {
			tensors[key+"_qbias"] = stack("_qbias")
		}
	}
	if withGlobal {
		tensors[key+".global_scale"] = mlx.Reshape(stack(".global_scale"), testExperts)
	}
	return tensors
}

func TestReadExpertBankLayouts(t *testing.T) {
	mlxtest.Run(t, func(mt *mlxtest.T) {
		for _, mode := range []string{"", "int4", "mxfp8", "nvfp4"} {
			withGlobal := mode == "nvfp4"
			// Every layout of the same weights reads to the same bank.
			var want []float32
			for _, layout := range []string{"stacked", "stacked bare", "per-expert", "per-expert bare"} {
				name := mode + "/" + layout
				tensors := map[string]*mlx.Array{}
				switch layout {
				case "stacked":
					tensors = stackedFixture("L.experts.up_proj.weight", mode, withGlobal)
				case "stacked bare":
					tensors = stackedFixture("L.experts.up_proj", mode, withGlobal)
				default:
					suffix := ".weight"
					if layout == "per-expert bare" {
						suffix = ""
					}
					for e := range testExperts {
						g := float32(0)
						if withGlobal {
							g = float32(e+1) / 1024
						}
						putExpert(tensors, fmt.Sprintf("L.experts.%d.up_proj%s", e, suffix), e, mode, g)
					}
				}
				n := len(tensors)
				var quant QuantDefaults
				if mode != "" {
					quant.GroupSize, quant.Bits, quant.Mode = QuantizationParams(mode)
				}
				bank, err := ReadExpertBank(tensors, quant, testExperts, "up_proj", "L.switch_mlp", "L.experts")
				if err != nil || bank == nil {
					t.Fatalf("%s: ReadExpertBank() = %v, %v", name, bank, err)
				}
				if bank.Quantized() != (mode != "") || bank.Weight.Dim(0) != testExperts || (bank.GlobalScale != nil) != withGlobal {
					t.Errorf("%s: quantized=%v experts=%d global=%v", name, bank.Quantized(), bank.Weight.Dim(0), bank.GlobalScale != nil)
				}
				if mode != "" {
					gs, bits, m := QuantizationParams(mode)
					if bank.GroupSize != gs || bank.Bits != bits || bank.Mode != m {
						t.Errorf("%s: params %d/%d/%s, want %d/%d/%s", name, bank.GroupSize, bank.Bits, bank.Mode, gs, bits, m)
					}
				}
				dense := bank.Dense().AsType(mlx.DTypeFloat32)
				mlx.Eval(dense)
				got := dense.Floats()
				if want == nil {
					want = got
				} else if !slices.Equal(got, want) {
					t.Errorf("%s: dense bank differs from the stacked layout", name)
				}
				bank.Release(tensors)
				if len(tensors) != 0 {
					t.Errorf("%s: Release left %d of %d tensors", name, len(tensors), n)
				}
			}
		}
	})
}

func TestReadExpertBankPrefixOrderAndMisses(t *testing.T) {
	mlxtest.Run(t, func(mt *mlxtest.T) {
		tensors := stackedFixture("L.experts.up_proj.weight", "", false)
		for k, v := range stackedFixture("L.switch_mlp.up_proj.weight", "", false) {
			tensors[k] = v
		}
		bank, err := ReadExpertBank(tensors, QuantDefaults{}, testExperts, "up_proj", "L.switch_mlp", "L.experts")
		if err != nil || bank.Key != "L.switch_mlp.up_proj.weight" {
			t.Errorf("first prefix should win: %v, %v", bank, err)
		}
		if bank, err := ReadExpertBank(tensors, QuantDefaults{}, testExperts, "gate_proj", "L.experts"); bank != nil || err != nil {
			t.Errorf("absent projection = %v, %v; want nil, nil", bank, err)
		}
	})
}

func TestReadExpertBankRejectsInconsistentBanks(t *testing.T) {
	mlxtest.Run(t, func(mt *mlxtest.T) {
		perExpert := func(modes []string, globals []float32) map[string]*mlx.Array {
			tensors := map[string]*mlx.Array{}
			for e, mode := range modes {
				putExpert(tensors, fmt.Sprintf("L.experts.%d.up_proj.weight", e), e, mode, globals[e])
			}
			return tensors
		}
		for _, tc := range []struct {
			name    string
			tensors map[string]*mlx.Array
			want    string
		}{
			{"missing expert", perExpert([]string{"nvfp4", "nvfp4"}, []float32{1, 1}), "expert 2 of 3 is missing"},
			{"quantized and dense experts", perExpert([]string{"nvfp4", "", "nvfp4"}, []float32{1, 0, 1}), "experts have scales"},
			{"different modes", perExpert([]string{"nvfp4", "mxfp8", "nvfp4"}, []float32{0, 0, 0}), "expert 1 is mxfp8"},
			{"partial global scales", perExpert([]string{"nvfp4", "nvfp4", "nvfp4"}, []float32{1, 0, 1}), "experts have global scales"},
			{"wrong expert count", stackedFixture("L.experts.up_proj.weight", "", false), "want [4, out, in]"},
		} {
			experts := testExperts
			if tc.name == "wrong expert count" {
				experts = 4
			}
			// Per-tensor metadata, as blobs carry it, names each expert's encoding.
			quant := QuantDefaults{GroupSize: 16, Bits: 4, Mode: "nvfp4", TensorQuant: map[string]*TensorQuantInfo{}}
			if tc.name == "different modes" {
				quant.TensorQuant["L.experts.1.up_proj.weight"] = &TensorQuantInfo{QuantType: "mxfp8", GroupSize: 32}
			}
			_, err := ReadExpertBank(tc.tensors, quant, experts, "up_proj", "L.experts")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("%s: err = %v, want %q", tc.name, err, tc.want)
			}
		}
	})
}

func TestExpertBankDenseAppliesGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(mt *mlxtest.T) {
		tensors := stackedFixture("L.experts.up_proj.weight", "", false)
		tensors["L.experts.up_proj.weight.global_scale"] = mlx.FromValues([]float32{0.5, 2, 4}, 3)
		bank, err := ReadExpertBank(tensors, QuantDefaults{}, testExperts, "up_proj", "L.experts")
		if err != nil {
			t.Fatal(err)
		}
		dense := bank.Dense().AsType(mlx.DTypeFloat32)
		raw := bank.Weight.AsType(mlx.DTypeFloat32)
		mlx.Eval(dense, raw)
		d, r := dense.Floats(), raw.Floats()
		per := len(d) / testExperts
		for e, scale := range []float32{0.5, 2, 4} {
			if d[e*per] != r[e*per]*scale {
				t.Errorf("expert %d: dense %v, want %v x %v", e, d[e*per], r[e*per], scale)
			}
		}
	})
}

func TestSupportsGatherQMM(t *testing.T) {
	for _, tt := range []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 4, want: true},
		{mode: "affine", bits: 8, want: true},
		{mode: "affine", bits: 2, want: false},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "unknown", bits: 4, want: false},
	} {
		if got := SupportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Errorf("SupportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}
