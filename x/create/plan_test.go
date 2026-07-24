package create

import (
	"slices"
	"sort"
	"strings"
	"testing"
)

func TestPlanRejectsOutputNameCollision(t *testing.T) {
	// A source shipping both an MLX-pattern weight (foo.weight + foo.scales)
	// and a compressed-tensors weight (foo.weight_packed + foo.weight_scale)
	// fuses both to the output name foo.weight.
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"model.a.weight":        "U32",
		"model.a.scales":        "BF16",
		"model.a.weight_packed": "U8",
		"model.a.weight_scale":  "F8_E4M3",
	})
	_, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err == nil || !strings.Contains(err.Error(), "clashing name") {
		t.Fatalf("Plan() error = %v, want a clashing-name error", err)
	}
}

func TestPlanFloat(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"model.embed_tokens.weight":              "BF16",
		"model.layers.0.input_layernorm.weight":  "BF16",
		"model.layers.0.self_attn.q_proj.weight": "BF16",
		"model.layers.0.self_attn.v_proj.weight": "BF16",
		"model.layers.0.mlp.down_proj.weight":    "BF16",
	})

	t.Run("no quantize", func(t *testing.T) {
		specs, err := Plan(inv, Classification{Kind: SourceFloat}, defaultQuantPolicy{})
		if err != nil {
			t.Fatalf("Plan() error = %v", err)
		}
		if len(specs) != len(inv.Tensors) {
			t.Fatalf("got %d specs, want %d", len(specs), len(inv.Tensors))
		}
		if !sort.SliceIsSorted(specs, func(i, j int) bool { return specs[i].Name < specs[j].Name }) {
			t.Error("specs are not sorted by name")
		}
		for _, s := range specs {
			if len(s.Tensors) != 1 || s.Tensors[0].Name != s.Name || sourceName(s.Tensors[0]) != s.Name {
				t.Errorf("%s: unexpected tensors %+v", s.Name, s.Tensors)
			}
			if s.Tensors[0].Quantize != "" {
				t.Errorf("%s: quantize = %q, want empty when no --quantize", s.Name, s.Tensors[0].Quantize)
			}
		}
	})

	t.Run("4-bit fp promotes sensitive tensors to mxfp8", func(t *testing.T) {
		for _, quant := range []string{"nvfp4", "mxfp4"} {
			specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: quant}, defaultQuantPolicy{})
			if err != nil {
				t.Fatalf("Plan(%s) error = %v", quant, err)
			}
			got := make(map[string]string)
			for _, s := range specs {
				got[s.Name] = s.Tensors[0].Quantize
			}
			want := map[string]string{
				"model.embed_tokens.weight":              "",
				"model.layers.0.input_layernorm.weight":  "",
				"model.layers.0.self_attn.q_proj.weight": quant,
				"model.layers.0.self_attn.v_proj.weight": "mxfp8",
				"model.layers.0.mlp.down_proj.weight":    "mxfp8",
			}
			for name, w := range want {
				if got[name] != w {
					t.Errorf("%s: %s quantize = %q, want %q", quant, name, got[name], w)
				}
			}
		}
	})

	t.Run("int4 applies the mixed-precision policy", func(t *testing.T) {
		specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "int4"}, defaultQuantPolicy{})
		if err != nil {
			t.Fatalf("Plan() error = %v", err)
		}
		got := make(map[string]string)
		for _, s := range specs {
			got[s.Name] = s.Tensors[0].Quantize
		}
		want := map[string]string{
			"model.embed_tokens.weight":              "",     // embeddings stay full precision
			"model.layers.0.input_layernorm.weight":  "",     // norms stay full precision
			"model.layers.0.self_attn.q_proj.weight": "int4", // standard linear
			"model.layers.0.self_attn.v_proj.weight": "int8", // promoted for quality
			"model.layers.0.mlp.down_proj.weight":    "int8", // promoted for quality
		}
		for name, w := range want {
			if got[name] != w {
				t.Errorf("%s: quantize = %q, want %q", name, got[name], w)
			}
		}
	})
}

func TestPlanFloatExpertGroup(t *testing.T) {
	// A float MoE layer that ships per-expert tensors: two experts, two
	// projections. Each projection is stacked into one [experts, out, in]
	// tensor in a single packed blob; the routing gate and norm stay plain.
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"model.layers.0.mlp.experts.0.gate_proj.weight": "BF16",
		"model.layers.0.mlp.experts.1.gate_proj.weight": "BF16",
		"model.layers.0.mlp.experts.0.down_proj.weight": "BF16",
		"model.layers.0.mlp.experts.1.down_proj.weight": "BF16",
		"model.layers.0.mlp.gate.weight":                "BF16",
		"model.layers.0.input_layernorm.weight":         "BF16",
	})

	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "int4"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	group, ok := specByName(specs, "model.layers.0.mlp.experts")
	if !ok {
		t.Fatalf("missing packed expert blob; got %v", specNames(specs))
	}
	if len(group.Tensors) != 2 {
		t.Fatalf("packed blob has %d tensors, want 2 (gate_proj, down_proj)", len(group.Tensors))
	}

	gate, ok := inputByOutput(group, "model.layers.0.mlp.experts.gate_proj.weight")
	if !ok {
		t.Fatal("packed blob missing stacked gate_proj")
	}
	if gate.Transform != TransformStackExperts || len(gate.Sources) != 2 {
		t.Errorf("gate_proj = %+v, want stack of 2 experts", gate)
	}
	if !slices.Equal(gate.OutShape, []int32{2, 128, 128}) {
		t.Errorf("gate_proj stacked shape = %v, want [2 128 128]", gate.OutShape)
	}
	if gate.Sources[0].Name != "model.layers.0.mlp.experts.0.gate_proj.weight" ||
		gate.Sources[1].Name != "model.layers.0.mlp.experts.1.gate_proj.weight" {
		t.Errorf("gate_proj sources out of order: %v", gate.Sources)
	}
	if gate.Quantize != "int4" {
		t.Errorf("gate_proj quantize = %q, want int4", gate.Quantize)
	}

	down, _ := inputByOutput(group, "model.layers.0.mlp.experts.down_proj.weight")
	if down.Quantize != "int8" {
		t.Errorf("down_proj quantize = %q, want int8 (promoted)", down.Quantize)
	}

	// Routing gate and norm are not expert tensors; they stay as their own blobs.
	for _, name := range []string{"model.layers.0.mlp.gate.weight", "model.layers.0.input_layernorm.weight"} {
		if _, ok := specByName(specs, name); !ok {
			t.Errorf("%s should be its own blob", name)
		}
	}
}

func TestPlanFloatPrestackedExperts(t *testing.T) {
	// qwen3.5 / gemma4 ship experts pre-stacked and fused: one tensor per
	// projection covering all experts. These are quantized as ordinary 3D
	// stacked tensors (each its own blob, no per-expert stacking); the runtime
	// splits the fused gate_up at load. Names are kept exactly.
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"model.layers.0.mlp.experts.gate_up_proj.weight": {Name: "model.layers.0.mlp.experts.gate_up_proj.weight", Dtype: "BF16", Shape: []int32{8, 256, 128}},
		"model.layers.0.mlp.experts.down_proj.weight":    {Name: "model.layers.0.mlp.experts.down_proj.weight", Dtype: "BF16", Shape: []int32{8, 128, 128}},
		"model.layers.0.mlp.gate.weight":                 {Name: "model.layers.0.mlp.gate.weight", Dtype: "BF16", Shape: []int32{8, 128}},
	}}

	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "int4"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	// Each tensor is its own blob; nothing is grouped/stacked.
	gateUp, ok := specByName(specs, "model.layers.0.mlp.experts.gate_up_proj.weight")
	if !ok {
		t.Fatalf("fused gate_up should be its own blob; got %v", specNames(specs))
	}
	if len(gateUp.Tensors) != 1 || gateUp.Tensors[0].Transform != TransformNone || len(gateUp.Tensors[0].Sources) != 1 {
		t.Errorf("gate_up should pass through unstacked: %+v", gateUp.Tensors)
	}
	if gateUp.Tensors[0].Quantize != "int4" {
		t.Errorf("gate_up quantize = %q, want int4", gateUp.Tensors[0].Quantize)
	}
	down, _ := specByName(specs, "model.layers.0.mlp.experts.down_proj.weight")
	if down.Tensors[0].Quantize != "int8" {
		t.Errorf("down_proj quantize = %q, want int8 (promoted)", down.Tensors[0].Quantize)
	}
	// No packed group blob should have been produced.
	if _, ok := specByName(specs, "model.layers.0.mlp.experts"); ok {
		t.Error("pre-stacked experts must not be re-grouped into a packed blob")
	}
}

func TestPlanExpertGroupMismatchedLayout(t *testing.T) {
	// Experts in the same projection with different shapes cannot be stacked.
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"model.layers.0.mlp.experts.0.gate_proj.weight": {Name: "model.layers.0.mlp.experts.0.gate_proj.weight", Dtype: "BF16", Shape: []int32{128, 128}},
		"model.layers.0.mlp.experts.1.gate_proj.weight": {Name: "model.layers.0.mlp.experts.1.gate_proj.weight", Dtype: "BF16", Shape: []int32{128, 64}},
	}}
	if _, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "int4"}, defaultQuantPolicy{}); err == nil {
		t.Fatal("Plan() error = nil, want mismatched expert layout error")
	}
}
