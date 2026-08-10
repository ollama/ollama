package create

import (
	"slices"
	"testing"
)

func blockFP8(tensors map[string]string) Inventory {
	// Block-fp8 sources are routed to planBlockFP8 by kind; the config is not
	// needed once classified, so build the inventory directly with realistic
	// shapes for the quantize policy.
	m := make(map[string]SourceTensor)
	for name, dtype := range tensors {
		shape := []int32{256, 256}
		if dtype != "F8_E4M3" && dtype != "BF16" {
			shape = []int32{1} // scale companions are small
		}
		m[name] = SourceTensor{Name: name, Dtype: dtype, Shape: shape}
	}
	return Inventory{Dir: "test", Tensors: m}
}

func TestPlanBlockFP8(t *testing.T) {
	inv := blockFP8(map[string]string{
		"model.layers.0.self_attn.q_proj.weight":           "F8_E4M3",
		"model.layers.0.self_attn.q_proj.weight_scale_inv": "F32",
		"model.layers.0.mlp.down_proj.weight":              "F8_E4M3",
		"model.layers.0.mlp.down_proj.weight_scale_inv":    "F32",
		"model.layers.0.input_layernorm.weight":            "F32", // norm stays as-is
		"model.embed_tokens.weight":                        "BF16",
		"lm_head.weight":                                   "BF16",
	})

	specs, err := Plan(inv, Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	// FP8 weight: decode + quantize to mxfp8, scale folded in (not its own blob).
	q, ok := specByName(specs, "model.layers.0.self_attn.q_proj.weight")
	if !ok {
		t.Fatal("missing q_proj blob")
	}
	ts := q.Tensors[0]
	if ts.Transform != TransformDecodeFP8 || ts.Quantize != "mxfp8" || ts.OutDtype != "BF16" {
		t.Errorf("q_proj = %+v, want decode_fp8 + mxfp8 + BF16", ts)
	}
	if len(ts.Sources) != 2 || ts.Sources[0].Name != "model.layers.0.self_attn.q_proj.weight" ||
		ts.Sources[1].Name != "model.layers.0.self_attn.q_proj.weight_scale_inv" {
		t.Errorf("q_proj sources = %v, want [weight, scale_inv]", ts.Sources)
	}
	if _, leaked := specByName(specs, "model.layers.0.self_attn.q_proj.weight_scale_inv"); leaked {
		t.Error("scale companion must not be its own blob")
	}

	// Norm stays at source precision (F32), not quantized, not cast.
	norm, _ := specByName(specs, "model.layers.0.input_layernorm.weight")
	if norm.Tensors[0].Quantize != "" || norm.Tensors[0].Transform != TransformNone || norm.Tensors[0].OutDtype != "" {
		t.Errorf("norm = %+v, want kept at source precision", norm.Tensors[0])
	}

	// BF16 weights pass through untouched.
	lmHead, _ := specByName(specs, "lm_head.weight")
	if lmHead.Tensors[0].Quantize != "" || lmHead.Tensors[0].Transform != TransformNone {
		t.Errorf("lm_head = %+v, want kept at source precision", lmHead.Tensors[0])
	}
}

func TestPlanBlockFP8PrestackedExperts(t *testing.T) {
	// A pre-stacked fp8 expert tensor (one [E, out, in] tensor + one scale)
	// decodes + quantizes as an ordinary tensor — no per-expert stacking.
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"model.layers.0.mlp.experts.gate_up_proj.weight":           {Name: "model.layers.0.mlp.experts.gate_up_proj.weight", Dtype: "F8_E4M3", Shape: []int32{8, 512, 256}},
		"model.layers.0.mlp.experts.gate_up_proj.weight_scale_inv": {Name: "model.layers.0.mlp.experts.gate_up_proj.weight_scale_inv", Dtype: "F32", Shape: []int32{8, 4, 2}},
	}}
	specs, err := Plan(inv, Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	w, ok := specByName(specs, "model.layers.0.mlp.experts.gate_up_proj.weight")
	if !ok || len(specs) != 1 {
		t.Fatalf("want single decode blob; got %v", specNames(specs))
	}
	ts := w.Tensors[0]
	if ts.Transform != TransformDecodeFP8 || ts.Quantize != "mxfp8" || !slices.Equal(ts.OutShape, []int32{8, 512, 256}) {
		t.Errorf("pre-stacked fp8 expert = %+v, want decode_fp8 + mxfp8 + 3D shape", ts)
	}
}

func TestPlanBlockFP8PerExpertStacked(t *testing.T) {
	// Disjoint per-expert fp8 weights are grouped by projection, stacked into one
	// [experts, out, in] tensor, and decoded+quantized together. Sources are the
	// N weights followed by the N scales, in expert-index order.
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"model.layers.0.mlp.experts.0.gate_proj.weight":       {Name: "model.layers.0.mlp.experts.0.gate_proj.weight", Dtype: "F8_E4M3", Shape: []int32{256, 256}},
		"model.layers.0.mlp.experts.0.gate_proj.weight_scale": {Name: "model.layers.0.mlp.experts.0.gate_proj.weight_scale", Dtype: "BF16", Shape: []int32{2, 2}},
		"model.layers.0.mlp.experts.1.gate_proj.weight":       {Name: "model.layers.0.mlp.experts.1.gate_proj.weight", Dtype: "F8_E4M3", Shape: []int32{256, 256}},
		"model.layers.0.mlp.experts.1.gate_proj.weight_scale": {Name: "model.layers.0.mlp.experts.1.gate_proj.weight_scale", Dtype: "BF16", Shape: []int32{2, 2}},
	}}
	specs, err := Plan(inv, Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("want single stacked blob; got %v", specNames(specs))
	}
	spec, ok := specByName(specs, "model.layers.0.mlp.experts")
	if !ok || len(spec.Tensors) != 1 {
		t.Fatalf("want one stacked expert group; got %v", specNames(specs))
	}
	ts := spec.Tensors[0]
	if ts.Name != "model.layers.0.mlp.experts.gate_proj.weight" {
		t.Errorf("stacked name = %q, want model.layers.0.mlp.experts.gate_proj.weight", ts.Name)
	}
	if ts.Transform != TransformDecodeStackFP8 || ts.Quantize != "mxfp8" || ts.OutDtype != "F8_E4M3" || !slices.Equal(ts.OutShape, []int32{2, 256, 256}) {
		t.Errorf("stacked fp8 expert = %+v, want decode_stack_fp8 + mxfp8 + F8_E4M3 + [2 256 256]", ts)
	}
	wantSources := []string{
		"model.layers.0.mlp.experts.0.gate_proj.weight",
		"model.layers.0.mlp.experts.1.gate_proj.weight",
		"model.layers.0.mlp.experts.0.gate_proj.weight_scale",
		"model.layers.0.mlp.experts.1.gate_proj.weight_scale",
	}
	if len(ts.Sources) != len(wantSources) {
		t.Fatalf("sources = %d, want %d (N weights then N scales)", len(ts.Sources), len(wantSources))
	}
	for i, want := range wantSources {
		if ts.Sources[i].Name != want {
			t.Errorf("source[%d] = %q, want %q", i, ts.Sources[i].Name, want)
		}
	}
}

func TestPlanBlockFP8MixedMXFP4PerExpert(t *testing.T) {
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		// A regular block-FP8 tensor keeps the overall source on the block-FP8
		// import path.
		"model.layers.0.self_attn.q_proj.weight":           {Name: "model.layers.0.self_attn.q_proj.weight", Dtype: "F8_E4M3", Shape: []int32{256, 256}},
		"model.layers.0.self_attn.q_proj.weight_scale_inv": {Name: "model.layers.0.self_attn.q_proj.weight_scale_inv", Dtype: "F8_E8M0", Shape: []int32{2, 2}},

		// Ling's routed experts pack two E2M1 values in each I8 byte and store
		// one E8M0 scale per 32 logical input values.
		"model.layers.0.mlp.experts.0.gate_proj.weight":           {Name: "model.layers.0.mlp.experts.0.gate_proj.weight", Dtype: "I8", Shape: []int32{768, 1280}},
		"model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": {Name: "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv", Dtype: "F8_E8M0", Shape: []int32{768, 80}},
		"model.layers.0.mlp.experts.1.gate_proj.weight":           {Name: "model.layers.0.mlp.experts.1.gate_proj.weight", Dtype: "I8", Shape: []int32{768, 1280}},
		"model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv": {Name: "model.layers.0.mlp.experts.1.gate_proj.weight_scale_inv", Dtype: "F8_E8M0", Shape: []int32{768, 80}},
	}}

	specs, err := Plan(inv, Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	group, ok := specByName(specs, "model.layers.0.mlp.experts")
	if !ok {
		t.Fatalf("missing MXFP4 expert group; got %v", specNames(specs))
	}
	if group.Metadata["quant_type"] != "mxfp4" || group.Metadata["group_size"] != "32" {
		t.Fatalf("metadata = %v, want mxfp4 group_size=32", group.Metadata)
	}

	weight, ok := inputByOutput(group, "model.layers.0.mlp.experts.gate_proj.weight")
	if !ok {
		t.Fatal("missing stacked MXFP4 weight")
	}
	if weight.Transform != TransformStackExperts || weight.OutDtype != "U32" ||
		!slices.Equal(weight.OutShape, []int32{2, 768, 320}) {
		t.Errorf("weight = %+v, want stacked U32 [2 768 320]", weight)
	}

	scale, ok := inputByOutput(group, "model.layers.0.mlp.experts.gate_proj.weight.scale")
	if !ok {
		t.Fatal("missing stacked MXFP4 scale")
	}
	if scale.Transform != TransformStackExperts || scale.OutDtype != "U8" ||
		!slices.Equal(scale.OutShape, []int32{2, 768, 80}) {
		t.Errorf("scale = %+v, want stacked U8 [2 768 80]", scale)
	}

	if _, leaked := specByName(specs, "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"); leaked {
		t.Error("MXFP4 scale companion must not be emitted as its own blob")
	}
}

func TestPlanBlockFP8ErrorsOnMissingScale(t *testing.T) {
	// An F8 weight without a scale companion must fail loudly instead of
	// passing through to the float path as raw FP8 bytes.
	inv := blockFP8(map[string]string{
		"model.layers.0.self_attn.q_proj.weight": "F8_E4M3",
	})
	if _, err := Plan(inv, Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, defaultQuantPolicy{}); err == nil {
		t.Fatal("Plan() = nil error, want missing-scale error")
	}
}

func TestPlanBlockFP8ErrorsOnInvalidMXFP4Pair(t *testing.T) {
	// A byte-packed weight with an E8M0 companion of the wrong shape must
	// fail validation instead of silently falling through to the float path.
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"model.layers.3.mlp.experts.0.gate_proj.weight": {
			Name: "model.layers.3.mlp.experts.0.gate_proj.weight", Dtype: "I8", Shape: []int32{64, 64},
		},
		"model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv": {
			Name: "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv", Dtype: "F8_E8M0", Shape: []int32{64, 3},
		},
	}}
	if _, err := Plan(inv, Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, defaultQuantPolicy{}); err == nil {
		t.Fatal("Plan() = nil error, want mxfp4 shape validation error")
	}
}
