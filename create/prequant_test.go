package create

import (
	"slices"
	"testing"
)

func specByName(specs []BlobSpec, name string) (BlobSpec, bool) {
	for _, s := range specs {
		if s.Name == name {
			return s, true
		}
	}
	return BlobSpec{}, false
}

func inputByOutput(spec BlobSpec, outputName string) (TensorSpec, bool) {
	for _, ts := range spec.Tensors {
		if ts.Name == outputName {
			return ts, true
		}
	}
	return TensorSpec{}, false
}

// sourceName returns the (single) source tensor name for a TensorSpec.
func sourceName(ts TensorSpec) string {
	if len(ts.Sources) == 0 {
		return ""
	}
	return ts.Sources[0].Name
}

func specNames(specs []BlobSpec) []string {
	names := make([]string, len(specs))
	for i, s := range specs {
		names[i] = s.Name
	}
	return names
}

func TestPlanPrequantizedMLX(t *testing.T) {
	cfg := sourceModelConfig{Quantization: sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 32}}
	inv := newInventory(cfg, map[string]string{
		"l.weight":    "U32",
		"l.scales":    "BF16",
		"l.biases":    "BF16",
		"norm.weight": "BF16",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	// l.weight (fused with scales+biases) and norm.weight (pass-through).
	if len(specs) != 2 {
		t.Fatalf("got %d specs %v, want 2", len(specs), specNames(specs))
	}

	w, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	for _, want := range []string{"l.weight", "l.weight.scale", "l.weight.bias"} {
		in, ok := inputByOutput(w, want)
		if !ok {
			t.Fatalf("l.weight blob missing input %q", want)
		}
		if in.Transform != TransformNone {
			t.Errorf("%s transform = %q, want none", want, in.Transform)
		}
	}
	if w.Metadata["quant_type"] != "int4" || w.Metadata["group_size"] != "32" {
		t.Errorf("metadata = %v, want quant_type=int4 group_size=32 from config", w.Metadata)
	}
	if _, ok := specByName(specs, "norm.weight"); !ok {
		t.Error("norm.weight should pass through as its own blob")
	}
}

func TestPlanPrequantizedModelOptNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":         "U8",
		"l.weight_scale":   "F8_E4M3",
		"l.weight_scale_2": "F32",
	})
	global := inv.Tensors["l.weight_scale_2"]
	global.Shape = nil
	inv.Tensors[global.Name] = global

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1", len(specs), specNames(specs))
	}
	w := specs[0]
	if w.Name != "l.weight" {
		t.Fatalf("blob name = %q, want l.weight", w.Name)
	}

	weightIn, _ := inputByOutput(w, "l.weight")
	if weightIn.Transform != TransformRepackFP4 || weightIn.OutDtype != "U32" || !slices.Equal(weightIn.OutShape, []int32{128, 32}) {
		t.Errorf("weight input = %+v, want repack to U32 [128 32]", weightIn)
	}
	scaleIn, _ := inputByOutput(w, "l.weight.scale")
	if scaleIn.Transform != TransformRelabelU8 || scaleIn.OutDtype != "U8" {
		t.Errorf("scale input = %+v, want relabel to U8", scaleIn)
	}
	globalIn, ok := inputByOutput(w, "l.weight.global_scale")
	if !ok || globalIn.Transform != TransformScalarF32 {
		t.Errorf("global_scale input = %+v ok=%v, want scalar_f32 (stored as-is)", globalIn, ok)
	}
	if w.Metadata["quant_type"] != "nvfp4" {
		t.Errorf("quant_type = %q, want nvfp4", w.Metadata["quant_type"])
	}
	if _, ok := w.Metadata["group_size"]; ok {
		t.Errorf("ModelOpt should not default group_size: %v", w.Metadata)
	}
}

func TestPlanPrequantizedModelOptRetainsActivationScale(t *testing.T) {
	// Preserve calibration metadata for an explicitly enabled quantized-
	// activation path without changing the existing weight-only runtime path.
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":             "U8",
		"l.weight_scale":       "F8_E4M3",
		"l.weight_scale_2":     "F32",
		"l.input_scale":        "F32",
		"l.input_global_scale": "F32",
	})
	for _, name := range []string{"l.weight_scale_2", "l.input_scale", "l.input_global_scale"} {
		scale := inv.Tensors[name]
		scale.Shape = nil
		inv.Tensors[name] = scale
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1", len(specs), specNames(specs))
	}
	w := specs[0]
	for _, scale := range []struct {
		source string
		output string
	}{
		{source: "l.input_scale", output: "l.weight.input_scale"},
		{source: "l.input_global_scale", output: "l.weight.input_global_scale"},
	} {
		input, ok := inputByOutput(w, scale.output)
		if !ok || input.Transform != TransformScalarF32 || sourceName(input) != scale.source {
			t.Errorf("activation scale %s = %+v ok=%v, want scalar_f32 companion %s", scale.output, input, ok, scale.source)
		}
		for _, s := range specs {
			if s.Name == scale.source {
				t.Errorf("activation scale %s should be stored beside its weight", scale.source)
			}
		}
	}
}

func TestPlanPrequantizedCompressedNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight_packed":       "U8",
		"l.weight_scale":        "F8_E4M3",
		"l.weight_global_scale": "F32",
		"l.input_global_scale":  "F32",
		"l.input_scale":         "F32",
	})
	for _, name := range []string{"l.weight_global_scale", "l.input_global_scale", "l.input_scale"} {
		scale := inv.Tensors[name]
		scale.Shape = nil
		inv.Tensors[name] = scale
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1", len(specs), specNames(specs))
	}
	w := specs[0]
	if w.Name != "l.weight" {
		t.Fatalf("blob name = %q, want l.weight", w.Name)
	}

	weightIn, _ := inputByOutput(w, "l.weight")
	if sourceName(weightIn) != "l.weight_packed" || weightIn.Transform != TransformRepackFP4 {
		t.Errorf("weight input = %+v, want source l.weight_packed repacked", weightIn)
	}
	globalIn, ok := inputByOutput(w, "l.weight.global_scale")
	if !ok || globalIn.Transform != TransformReciprocalF32 {
		t.Errorf("global_scale input = %+v ok=%v, want reciprocal_f32", globalIn, ok)
	}
	inputGlobal, ok := inputByOutput(w, "l.weight.input_global_scale")
	if !ok || inputGlobal.Transform != TransformReciprocalF32 || sourceName(inputGlobal) != "l.input_global_scale" {
		t.Errorf("input_global_scale = %+v ok=%v, want reciprocal_f32 companion", inputGlobal, ok)
	}
	inputScale, ok := inputByOutput(w, "l.weight.input_scale")
	if !ok || inputScale.Transform != TransformScalarF32 || sourceName(inputScale) != "l.input_scale" {
		t.Errorf("input_scale = %+v ok=%v, want unchanged F32 companion", inputScale, ok)
	}
	if w.Metadata["quant_type"] != "nvfp4" || w.Metadata["group_size"] != "16" {
		t.Errorf("metadata = %v, want quant_type=nvfp4 group_size=16", w.Metadata)
	}
}

func TestPlanPrequantizedModelOptRetainsExpertScaleBanks(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":         "U32",
		"l.weight_scale":   "U8",
		"l.weight_scale_2": "F32",
		"l.input_scale":    "F32",
	})
	for _, name := range []string{"l.weight_scale_2", "l.input_scale"} {
		scale := inv.Tensors[name]
		scale.Shape = []int32{128}
		inv.Tensors[name] = scale
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	w, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	for _, name := range []string{"l.weight.global_scale", "l.weight.input_scale"} {
		scale, ok := inputByOutput(w, name)
		if !ok || scale.Transform != TransformF32 || !slices.Equal(scale.Sources[0].Shape, []int32{128}) {
			t.Errorf("%s = %+v ok=%v, want F32 expert scale bank", name, scale, ok)
		}
	}
}

func TestPlanPrequantizedRetainsKVCacheScales(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.k_proj.weight":       "U8",
		"l.k_proj.weight_scale": "F8_E4M3",
		"l.k_proj.k_scale":      "F32",
		"l.v_proj.weight":       "U8",
		"l.v_proj.weight_scale": "F8_E4M3",
		"l.v_proj.v_scale":      "F32",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	for _, name := range []string{"l.k_proj.k_scale", "l.v_proj.v_scale"} {
		spec, ok := specByName(specs, name)
		if !ok || len(spec.Tensors) != 1 || sourceName(spec.Tensors[0]) != name {
			t.Errorf("%s = %+v ok=%v, want unchanged standalone tensor", name, spec, ok)
		}
	}
}
