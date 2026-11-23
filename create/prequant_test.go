package create

import (
	"slices"
	"strings"
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

func TestPlanPrequantizedFlintEmbeddingBits(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
		want string
	}{
		{name: "int8 embedding", raw: `{"mlx_export":{"embedding_bits":8}}`, want: "int8"},
		{name: "int4 embedding", raw: `{"mlx_export":{"embedding_bits":4}}`, want: "int4"},
		{name: "missing bit width", raw: `{"mlx_export":{}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inv := newInventory(sourceModelConfig{
				Architectures: []string{"FlintForCausalLM"},
				Quantization:  sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 64},
			}, map[string]string{
				"model.embed_tokens.weight":              "U32",
				"model.embed_tokens.scales":              "BF16",
				"model.embed_tokens.biases":              "BF16",
				"model.layers.0.self_attn.q_proj.weight": "U32",
				"model.layers.0.self_attn.q_proj.scales": "BF16",
				"model.layers.0.self_attn.q_proj.biases": "BF16",
			})
			inv.RawConfig = []byte(tc.raw)
			specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
			if tc.want == "" {
				if err == nil || !strings.Contains(err.Error(), "mlx_export.embedding_bits") {
					t.Fatalf("Plan() error = %v, want missing embedding bit width", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			embedding, ok := specByName(specs, "model.embed_tokens.weight")
			if !ok || embedding.Metadata["quant_type"] != tc.want {
				t.Errorf("embedding quant_type = %q, want %q", embedding.Metadata["quant_type"], tc.want)
			}
			projection, ok := specByName(specs, "model.layers.0.self_attn.q_proj.weight")
			if !ok || projection.Metadata["quant_type"] != "int4" {
				t.Errorf("projection quant_type = %q, want int4", projection.Metadata["quant_type"])
			}
		})
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

func TestPlanPrequantizedModelOptExpertBank(t *testing.T) {
	inv := newInventory(sourceModelConfig{Architectures: []string{"FlintForCausalLM"}}, map[string]string{
		"model.layers.2.mlp.experts.0.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale":   "U8",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale_2": "F32",
		"model.layers.2.mlp.experts.1.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.1.gate_proj.weight_scale":   "U8",
		"model.layers.2.mlp.experts.1.gate_proj.weight_scale_2": "F32",
	})
	for name, tensor := range inv.Tensors {
		if slices.Contains([]string{"model.layers.2.mlp.experts.0.gate_proj.weight_scale_2", "model.layers.2.mlp.experts.1.gate_proj.weight_scale_2"}, name) {
			tensor.Shape = []int32{1}
		}
		inv.Tensors[name] = tensor
	}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got blobs %v, want one expert bank", specNames(specs))
	}
	name := "model.layers.2.mlp.experts.gate_proj.weight"
	bank := specs[0]
	if bank.Name != "model.layers.2.mlp.experts" || bank.Metadata["quant_type"] != "nvfp4" {
		t.Fatalf("expert bank = %+v", bank)
	}
	for _, tc := range []struct {
		name  string
		shape []int32
	}{
		{name, []int32{2, 128, 128}},
		{name + ".scale", []int32{2, 128, 128}},
		{name + ".global_scale", []int32{2}},
	} {
		tensor, ok := inputByOutput(bank, tc.name)
		if !ok || tensor.Transform != TransformStackExperts || !slices.Equal(tensor.OutShape, tc.shape) || len(tensor.Sources) != 2 {
			t.Errorf("%s = %+v, want stacked %v", tc.name, tensor, tc.shape)
		}
	}
}

func TestPlanPrequantizedModelOptExpertBankRejectsIncompleteScale(t *testing.T) {
	inv := newInventory(sourceModelConfig{Architectures: []string{"FlintForCausalLM"}}, map[string]string{
		"model.layers.2.mlp.experts.0.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale":   "U8",
		"model.layers.2.mlp.experts.0.gate_proj.weight_scale_2": "F32",
		"model.layers.2.mlp.experts.1.gate_proj.weight":         "U32",
		"model.layers.2.mlp.experts.1.gate_proj.weight_scale":   "U8",
	})
	global := inv.Tensors["model.layers.2.mlp.experts.0.gate_proj.weight_scale_2"]
	global.Shape = []int32{1}
	inv.Tensors[global.Name] = global

	_, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err == nil || !strings.Contains(err.Error(), "model.layers.2.mlp.experts") {
		t.Fatalf("Plan() error = %v, want named incomplete expert bank", err)
	}
}

func TestPlanPrequantizedModelOptMixedPrecision(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"fp4.weight":         "U8",
		"fp4.weight_scale":   "F8_E4M3",
		"fp4.weight_scale_2": "F32",
		"fp8.weight":         "F8_E4M3",
		"fp8.weight_scale":   "U8",
	})
	classification, err := Classify(inv, "")
	if err != nil || classification.Kind != SourcePrequantized || classification.Quantize != "" {
		t.Fatalf("mixed ModelOpt classification = %+v, %v", classification, err)
	}

	specs, err := Plan(inv, classification, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 2 {
		t.Fatalf("got %d specs %v, want two fused weights", len(specs), specNames(specs))
	}
	fp4, _ := specByName(specs, "fp4.weight")
	if fp4.Metadata["quant_type"] != "nvfp4" {
		t.Errorf("FP4 metadata = %v", fp4.Metadata)
	}
	fp8, _ := specByName(specs, "fp8.weight")
	if fp8.Metadata["quant_type"] != "mxfp8" || fp8.Metadata["group_size"] != "32" {
		t.Errorf("FP8 metadata = %v", fp8.Metadata)
	}
	weight, _ := inputByOutput(fp8, "fp8.weight")
	scale, _ := inputByOutput(fp8, "fp8.weight.scale")
	if weight.Transform != TransformRepackFP8 || weight.OutDtype != "U32" || !slices.Equal(weight.OutShape, []int32{128, 32}) || scale.Transform != TransformNone || scale.OutDtype != "" {
		t.Errorf("FP8 weight must pack into U32 and exponent scale must keep source bytes: weight=%+v scale=%+v", weight, scale)
	}
	if q := detectPrequantizedQuantization(inv); q != "" {
		t.Errorf("mixed precision classification = %q, want no single file type", q)
	}
}

func TestClassifyModelOptMXFP8WithoutNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":       "F8_E4M3",
		"l.weight_scale": "U8",
	})
	classification, err := Classify(inv, "")
	if err != nil || classification.Kind != SourcePrequantized || classification.Quantize != "mxfp8" {
		t.Fatalf("ModelOpt MXFP8 classification = %+v, %v", classification, err)
	}
	specs, err := Plan(inv, classification, defaultQuantPolicy{})
	if err != nil || len(specs) != 1 || specs[0].Metadata["quant_type"] != "mxfp8" {
		t.Fatalf("ModelOpt MXFP8 plan = %+v, %v", specs, err)
	}
}

func TestPlanPrequantizedModelOptPackedNVFP4(t *testing.T) {
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":         "U32",
		"l.weight_scale":   "F8_E4M3",
		"l.weight_scale_2": "F32",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want one fused weight", len(specs), specNames(specs))
	}
	weight, ok := inputByOutput(specs[0], "l.weight")
	if !ok || weight.Transform != TransformNone || weight.OutDtype != "" || len(weight.OutShape) != 0 {
		t.Errorf("already-packed U32 weight must keep its source shape: %+v ok=%v", weight, ok)
	}
	if specs[0].Metadata["quant_type"] != "nvfp4" {
		t.Errorf("quant_type = %q, want nvfp4", specs[0].Metadata["quant_type"])
	}
}

func TestPlanPrequantizedModelOptDropsActivationScale(t *testing.T) {
	// ModelOpt ships per-weight activation scales (.input_scale and, in some
	// variants, .input_global_scale) that are unused for weight-only
	// inference. They must be consumed, not emitted as their own blobs.
	inv := newInventory(sourceModelConfig{}, map[string]string{
		"l.weight":             "U8",
		"l.weight_scale":       "F8_E4M3",
		"l.weight_scale_2":     "F32",
		"l.input_scale":        "F32",
		"l.input_global_scale": "F32",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1 (activation scales must not become blobs)", len(specs), specNames(specs))
	}
	w := specs[0]
	for _, act := range []string{"l.input_scale", "l.input_global_scale"} {
		if _, leaked := inputByOutput(w, act); leaked {
			t.Errorf("activation scale %s leaked into the fused blob", act)
		}
		for _, s := range specs {
			if s.Name == act {
				t.Errorf("activation scale %s emitted as its own blob", act)
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
	if len(w.Tensors) != 3 {
		t.Errorf("fused tensors = %v, want only weight, block scale and global scale", w.Tensors)
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
	scale, ok := inputByOutput(w, "l.weight.global_scale")
	if !ok || scale.Transform != TransformF32 || !slices.Equal(scale.Sources[0].Shape, []int32{128}) {
		t.Errorf("global scale = %+v ok=%v, want F32 expert scale bank", scale, ok)
	}
	if len(specs) != 1 || len(w.Tensors) != 3 {
		t.Errorf("specs = %v, want only fused weight, block scale and global scale", specs)
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
