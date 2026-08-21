package create

import (
	"encoding/json"
	"slices"
	"strconv"
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

func compressedInt4TestConfig(t *testing.T) sourceModelConfig {
	t.Helper()
	var cfg sourceModelConfig
	err := json.Unmarshal([]byte(`{
		"architectures":["BailingMoeV3ForCausalLM"],
		"quantization_config":{
			"quant_method":"compressed-tensors",
			"format":"pack-quantized",
			"config_groups":{"group_0":{"weights":{
				"num_bits":4,"type":"int","symmetric":true,
				"strategy":"group","group_size":32
			}}}
		}
	}`), &cfg)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
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
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("got %d specs %v, want 1 (input_global_scale must be consumed)", len(specs), specNames(specs))
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
	if w.Metadata["quant_type"] != "nvfp4" || w.Metadata["group_size"] != "16" {
		t.Errorf("metadata = %v, want quant_type=nvfp4 group_size=16", w.Metadata)
	}
}

func TestPlanPrequantizedCompressedInt4(t *testing.T) {
	inv := Inventory{Dir: "test", Config: compressedInt4TestConfig(t), Tensors: map[string]SourceTensor{
		"linear.weight_packed": {Name: "linear.weight_packed", Dtype: "I32", Shape: []int32{2, 4}, File: "model.safetensors"},
		"linear.weight_scale":  {Name: "linear.weight_scale", Dtype: "BF16", Shape: []int32{2, 1}, File: "model.safetensors"},
		"linear.weight_shape":  {Name: "linear.weight_shape", Dtype: "I64", Shape: []int32{2}, File: "model.safetensors"},
	}}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	if len(specs) != 1 || specs[0].Name != "linear.weight" {
		t.Fatalf("specs = %v, want one linear.weight blob", specNames(specs))
	}
	w := specs[0]
	weight, ok := inputByOutput(w, "linear.weight")
	if !ok || weight.Transform != TransformRelabelU32 || weight.OutDtype != "U32" || !slices.Equal(weight.OutShape, []int32(nil)) {
		t.Errorf("weight = %+v ok=%v, want I32 words relabeled U32", weight, ok)
	}
	scale, ok := inputByOutput(w, "linear.weight.scale")
	if !ok || scale.Transform != TransformNone || scale.OutDtype != "" {
		t.Errorf("scale = %+v ok=%v, want BF16 pass-through", scale, ok)
	}
	bias, ok := inputByOutput(w, "linear.weight.bias")
	if !ok || bias.Transform != TransformInt4SymmetricQBias || bias.OutDtype != "BF16" {
		t.Errorf("bias = %+v ok=%v, want derived BF16 qbias", bias, ok)
	}
	if w.Metadata["quant_type"] != "int4" || w.Metadata["group_size"] != "32" {
		t.Errorf("metadata = %v, want int4 group_size=32", w.Metadata)
	}
	if _, leaked := specByName(specs, "linear.weight_shape"); leaked {
		t.Error("weight_shape companion leaked as its own blob")
	}
}

func TestPlanPrequantizedCompressedInt4StacksExperts(t *testing.T) {
	const group = "model.layers.1.mlp.experts"
	tensors := map[string]SourceTensor{
		"norm.weight": {Name: "norm.weight", Dtype: "BF16", Shape: []int32{2}, File: "model.safetensors"},
	}
	for expert := 0; expert < 2; expert++ {
		base := group + "." + strconv.Itoa(expert) + ".gate_proj"
		tensors[base+".weight_packed"] = SourceTensor{Name: base + ".weight_packed", Dtype: "I32", Shape: []int32{2, 4}, File: "model.safetensors"}
		tensors[base+".weight_scale"] = SourceTensor{Name: base + ".weight_scale", Dtype: "BF16", Shape: []int32{2, 1}, File: "model.safetensors"}
		tensors[base+".weight_shape"] = SourceTensor{Name: base + ".weight_shape", Dtype: "I64", Shape: []int32{2}, File: "model.safetensors"}
	}
	inv := Inventory{Dir: "test", Config: compressedInt4TestConfig(t), Tensors: tensors}

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	experts, ok := specByName(specs, group)
	if !ok {
		t.Fatalf("missing stacked expert blob; got %v", specNames(specs))
	}
	weight, _ := inputByOutput(experts, group+".gate_proj.weight")
	if weight.Transform != TransformStackExperts || weight.OutDtype != "U32" || !slices.Equal(weight.OutShape, []int32{2, 2, 4}) || len(weight.Sources) != 2 {
		t.Errorf("stacked weight = %+v, want U32 [2 2 4] from two experts", weight)
	}
	scale, _ := inputByOutput(experts, group+".gate_proj.weight.scale")
	if scale.Transform != TransformStackExperts || !slices.Equal(scale.OutShape, []int32{2, 2, 1}) || len(scale.Sources) != 2 {
		t.Errorf("stacked scale = %+v, want BF16 [2 2 1]", scale)
	}
	bias, _ := inputByOutput(experts, group+".gate_proj.weight.bias")
	if bias.Transform != TransformInt4SymmetricQBias || !slices.Equal(bias.OutShape, []int32{2, 2, 1}) || len(bias.Sources) != 2 {
		t.Errorf("stacked bias = %+v, want derived BF16 [2 2 1]", bias)
	}
	if _, ok := specByName(specs, "norm.weight"); !ok {
		t.Error("plain norm tensor should pass through")
	}
}
