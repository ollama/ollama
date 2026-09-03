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

// newInventoryWithShapes creates a test inventory where each tensor has the
// given dtype and shape. If a name appears in shapes it uses that; otherwise
// it falls back to a default 2-D shape.
func newInventoryWithShapes(cfg sourceModelConfig, tensors map[string]string, shapes map[string][]int32) Inventory {
	m := make(map[string]SourceTensor)
	for name, dtype := range tensors {
		shape := shapes[name]
		if shape == nil {
			shape = []int32{128, 128}
		}
		m[name] = SourceTensor{Name: name, Dtype: dtype, Shape: shape, File: "model.safetensors"}
	}
	return Inventory{Dir: "test", Config: cfg, Tensors: m}
}

func TestPlanPrequantizedMLXSplitGateUp(t *testing.T) {
	// MLX-converted GraniteMoe: gate_proj and up_proj are stored separately.
	// Plan should fuse them into a single input_linear.weight blob.
	E, I, H := int32(4), int32(8), int32(16)
	scaleH := int32(2) // H / group_size
	cfg := sourceModelConfig{Quantization: sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 8}}
	inv := newInventoryWithShapes(cfg, map[string]string{
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight":  "U32",
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales":  "BF16",
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.biases":  "BF16",
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight":    "U32",
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.scales":    "BF16",
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.biases":    "BF16",
		"model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight":  "U32",
		"model.layers.0.block_sparse_moe.switch_mlp.down_proj.scales":  "BF16",
		"norm.weight": "BF16",
	}, map[string][]int32{
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight": {E, I, H},
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales": {E, I, scaleH},
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.biases": {E, I, scaleH},
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight":   {E, I, H},
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.scales":   {E, I, scaleH},
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.biases":   {E, I, scaleH},
		"model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight": {E, H, I},
		"model.layers.0.block_sparse_moe.switch_mlp.down_proj.scales": {E, H, 1},
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	fusedName := "model.layers.0.block_sparse_moe.input_linear.weight"
	downName := "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"

	for _, want := range []string{fusedName, downName, "norm.weight"} {
		if _, ok := specByName(specs, want); !ok {
			t.Errorf("missing blob %q in %v", want, specNames(specs))
		}
	}

	// gate_proj and up_proj must not appear as their own blobs
	for _, unwanted := range []string{
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight",
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight",
	} {
		if _, ok := specByName(specs, unwanted); ok {
			t.Errorf("blob %q must not appear after fusion", unwanted)
		}
	}

	fusedSpec, _ := specByName(specs, fusedName)

	weightTS, ok := inputByOutput(fusedSpec, fusedName)
	if !ok {
		t.Fatalf("fused blob missing weight tensor %q", fusedName)
	}
	if weightTS.Transform != TransformConcatAxis1 {
		t.Errorf("weight transform = %q, want concat_axis1", weightTS.Transform)
	}
	if !slices.Equal(weightTS.OutShape, []int32{E, 2 * I, H}) {
		t.Errorf("weight OutShape = %v, want [%d %d %d]", weightTS.OutShape, E, 2*I, H)
	}
	if len(weightTS.Sources) != 2 {
		t.Fatalf("weight sources len = %d, want 2", len(weightTS.Sources))
	}
	if weightTS.Sources[0].Name != "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight" {
		t.Errorf("weight source[0] = %q, want gate_proj", weightTS.Sources[0].Name)
	}
	if weightTS.Sources[1].Name != "model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight" {
		t.Errorf("weight source[1] = %q, want up_proj", weightTS.Sources[1].Name)
	}

	// Scale companions must be fused
	scaleTS, ok := inputByOutput(fusedSpec, fusedName+".scale")
	if !ok {
		t.Fatal("fused blob missing scale tensor")
	}
	if scaleTS.Transform != TransformConcatAxis1 {
		t.Errorf("scale transform = %q, want concat_axis1", scaleTS.Transform)
	}
	if !slices.Equal(scaleTS.OutShape, []int32{E, 2 * I, scaleH}) {
		t.Errorf("scale OutShape = %v, want [%d %d %d]", scaleTS.OutShape, E, 2*I, scaleH)
	}

	// Bias companions must be fused
	biasTS, ok := inputByOutput(fusedSpec, fusedName+".bias")
	if !ok {
		t.Fatal("fused blob missing bias tensor")
	}
	if biasTS.Transform != TransformConcatAxis1 {
		t.Errorf("bias transform = %q, want concat_axis1", biasTS.Transform)
	}
}

func TestPlanPrequantizedMLXSplitGateUpNoBias(t *testing.T) {
	// No biases companion — fuse should still work.
	E, I, H := int32(2), int32(4), int32(8)
	cfg := sourceModelConfig{Quantization: sourceQuantization{Bits: 4, Mode: "affine", GroupSize: 8}}
	inv := newInventoryWithShapes(cfg, map[string]string{
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight": "U32",
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales": "BF16",
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight":   "U32",
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.scales":   "BF16",
	}, map[string][]int32{
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight": {E, I, H},
		"model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales": {E, I, H / 8},
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight":   {E, I, H},
		"model.layers.0.block_sparse_moe.switch_mlp.up_proj.scales":   {E, I, H / 8},
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	fusedName := "model.layers.0.block_sparse_moe.input_linear.weight"
	if _, ok := specByName(specs, fusedName); !ok {
		t.Errorf("missing fused blob %q", fusedName)
	}
	fusedSpec, _ := specByName(specs, fusedName)
	if _, ok := inputByOutput(fusedSpec, fusedName+".bias"); ok {
		t.Error("fused blob should not have a bias when source has none")
	}
}
