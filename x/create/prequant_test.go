package create

import (
	"encoding/json"
	"maps"
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

// mlxQuantConfig parses a config.json "quantization" object the way the
// importer does, so these tests exercise the real JSON shape MLX writes rather
// than a hand-built struct.
func mlxQuantConfig(t *testing.T, quantization string) sourceModelConfig {
	t.Helper()
	var cfg sourceModelConfig
	if err := json.Unmarshal([]byte(`{"quantization": `+quantization+`}`), &cfg); err != nil {
		t.Fatalf("parse config: %v", err)
	}
	return cfg
}

func TestPlanPrequantizedMLXImplicitAffineMode(t *testing.T) {
	// mlx-lm omits "mode" for its default affine quantization. Without it the
	// blob used to carry no quant_type at all, and the runtime cannot infer
	// one: a 4-bit group-64 weight and an 8-bit group-32 weight pack to the
	// same shapes.
	inv := newInventory(mlxQuantConfig(t, `{"group_size": 64, "bits": 4}`), map[string]string{
		"l.weight": "U32",
		"l.scales": "BF16",
		"l.biases": "BF16",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	w, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	if w.Metadata["quant_type"] != "int4" || w.Metadata["group_size"] != "64" {
		t.Errorf("metadata = %v, want quant_type=int4 group_size=64", w.Metadata)
	}
}

func TestPlanPrequantizedMLXMixedPrecision(t *testing.T) {
	// MLX records per-module overrides beside the defaults, keyed by module
	// path, and fills in what an override omits from the mode's defaults
	// (affine: 4 bits, group size 64) rather than from the model-wide values.
	cfg := mlxQuantConfig(t, `{
		"group_size": 64,
		"bits": 4,
		"model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32},
		"model.layers.0.self_attn.k_proj": {"group_size": 32},
		"model.layers.0.mlp.experts": {"bits": 4, "group_size": 64},
		"model.layers.0.mlp.up_proj": {"bits": 8}
	}`)
	inv := newInventory(cfg, map[string]string{
		"model.layers.0.self_attn.q_proj.weight": "U32",
		"model.layers.0.self_attn.q_proj.scales": "BF16",
		"model.layers.0.self_attn.q_proj.biases": "BF16",
		"model.layers.0.self_attn.k_proj.weight": "U32",
		"model.layers.0.self_attn.k_proj.scales": "BF16",
		"model.layers.0.self_attn.k_proj.biases": "BF16",
		"model.layers.0.mlp.experts.weight":      "U32",
		"model.layers.0.mlp.experts.scales":      "BF16",
		"model.layers.0.mlp.experts.biases":      "BF16",
		"model.layers.0.mlp.up_proj.weight":      "U32",
		"model.layers.0.mlp.up_proj.scales":      "BF16",
		"model.layers.0.mlp.up_proj.biases":      "BF16",
		"model.layers.0.mlp.gate_proj.weight":    "U32",
		"model.layers.0.mlp.gate_proj.scales":    "BF16",
		"model.layers.0.mlp.gate_proj.biases":    "BF16",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	tests := []struct {
		blob          string
		wantQuantType string
		wantGroupSize string
	}{
		// Promoted to 8 bits at a different group size.
		{"model.layers.0.self_attn.q_proj.weight", "int8", "32"},
		// Regrouped at the model's bit width: only the group size differs.
		{"model.layers.0.self_attn.k_proj.weight", "", "32"},
		// Promoted to 8 bits at MLX's default group size, which happens to be
		// the model's: only the bit width differs.
		{"model.layers.0.mlp.up_proj.weight", "int8", ""},
		// Overridden, but to what the blob already says.
		{"model.layers.0.mlp.experts.weight", "", ""},
		// No override at all.
		{"model.layers.0.mlp.gate_proj.weight", "", ""},
	}
	for _, tt := range tests {
		w, ok := specByName(specs, tt.blob)
		if !ok {
			t.Errorf("missing %s blob", tt.blob)
			continue
		}
		if w.Metadata["quant_type"] != "int4" || w.Metadata["group_size"] != "64" {
			t.Errorf("%s: blob defaults = %v, want quant_type=int4 group_size=64", tt.blob, w.Metadata)
		}
		if got := w.Metadata[tt.blob+".quant_type"]; got != tt.wantQuantType {
			t.Errorf("%s: per-tensor quant_type = %q, want %q", tt.blob, got, tt.wantQuantType)
		}
		if got := w.Metadata[tt.blob+".group_size"]; got != tt.wantGroupSize {
			t.Errorf("%s: per-tensor group_size = %q, want %q", tt.blob, got, tt.wantGroupSize)
		}
	}
}

func TestPlanPrequantizedUnmappableQuantization(t *testing.T) {
	// A bit width we cannot map records no quant_type, as before: the runtime
	// falls back to inferring one from the shapes.
	inv := newInventory(mlxQuantConfig(t, `{"group_size": 64, "bits": 3}`), map[string]string{
		"l.weight": "U32",
		"l.scales": "BF16",
		"l.biases": "BF16",
	})

	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	w, ok := specByName(specs, "l.weight")
	if !ok {
		t.Fatal("missing l.weight blob")
	}
	if len(w.Metadata) != 0 {
		t.Errorf("metadata = %v, want none", w.Metadata)
	}
}

func TestSourceQuantizationModules(t *testing.T) {
	tests := []struct {
		name         string
		quantization string
		want         map[string]moduleQuantization
	}{
		{
			name:         "mlx overrides",
			quantization: `{"group_size": 64, "bits": 4, "model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32}}`,
			want: map[string]moduleQuantization{
				"model.layers.0.self_attn.q_proj": {Bits: 8, GroupSize: 32},
			},
		},
		{
			name:         "model-wide fields only",
			quantization: `{"group_size": 64, "bits": 4, "mode": "mxfp4"}`,
		},
		{
			// compressed-tensors nests objects of its own; they are not
			// module overrides and must not be read as any.
			name: "compressed-tensors config groups",
			quantization: `{
				"quant_method": "compressed-tensors",
				"format": "nvfp4-pack-quantized",
				"config_groups": {"group_0": {"format": "nvfp4-pack-quantized", "weights": {"num_bits": 4, "type": "float", "group_size": 16}}},
				"kv_cache_scheme": {"num_bits": 8, "type": "float"}
			}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mlxQuantConfig(t, tt.quantization).Quantization.Modules
			if !maps.Equal(got, tt.want) {
				t.Errorf("modules = %v, want %v", got, tt.want)
			}
		})
	}
}
