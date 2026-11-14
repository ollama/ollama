package create

import (
	"encoding/binary"
	"io"
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

// A ModelOpt mixed-precision NemotronH checkpoint: NVFP4 routed experts with
// scalar global scales, per-tensor FP8 projections, BF16 attention, and a BF16
// MTP expert group. Every encoding is planned on its own, and every per-expert
// group is stacked regardless of architecture.
func TestPlanMixedPrecisionStacksEveryExpertGroup(t *testing.T) {
	tensors := map[string]string{
		"backbone.layers.0.mixer.in_proj.weight":       "F8_E4M3",
		"backbone.layers.0.mixer.in_proj.weight_scale": "F32",
		"backbone.layers.0.mixer.in_proj.input_scale":  "F32",
		"backbone.layers.1.mixer.q_proj.weight":        "BF16",
		"mtp.layers.0.mixer.experts.0.up_proj.weight":  "BF16",
		"mtp.layers.0.mixer.experts.1.up_proj.weight":  "BF16",
	}
	shapes := map[string][]int32{
		"backbone.layers.0.mixer.in_proj.weight_scale": {},
		"backbone.layers.0.mixer.in_proj.input_scale":  {1},
	}
	for _, e := range []string{"0", "1"} {
		for _, p := range []string{"up_proj", "down_proj"} {
			base := "backbone.layers.2.mixer.experts." + e + "." + p
			tensors[base+".weight"] = "U8"
			tensors[base+".weight_scale"] = "F8_E4M3"
			tensors[base+".weight_scale_2"] = "F32"
			tensors[base+".input_scale"] = "F32"
			shapes[base+".weight_scale"] = []int32{128, 8}
			shapes[base+".weight_scale_2"] = []int32{}
			shapes[base+".input_scale"] = []int32{}
		}
	}
	inv := withShapes(newInventory(sourceModelConfig{Architectures: []string{"NemotronHForCausalLM"}}, tensors), shapes)

	class, err := Classify(inv, "")
	if err != nil || class.Kind != SourcePrequantized || class.Quantize != "nvfp4" {
		t.Fatalf("Classify() = %+v, %v; want prequantized nvfp4", class, err)
	}
	specs, err := Plan(inv, class, defaultQuantPolicy{})
	if err != nil {
		t.Fatal(err)
	}
	if got := specNames(specs); !slices.Equal(got, []string{
		"backbone.layers.0.mixer.in_proj.weight",
		"backbone.layers.1.mixer.q_proj.weight",
		"backbone.layers.2.mixer.experts",
		"mtp.layers.0.mixer.experts",
	}) {
		t.Fatalf("blobs = %v", got)
	}

	fp8, _ := specByName(specs, "backbone.layers.0.mixer.in_proj.weight")
	checkExact(t, fp8, "backbone.layers.0.mixer.in_proj.weight", TransformNone, []int32{128, 32}, []int32{128, 4}, []int32{})

	attn, _ := specByName(specs, "backbone.layers.1.mixer.q_proj.weight")
	if ts := attn.Tensors[0]; ts.Quantize != "" || ts.Transform != TransformNone {
		t.Errorf("BF16 attention = %+v, want source precision (the publisher excluded it)", ts)
	}

	bank, _ := specByName(specs, "backbone.layers.2.mixer.experts")
	if bank.Metadata["quant_type"] != "nvfp4" {
		t.Errorf("bank metadata = %v", bank.Metadata)
	}
	for _, want := range []struct {
		name  string
		each  Transform
		dtype string
		shape []int32
	}{
		{"backbone.layers.2.mixer.experts.down_proj.weight", TransformRepackFP4, "U32", []int32{2, 128, 32}},
		{"backbone.layers.2.mixer.experts.down_proj.weight.scale", TransformRelabelU8, "U8", []int32{2, 128, 8}},
		{"backbone.layers.2.mixer.experts.down_proj.weight.global_scale", TransformScalarF32, "F32", []int32{2}},
		{"backbone.layers.2.mixer.experts.up_proj.weight", TransformRepackFP4, "U32", []int32{2, 128, 32}},
		{"backbone.layers.2.mixer.experts.up_proj.weight.scale", TransformRelabelU8, "U8", []int32{2, 128, 8}},
		{"backbone.layers.2.mixer.experts.up_proj.weight.global_scale", TransformScalarF32, "F32", []int32{2}},
	} {
		ts, ok := inputByOutput(bank, want.name)
		if !ok || ts.Transform != TransformStackExperts || ts.Each != want.each || ts.OutDtype != want.dtype ||
			!slices.Equal(ts.OutShape, want.shape) || len(ts.Sources) != 2 {
			t.Errorf("%s = %+v ok=%v, want stacked %s %v (each %q)", want.name, ts, ok, want.dtype, want.shape, want.each)
		}
	}
	if len(bank.Tensors) != 6 {
		t.Errorf("bank has %d tensors, want 6 (activation scales dropped)", len(bank.Tensors))
	}

	mtp, _ := specByName(specs, "mtp.layers.0.mixer.experts")
	if ts, ok := inputByOutput(mtp, "mtp.layers.0.mixer.experts.up_proj.weight"); !ok || ts.Transform != TransformStackExperts || ts.Quantize != "" {
		t.Errorf("BF16 MTP experts = %+v ok=%v, want stacked at source precision", ts, ok)
	}
}

func TestPlanStacksCompressedTensorsExperts(t *testing.T) {
	tensors := map[string]string{}
	shapes := map[string][]int32{}
	for _, e := range []string{"0", "1"} {
		base := "model.layers.0.mlp.experts." + e + ".gate_proj"
		tensors[base+".weight_packed"] = "U8"
		tensors[base+".weight_scale"] = "F8_E4M3"
		tensors[base+".weight_global_scale"] = "F32"
		tensors[base+".input_global_scale"] = "F32"
		shapes[base+".weight_scale"] = []int32{128, 16}
		shapes[base+".weight_global_scale"] = []int32{1}
		shapes[base+".input_global_scale"] = []int32{1}
	}
	inv := withShapes(newInventory(sourceModelConfig{Architectures: []string{"LagunaForCausalLM"}}, tensors), shapes)
	specs, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{})
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || specs[0].Name != "model.layers.0.mlp.experts" {
		t.Fatalf("blobs = %v, want one bank", specNames(specs))
	}
	global, ok := inputByOutput(specs[0], "model.layers.0.mlp.experts.gate_proj.weight.global_scale")
	if !ok || global.Each != TransformReciprocalF32 || !slices.Equal(global.OutShape, []int32{2}) ||
		global.Sources[0].Name != "model.layers.0.mlp.experts.0.gate_proj.weight_global_scale" {
		t.Errorf("global scale bank = %+v ok=%v, want each expert's reciprocal stacked to [2]", global, ok)
	}
}

func TestPlanRejectsExpertGroupMixingEncodings(t *testing.T) {
	inv := withShapes(newInventory(sourceModelConfig{}, map[string]string{
		"model.layers.0.mlp.experts.0.up_proj.weight":         "U8",
		"model.layers.0.mlp.experts.0.up_proj.weight_scale":   "F8_E4M3",
		"model.layers.0.mlp.experts.0.up_proj.weight_scale_2": "F32",
		"model.layers.0.mlp.experts.1.up_proj.weight":         "BF16",
	}), map[string][]int32{"model.layers.0.mlp.experts.0.up_proj.weight_scale_2": {}})
	if _, err := Plan(inv, Classification{Kind: SourcePrequantized}, defaultQuantPolicy{}); err == nil {
		t.Fatal("Plan() succeeded on an expert group mixing nvfp4 and bf16")
	}
}

func f32Tensor(name string, shape []int32, values ...float32) *safetensors.TensorData {
	raw := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	return safetensors.NewTensorDataFromBytes(name, "F32", shape, raw)
}

func TestStackExpertsAppliesEachTransform(t *testing.T) {
	ts := TensorSpec{Name: "bank.global_scale", Transform: TransformStackExperts, Each: TransformReciprocalF32, OutDtype: "F32", OutShape: []int32{2}}
	out, err := applyByteTransform(ts, []*safetensors.TensorData{f32Tensor("e0", nil, 4), f32Tensor("e1", []int32{1}, 0.5)})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(out.Reader())
	if err != nil {
		t.Fatal(err)
	}
	got := []float32{math.Float32frombits(binary.LittleEndian.Uint32(raw)), math.Float32frombits(binary.LittleEndian.Uint32(raw[4:]))}
	if !slices.Equal(got, []float32{0.25, 2}) || !slices.Equal(out.Shape, []int32{2}) {
		t.Errorf("stacked reciprocals = %v %v, want [0.25 2] [2]", got, out.Shape)
	}
}
