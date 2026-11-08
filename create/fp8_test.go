package create

import (
	"encoding/binary"
	"io"
	"maps"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/safetensors"
)

// fp8Inventory gives FP8 and BF16 weights a realistic [256, 256] shape and
// every other tensor (the scales) one value per tensor.
func fp8Inventory(tensors map[string]string) Inventory {
	m := make(map[string]SourceTensor)
	for name, dtype := range tensors {
		shape := []int32{256, 256}
		if dtype != "F8_E4M3" && dtype != "BF16" {
			shape = []int32{}
		}
		m[name] = SourceTensor{Name: name, Dtype: dtype, Shape: shape}
	}
	return Inventory{Dir: "test", Tensors: m}
}

func planFP8(t *testing.T, inv Inventory, policy quantizePolicy) []BlobSpec {
	t.Helper()
	specs, err := Plan(inv, Classification{Kind: SourceFP8, Quantize: fp8DecodeTarget}, policy)
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}
	return specs
}

// checkExact asserts blob is the exact mxfp8 form of an FP8 weight: the FP8
// bytes as U32 words, E8M0 block scales and an F32 global scale, each built
// from the given sources with the given transforms.
func checkExact(t *testing.T, blob BlobSpec, name string, transform Transform, weight, scales, global []int32) {
	t.Helper()
	want := []struct {
		name  string
		dtype string
		shape []int32
		each  Transform
	}{
		{name, "U32", weight, TransformRepackFP8},
		{name + ".scale", "U8", scales, TransformFP8ScaleExponent},
		{name + ".global_scale", "F32", global, TransformFP8ScaleRemainder},
	}
	if len(blob.Tensors) != len(want) {
		t.Fatalf("%s: tensors %v, want weight, scale and global scale", blob.Name, blob.Tensors)
	}
	for i, w := range want {
		ts := blob.Tensors[i]
		gotTransform, gotEach := ts.Transform, ts.Each
		wantTransform, wantEach := w.each, TransformNone
		if transform == TransformStackExperts {
			wantTransform, wantEach = TransformStackExperts, w.each
		}
		if ts.Name != w.name || ts.OutDtype != w.dtype || !slices.Equal(ts.OutShape, w.shape) || gotTransform != wantTransform || gotEach != wantEach || ts.Quantize != "" {
			t.Errorf("tensor %d = %s %s %v %s/%s q=%q, want %s %s %v %s/%s", i, ts.Name, ts.OutDtype, ts.OutShape, gotTransform, gotEach, ts.Quantize, w.name, w.dtype, w.shape, wantTransform, wantEach)
		}
	}
	if !maps.Equal(blob.Metadata, map[string]string{"quant_type": "mxfp8", "group_size": "32"}) {
		t.Errorf("%s metadata = %v, want mxfp8 group 32", blob.Name, blob.Metadata)
	}
}

func TestPlanFP8Exact(t *testing.T) {
	inv := fp8Inventory(map[string]string{
		"model.layers.0.self_attn.q_proj.weight":       "F8_E4M3",
		"model.layers.0.self_attn.q_proj.weight_scale": "F32",
		"model.layers.0.self_attn.q_proj.input_scale":  "F32",
		"model.layers.0.input_layernorm.weight":        "F32",
		"lm_head.weight":                               "BF16",
	})
	specs := planFP8(t, inv, defaultQuantPolicy{})

	q, ok := specByName(specs, "model.layers.0.self_attn.q_proj.weight")
	if !ok {
		t.Fatalf("missing q_proj blob; got %v", specNames(specs))
	}
	checkExact(t, q, "model.layers.0.self_attn.q_proj.weight", TransformNone, []int32{256, 64}, []int32{256, 8}, []int32{})
	for _, name := range []string{"model.layers.0.self_attn.q_proj.weight_scale", "model.layers.0.self_attn.q_proj.input_scale"} {
		if _, leaked := specByName(specs, name); leaked {
			t.Errorf("%s must not be its own blob", name)
		}
	}

	norm, _ := specByName(specs, "model.layers.0.input_layernorm.weight")
	if norm.Tensors[0].Quantize != "" || norm.Tensors[0].Transform != TransformNone {
		t.Errorf("norm = %+v, want kept at source precision", norm.Tensors[0])
	}
	lmHead, _ := specByName(specs, "lm_head.weight")
	if lmHead.Tensors[0].Quantize != "" || lmHead.Tensors[0].Transform != TransformNone {
		t.Errorf("lm_head = %+v, want kept at source precision", lmHead.Tensors[0])
	}
}

func TestPlanFP8PerChannelRejected(t *testing.T) {
	for _, tt := range []struct {
		name               string
		weight, scaleShape []int32
	}{
		{"dense [rows, 1]", []int32{512, 256}, []int32{512, 1}},
		{"dense [rows]", []int32{512, 256}, []int32{512}},
		{"stacked bank", []int32{8, 512, 256}, []int32{8, 512, 1}},
	} {
		inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
			"model.layers.0.mlp.down_proj.weight":       {Name: "model.layers.0.mlp.down_proj.weight", Dtype: "F8_E4M3", Shape: tt.weight},
			"model.layers.0.mlp.down_proj.weight_scale": {Name: "model.layers.0.mlp.down_proj.weight_scale", Dtype: "BF16", Shape: tt.scaleShape},
		}}
		_, err := Plan(inv, Classification{Kind: SourceFP8, Quantize: fp8DecodeTarget}, defaultQuantPolicy{})
		if err == nil || !strings.Contains(err.Error(), "per-channel FP8 import is not supported yet") {
			t.Errorf("%s: Plan() error = %v, want per-channel FP8 rejected", tt.name, err)
		}
	}
}

func TestPlanFP8ExactPrestackedExperts(t *testing.T) {
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"model.layers.0.mlp.experts.gate_up_proj.weight":       {Name: "model.layers.0.mlp.experts.gate_up_proj.weight", Dtype: "F8_E4M3", Shape: []int32{8, 512, 256}},
		"model.layers.0.mlp.experts.gate_up_proj.weight_scale": {Name: "model.layers.0.mlp.experts.gate_up_proj.weight_scale", Dtype: "F32", Shape: []int32{8}},
	}}
	specs := planFP8(t, inv, defaultQuantPolicy{})
	if len(specs) != 1 {
		t.Fatalf("want one blob, got %v", specNames(specs))
	}
	checkExact(t, specs[0], "model.layers.0.mlp.experts.gate_up_proj.weight", TransformNone, []int32{8, 512, 64}, []int32{8, 512, 8}, []int32{8})
}

func TestPlanFP8ExactPerExpertStacked(t *testing.T) {
	tensors := make(map[string]SourceTensor)
	for _, e := range []string{"0", "1"} {
		for _, proj := range []string{"gate_proj", "down_proj"} {
			w := "model.layers.0.mlp.experts." + e + "." + proj + ".weight"
			tensors[w] = SourceTensor{Name: w, Dtype: "F8_E4M3", Shape: []int32{256, 256}}
			tensors[w+"_scale"] = SourceTensor{Name: w + "_scale", Dtype: "F32", Shape: []int32{}}
		}
	}
	specs := planFP8(t, Inventory{Dir: "test", Tensors: tensors}, defaultQuantPolicy{})
	bank, ok := specByName(specs, "model.layers.0.mlp.experts")
	if !ok || len(specs) != 1 || len(bank.Tensors) != 6 {
		t.Fatalf("want one bank of two projections, got %v", specNames(specs))
	}
	for i, proj := range []string{"down_proj", "gate_proj"} {
		part := BlobSpec{Name: bank.Name, Tensors: bank.Tensors[3*i : 3*i+3], Metadata: bank.Metadata}
		name := "model.layers.0.mlp.experts." + proj + ".weight"
		checkExact(t, part, name, TransformStackExperts, []int32{2, 256, 64}, []int32{2, 256, 8}, []int32{2})
		if src := part.Tensors[0].Sources; len(src) != 2 || src[0].Name != "model.layers.0.mlp.experts.0."+proj+".weight" || src[1].Name != "model.layers.0.mlp.experts.1."+proj+".weight" {
			t.Errorf("%s sources = %v, want experts 0 and 1 in order", name, src)
		}
	}
}

func TestPlanFP8BlockScalesRejected(t *testing.T) {
	inv := Inventory{Dir: "test", Tensors: map[string]SourceTensor{
		"model.layers.0.mlp.down_proj.weight":           {Name: "model.layers.0.mlp.down_proj.weight", Dtype: "F8_E4M3", Shape: []int32{256, 256}},
		"model.layers.0.mlp.down_proj.weight_scale_inv": {Name: "model.layers.0.mlp.down_proj.weight_scale_inv", Dtype: "BF16", Shape: []int32{2, 2}},
	}}
	_, err := Plan(inv, Classification{Kind: SourceFP8, Quantize: fp8DecodeTarget}, defaultQuantPolicy{})
	if err == nil || !strings.Contains(err.Error(), "block-scaled FP8 cannot be stored without re-rounding") {
		t.Fatalf("Plan() error = %v, want block-scaled FP8 rejected", err)
	}
}

// fixedPolicy answers every request with one quantization type.
type fixedPolicy string

func (p fixedPolicy) quantizationType(string, []int32, string) string { return string(p) }

func TestPlanFP8NeverRequantizes(t *testing.T) {
	single := fp8Inventory(map[string]string{
		"model.layers.0.mlp.down_proj.weight":       "F8_E4M3",
		"model.layers.0.mlp.down_proj.weight_scale": "F32",
	})
	experts := fp8Inventory(map[string]string{
		"model.layers.0.mlp.experts.0.down_proj.weight":       "F8_E4M3",
		"model.layers.0.mlp.experts.0.down_proj.weight_scale": "F32",
	})

	// Kept at source precision: widened to BF16, not quantized.
	for _, inv := range []Inventory{single, experts} {
		specs := planFP8(t, inv, fixedPolicy(""))
		ts := specs[0].Tensors[0]
		if (ts.Transform != TransformDecodeFP8 && ts.Transform != TransformDecodeStackFP8) || ts.Quantize != "" {
			t.Errorf("declined fp8 weight = %+v, want widened to BF16", ts)
		}
	}

	// Anything else would quantize the FP8 values a second time.
	for _, inv := range []Inventory{single, experts} {
		_, err := Plan(inv, Classification{Kind: SourceFP8, Quantize: fp8DecodeTarget}, fixedPolicy("nvfp4"))
		if err == nil || !strings.Contains(err.Error(), "would be re-quantized as nvfp4") {
			t.Errorf("Plan() error = %v, want re-quantization refused", err)
		}
	}
}

func TestPlanFP8InvalidExpertIndices(t *testing.T) {
	tests := []struct {
		name    string
		tensors map[string]string
		wantErr string
	}{
		{
			name: "gap",
			tensors: map[string]string{
				"model.layers.0.mlp.experts.0.gate_proj.weight":       "F8_E4M3",
				"model.layers.0.mlp.experts.0.gate_proj.weight_scale": "F32",
				"model.layers.0.mlp.experts.2.gate_proj.weight":       "F8_E4M3",
				"model.layers.0.mlp.experts.2.gate_proj.weight_scale": "F32",
			},
			wantErr: "missing expert index 1",
		},
		{
			name: "projection count mismatch",
			tensors: map[string]string{
				"model.layers.0.mlp.experts.0.down_proj.weight":       "F8_E4M3",
				"model.layers.0.mlp.experts.0.down_proj.weight_scale": "F32",
				"model.layers.0.mlp.experts.0.gate_proj.weight":       "F8_E4M3",
				"model.layers.0.mlp.experts.0.gate_proj.weight_scale": "F32",
				"model.layers.0.mlp.experts.1.gate_proj.weight":       "F8_E4M3",
				"model.layers.0.mlp.experts.1.gate_proj.weight_scale": "F32",
			},
			wantErr: "want 1 to match the other projections",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Plan(fp8Inventory(tt.tensors), Classification{Kind: SourceFP8, Quantize: fp8DecodeTarget}, defaultQuantPolicy{})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("Plan() error = %v, want error containing %q", err, tt.wantErr)
			}
		})
	}
}

func f32Bytes(values ...float32) []byte {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

func TestFP8ScaleSplit(t *testing.T) {
	read := func(t *testing.T, td *safetensors.TensorData) []byte {
		t.Helper()
		b, err := io.ReadAll(td.Reader())
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	// Four one-row experts: 0.75 = 1.5 * 2^-1, 6 = 1.5 * 2^2, 1 = 1 * 2^0, and 0
	// keeps a zero remainder.
	scale := safetensors.NewTensorDataFromBytes("s", "F32", []int32{4}, f32Bytes(0.75, 6, 1, 0))

	exps, err := fp8ScaleExponents(scale, "w.scale", []int32{4, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := read(t, exps), []byte{126, 126, 129, 129, 127, 127, 127, 127}; !slices.Equal(got, want) || exps.Dtype != "U8" {
		t.Errorf("block scales = %s %v, want U8 %v", exps.Dtype, got, want)
	}
	scale = safetensors.NewTensorDataFromBytes("s", "F32", []int32{4}, f32Bytes(0.75, 6, 1, 0))
	rems, err := fp8ScaleRemainders(scale, "w.global_scale", []int32{4})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := read(t, rems), f32Bytes(1.5, 1.5, 1, 0); !slices.Equal(got, want) || !slices.Equal(rems.Shape, []int32{4}) {
		t.Errorf("global scale = %v %v, want %v", rems.Shape, got, want)
	}

	// BF16 and F16 scales decode to the same split (1.5 * 2^-1 is exact in both).
	for _, dtype := range []string{"BF16", "F16"} {
		raw := []byte{0x40, 0x3f} // BF16 0.75
		if dtype == "F16" {
			raw = []byte{0x00, 0x3a} // F16 0.75
		}
		td := safetensors.NewTensorDataFromBytes("s", dtype, []int32{}, raw)
		exps, err := fp8ScaleExponents(td, "w.scale", []int32{2, 1})
		if err != nil {
			t.Fatal(err)
		}
		if got := read(t, exps); !slices.Equal(got, []byte{126, 126}) {
			t.Errorf("%s block scales = %v, want [126 126]", dtype, got)
		}
	}

	// A stacked bank: one scale per expert spans that expert's rows.
	bank := safetensors.NewTensorDataFromBytes("s", "F32", []int32{2}, f32Bytes(2, 0.5))
	exps, err = fp8ScaleExponents(bank, "w.scale", []int32{2, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got := read(t, exps); !slices.Equal(got, []byte{128, 128, 126, 126}) {
		t.Errorf("per-expert block scales = %v, want [128 128 126 126]", got)
	}

	for _, bad := range []float32{-1, float32(math.Inf(1)), float32(math.NaN()), 1e-39} {
		td := safetensors.NewTensorDataFromBytes("s", "F32", []int32{}, f32Bytes(bad))
		if _, err := fp8ScaleExponents(td, "w.scale", []int32{1, 1}); err == nil {
			t.Errorf("scale %v was accepted", bad)
		}
	}
}
