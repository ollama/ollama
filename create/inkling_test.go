package create

import (
	"bytes"
	"context"
	"io"
	"path/filepath"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func TestInkling2ExpertBankImport(t *testing.T) {
	for _, tc := range []struct {
		mode, bank, weightDtype, scaleDtype, groupSize string
		width                                          int32
	}{
		{"nvfp4", "w13_weight", "U8", "F8_E4M3", "16", 16},
		{"mxfp8", "w2_weight", "F8_E4M3", "U8", "32", 64},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			bank := "model.llm.layers.6.mlp.experts." + tc.bank
			weights := make([]byte, 2*tc.width)
			for i := range weights {
				weights[i] = byte(i*7 + 5)
			}
			scales := []byte{1, 9, 127, 254}
			tensors := []*st.TensorData{
				st.NewTensorDataFromBytes(bank, tc.weightDtype, []int32{2, 1, tc.width}, weights),
				st.NewTensorDataFromBytes(bank+"_scale", tc.scaleDtype, []int32{2, 1, 2}, scales),
			}
			if tc.mode == "nvfp4" {
				tensors = append(tensors,
					st.NewTensorDataFromBytes(bank+"_amax", "F32", []int32{2}, encodeFloat32s(3, 5)),
					st.NewTensorDataFromBytes(bank+"_outer_scale_denominator", "F32", []int32{1}, f32le(7)),
				)
			}
			dir := t.TempDir()
			writeConfigJSON(t, dir, `{"architectures":["Inkling2LoopForConditionalGeneration"],"text_config":{"model_max_length":32768}}`)
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)
			inv, err := ReadInventory(dir)
			if err != nil {
				t.Fatal(err)
			}
			class, err := Classify(inv, "")
			if err != nil || class != (Classification{Kind: SourcePrequantized, Quantize: tc.mode}) {
				t.Fatalf("Classify() = %+v, %v; want prequantized %s", class, err, tc.mode)
			}
			if _, err := Classify(inv, tc.mode); err == nil || !strings.Contains(err.Error(), "cannot requantize") {
				t.Fatalf("Classify(%q) error = %v, want already-quantized source error", tc.mode, err)
			}
			store := newCaptureStore()
			writeManifest := func(context.Context, string, ManifestInfo) error { return nil }
			if err := Create(t.Context(), "inkling2", dir, testPipelineOptions(), store, writeManifest, func(string) {}); err != nil {
				t.Fatal(err)
			}
			want := map[string]io.Reader{
				"config.json": strings.NewReader(`{"architectures":["Inkling2LoopForConditionalGeneration"],"text_config":{"max_position_embeddings":32768,"model_max_length":32768}}`),
				bank: st.BuildPackedSafetensorsReaderWithMetadata([]*st.TensorData{
					st.NewTensorDataFromBytes(bank, "U32", []int32{2, 1, tc.width / 4}, weights),
					st.NewTensorDataFromBytes(bank+"_scale", "U8", []int32{2, 1, 2}, scales),
				}, map[string]string{"quant_type": tc.mode, "group_size": tc.groupSize}),
			}
			for _, calibration := range tensors[2:] {
				want[calibration.Name] = st.BuildPackedSafetensorsReader([]*st.TensorData{calibration})
			}
			if len(store.blobs) != len(want) {
				t.Fatalf("got blobs %v, want %d blobs", store.names(), len(want))
			}
			for name, reader := range want {
				expected, err := io.ReadAll(reader)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(store.blobs[name], expected) {
					t.Errorf("imported %s differs from the expected safetensors header or payload", name)
				}
			}
		})
	}
}

func TestCreateInkling2RejectsTargetQuantization(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Inkling2LoopForConditionalGeneration"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.llm.layers.0.attn.wq_du.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
	})
	for _, quantize := range []string{"nvfp4", "mxfp4", "mxfp8", "int4", "int8"} {
		t.Run(quantize, func(t *testing.T) {
			opts := testPipelineOptions()
			opts.Quantize = quantize
			store := newCaptureStore()
			err := Create(t.Context(), "inkling2", dir, opts, store, func(context.Context, string, ManifestInfo) error {
				t.Error("unexpected manifest write")
				return nil
			}, func(string) {})
			if err == nil || !strings.Contains(err.Error(), "without --quantize") {
				t.Fatalf("Create() error = %v, want advice to omit --quantize", err)
			}
			if len(store.blobs) != 0 {
				t.Fatalf("Create() wrote blobs before rejecting target quantization: %v", store.names())
			}
		})
	}
}

func TestInkling2DraftQuantization(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Inkling2ForConditionalGenerationMTP"]}`)
	const name = "layers.0.self_attn.q_proj.weight"
	weights := make([]byte, 64*64*2)
	tensor := st.NewTensorDataFromBytes(name, "BF16", []int32{64, 64}, weights)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{tensor})
	validation := testPipelineOptions().Validation
	for _, quantize := range []string{"nvfp4", "mxfp4", "mxfp8", "int4", "int8"} {
		t.Run(quantize, func(t *testing.T) {
			store := newCaptureStore()
			_, err := CreateDraftLayers(t.Context(), dir, "draft.", "draft/", quantize, validation, store, func(string) {})
			if err == nil || !strings.Contains(err.Error(), "without --draft-quantize") {
				t.Fatalf("CreateDraftLayers(%q) error = %v, want advice to omit --draft-quantize", quantize, err)
			}
			if len(store.blobs) != 0 {
				t.Fatalf("CreateDraftLayers(%q) wrote blobs: %v", quantize, store.names())
			}
		})
	}
	store := newCaptureStore()
	if _, err := CreateDraftLayers(t.Context(), dir, "draft.", "draft/", "", validation, store, func(string) {}); err != nil {
		t.Fatal(err)
	}
	const draftName = "draft." + name
	want, err := io.ReadAll(st.BuildPackedSafetensorsReader([]*st.TensorData{tensor.WithName(draftName)}))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(store.blobs[draftName], want) {
		t.Errorf("raw import changed %s", draftName)
	}
}

func TestInkling2BankScope(t *testing.T) {
	for _, tc := range []struct{ arch, weight string }{
		{"Qwen3ForCausalLM", "model.llm.layers.6.mlp.experts.w13_weight"},
		{inkling2Architecture, "model.llm.layers.6.mlp_sconv.down_weight"},
	} {
		t.Run(tc.arch, func(t *testing.T) {
			inv := newInventory(sourceModelConfig{Architectures: []string{tc.arch}}, map[string]string{
				tc.weight: "U8", tc.weight + "_scale": "F8_E4M3",
			})
			weight := inv.Tensors[tc.weight]
			weight.Shape = []int32{2, 1, 16}
			inv.Tensors[tc.weight] = weight
			class, err := Classify(inv, "")
			if err != nil || class.Kind != SourceFloat {
				t.Fatalf("Classify() = %+v, %v; want float", class, err)
			}
			// A different, ordinary prequantized weight in the same inventory
			// must not cause the underscore names to fuse during planning.
			inv.Tensors["linear.weight"] = SourceTensor{Name: "linear.weight", Dtype: "U32"}
			inv.Tensors["linear.scales"] = SourceTensor{Name: "linear.scales", Dtype: "BF16"}
			class, err = Classify(inv, "")
			if err != nil || class.Kind != SourcePrequantized {
				t.Fatalf("Classify() with MLX scales = %+v, %v; want prequantized", class, err)
			}
			specs, err := Plan(inv, class, nil)
			if err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{tc.weight, tc.weight + "_scale"} {
				blob, ok := specByName(specs, name)
				if !ok || len(blob.Tensors) != 1 || blob.Tensors[0].Transform != TransformNone || blob.Metadata != nil {
					t.Errorf("blob %q = %+v, want unchanged source tensor", name, blob)
				}
			}
		})
	}
}

func TestInkling2RejectsMalformedPackedBank(t *testing.T) {
	const bank = "model.llm.layers.6.mlp.experts.w13_weight"
	for _, shape := range [][]int32{{1, 16}, {1, 1, 7}} {
		inv := newInventory(sourceModelConfig{Architectures: []string{inkling2Architecture}}, map[string]string{
			bank: "U8", bank + "_scale": "F8_E4M3",
		})
		weight := inv.Tensors[bank]
		weight.Shape = shape
		inv.Tensors[bank] = weight
		for _, request := range []string{"", "int4"} {
			if _, err := Classify(inv, request); err == nil {
				t.Errorf("Classify() accepted packed bank of shape %v for quantization %q", shape, request)
			}
		}
	}
}
