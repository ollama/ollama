package create

import (
	"path/filepath"
	"testing"

	st "github.com/ollama/ollama/x/safetensors"
)

func TestCreatePipeline(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 8*2)),
	})

	store := newCaptureStore()
	var gotName string
	var gotConfig LayerInfo
	var gotLayers []LayerInfo
	var gotClass Classification
	writeManifest := func(name string, config LayerInfo, layers []LayerInfo, class Classification) error {
		gotName, gotConfig, gotLayers = name, config, layers
		gotClass = class
		return nil
	}

	if err := Create("mymodel", dir, "", store, writeManifest, func(string) {}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	if gotName != "mymodel" {
		t.Errorf("manifest name = %q, want mymodel", gotName)
	}
	if gotConfig.Name != "config.json" {
		t.Errorf("config layer = %q, want config.json", gotConfig.Name)
	}
	if gotClass.Kind != SourceFloat || gotClass.Quantize != "" {
		t.Errorf("classification = {%s %q}, want {float %q}", gotClass.Kind, gotClass.Quantize, "")
	}
	if len(gotLayers) != 3 {
		t.Fatalf("manifest layers = %d, want 3 (2 tensors + config.json)", len(gotLayers))
	}
	for _, n := range []string{"model.embed_tokens.weight", "model.norm.weight", "config.json"} {
		if _, ok := store.blobs[n]; !ok {
			t.Errorf("missing written blob %q (have %v)", n, store.names())
		}
	}
}

func TestCreatePipelineReportsPrequantizedFileType(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("linear.weight", "U8", []int32{16, 8}, make([]byte, 16*8)),
		st.NewTensorDataFromBytes("linear.weight_scale", "F8_E4M3", []int32{16, 1}, make([]byte, 16)),
		st.NewTensorDataFromBytes("linear.weight_scale_2", "F32", []int32{}, f32le(1)),
	})

	store := newCaptureStore()
	var got Classification
	writeManifest := func(_ string, _ LayerInfo, _ []LayerInfo, class Classification) error {
		got = class
		return nil
	}
	if err := Create("mymodel", dir, "", store, writeManifest, func(string) {}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if got.Kind != SourcePrequantized || got.Quantize != "nvfp4" {
		t.Errorf("classification = {%s %q}, want {prequantized nvfp4}", got.Kind, got.Quantize)
	}
}
