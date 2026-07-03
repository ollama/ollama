package create

import (
	"context"
	"errors"
	"io"
	"os"
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
	writeManifest := func(name string, info ManifestInfo) error {
		gotName, gotConfig, gotLayers = name, info.Config, info.Layers
		gotClass = info.Class
		return nil
	}

	if err := Create(context.Background(), "mymodel", dir, "", store, writeManifest, func(string) {}); err != nil {
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
	writeManifest := func(_ string, info ManifestInfo) error {
		got = info.Class
		return nil
	}
	if err := Create(context.Background(), "mymodel", dir, "", store, writeManifest, func(string) {}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if got.Kind != SourcePrequantized || got.Quantize != "nvfp4" {
		t.Errorf("classification = {%s %q}, want {prequantized nvfp4}", got.Kind, got.Quantize)
	}
}

func TestCreatePipelineReturnsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := Create(ctx, "mymodel", t.TempDir(), "", newCaptureStore(), func(string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Create() error = %v, want context.Canceled", err)
	}
}

func TestCreatePipelineDoesNotPublishAfterCancellation(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})
	if err := os.WriteFile(filepath.Join(dir, "z.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	store := newCaptureStore()
	cancelingStore := StoreFromLayerCreator(func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		layer, err := store.WriteBlob(r, mediaType, name)
		if name == "z.json" {
			cancel()
		}
		return layer, err
	})
	manifestCalled := false
	err := Create(ctx, "mymodel", dir, "", cancelingStore, func(string, ManifestInfo) error {
		manifestCalled = true
		return nil
	}, func(string) {})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Create() error = %v, want context.Canceled", err)
	}
	if manifestCalled {
		t.Fatal("Create() published a manifest after cancellation")
	}
}

func TestCreatePipelineRejectsNilContext(t *testing.T) {
	var ctx context.Context
	err := Create(ctx, "mymodel", t.TempDir(), "", newCaptureStore(), func(string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || err.Error() != "nil context" {
		t.Fatalf("Create() error = %v, want nil context", err)
	}
}
