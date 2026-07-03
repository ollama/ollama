package create

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
	st "github.com/ollama/ollama/x/safetensors"
)

type blobStoreFunc func(io.Reader, string, string) (LayerInfo, error)

func (f blobStoreFunc) WriteBlob(r io.Reader, mediaType, name string) (LayerInfo, error) {
	return f(r, mediaType, name)
}

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
	var gotModelConfig model.ConfigV2
	var gotLayers []LayerInfo
	var gotClass Classification
	writeManifest := func(_ context.Context, name string, info ManifestInfo) error {
		gotName, gotConfig, gotLayers = name, info.ConfigLayer, info.Layers
		gotModelConfig = info.ModelConfig
		gotClass = info.Class
		return nil
	}

	opts := testPipelineOptions()
	opts.Requires = "v0.20.0"
	if err := Create(context.Background(), "mymodel", dir, opts, store, writeManifest, func(string) {}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	if gotName != "mymodel" {
		t.Errorf("manifest name = %q, want mymodel", gotName)
	}
	if gotConfig.Name != "config.json" {
		t.Errorf("config layer = %q, want config.json", gotConfig.Name)
	}
	if gotModelConfig.ModelFormat != "safetensors" || !slices.Equal(gotModelConfig.Capabilities, []string{"completion"}) {
		t.Errorf("model config = %#v, want safetensors completion model", gotModelConfig)
	}
	if gotModelConfig.Requires != "0.20.0" {
		t.Errorf("requires = %q, want 0.20.0", gotModelConfig.Requires)
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

func TestCreatePipelineRejectsUnsupportedArchitectureBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["UnsupportedForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})

	store := newCaptureStore()
	manifestCalled := false
	err := Create(context.Background(), "mymodel", dir, PipelineOptions{}, store, func(context.Context, string, ManifestInfo) error {
		manifestCalled = true
		return nil
	}, func(string) {})
	if !errors.Is(err, ErrUnsupportedMLXArchitecture) {
		t.Fatalf("Create() error = %v, want ErrUnsupportedMLXArchitecture", err)
	}
	if len(store.blobs) != 0 {
		t.Fatalf("Create() wrote blobs before validation: %v", store.names())
	}
	if manifestCalled {
		t.Fatal("Create() published a manifest before validation")
	}
}

func TestCreatePipelineRejectsInvalidRequiresBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})

	store := newCaptureStore()
	opts := testPipelineOptions()
	opts.Requires = "not-semver"
	err := Create(context.Background(), "mymodel", dir, opts, store, func(context.Context, string, ManifestInfo) error {
		t.Fatal("Create() published a manifest with invalid requires")
		return nil
	}, func(string) {})
	if !errors.Is(err, ErrInvalidRequires) {
		t.Fatalf("Create() error = %v, want ErrInvalidRequires", err)
	}
	if len(store.blobs) != 0 {
		t.Fatalf("Create() wrote blobs before requires validation: %v", store.names())
	}
}

func TestCreatePipelineIncludesDraftLayers(t *testing.T) {
	modelDir := t.TempDir()
	writeConfigJSON(t, modelDir, `{"architectures":["Qwen3ForCausalLM"]}`)
	createTestSafetensors(t, filepath.Join(modelDir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})
	draftDir := t.TempDir()
	writeConfigJSON(t, draftDir, `{"architectures":["DFlashDraftModel"]}`)
	createTestSafetensors(t, filepath.Join(draftDir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
	})

	store := newCaptureStore()
	var layers []LayerInfo
	opts := testPipelineOptions()
	opts.DraftDir = draftDir
	err := Create(context.Background(), "mymodel", modelDir, opts, store, func(_ context.Context, _ string, info ManifestInfo) error {
		layers = info.Layers
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"model.norm.weight", "draft.model.norm.weight", "config.json", "draft/config.json"} {
		if !slices.ContainsFunc(layers, func(layer LayerInfo) bool { return layer.Name == name }) {
			t.Errorf("manifest layers are missing %q: %#v", name, layers)
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
	writeManifest := func(_ context.Context, _ string, info ManifestInfo) error {
		got = info.Class
		return nil
	}
	if err := Create(context.Background(), "mymodel", dir, testPipelineOptions(), store, writeManifest, func(string) {}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if got.Kind != SourcePrequantized || got.Quantize != "nvfp4" {
		t.Errorf("classification = {%s %q}, want {prequantized nvfp4}", got.Kind, got.Quantize)
	}
}

func TestCreatePipelineRejectsMalformedMetadata(t *testing.T) {
	tests := []struct {
		name  string
		file  string
		value string
	}{
		{name: "config", file: "config.json", value: "{"},
		{name: "tokenizer config", file: "tokenizer_config.json", value: "{"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			writeConfigJSON(t, dir, `{"architectures":["Qwen3ForCausalLM"]}`)
			createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
				st.NewTensorDataFromBytes("model.norm.weight", "BF16", []int32{8}, make([]byte, 16)),
			})
			if err := os.WriteFile(filepath.Join(dir, tt.file), []byte(tt.value), 0o600); err != nil {
				t.Fatal(err)
			}

			err := Create(context.Background(), "mymodel", dir, testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
				return nil
			}, func(string) {})
			if err == nil || !strings.Contains(err.Error(), "parse") {
				t.Fatalf("Create() error = %v, want parse error", err)
			}
		})
	}
}

func TestCreatePipelineReturnsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := Create(ctx, "mymodel", t.TempDir(), testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
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
	cancelingStore := blobStoreFunc(func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		layer, err := store.WriteBlob(r, mediaType, name)
		if name == "z.json" {
			cancel()
		}
		return layer, err
	})
	manifestCalled := false
	err := Create(ctx, "mymodel", dir, testPipelineOptions(), cancelingStore, func(context.Context, string, ManifestInfo) error {
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

func TestCreatePipelineStopsBlobReadAfterCancellation(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["TestModel"]}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 8}, make([]byte, 8*8*2)),
	})

	ctx, cancel := context.WithCancel(context.Background())
	var streamErr error
	store := blobStoreFunc(func(r io.Reader, mediaType, name string) (LayerInfo, error) {
		buf := make([]byte, 1)
		if _, err := r.Read(buf); err != nil {
			return LayerInfo{}, err
		}
		cancel()
		_, streamErr = io.Copy(io.Discard, r)
		return LayerInfo{}, streamErr
	})

	err := Create(ctx, "mymodel", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Create() error = %v, want context.Canceled", err)
	}
	if !errors.Is(streamErr, context.Canceled) {
		t.Fatalf("blob stream error = %v, want context.Canceled", streamErr)
	}
}

func TestCreatePipelineRejectsNilContext(t *testing.T) {
	var ctx context.Context
	err := Create(ctx, "mymodel", t.TempDir(), testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || err.Error() != "nil context" {
		t.Fatalf("Create() error = %v, want nil context", err)
	}
}

func testPipelineOptions() PipelineOptions {
	return PipelineOptions{Validation: MLXValidationOptions{
		Force:   true,
		Warning: func(string) {},
	}}
}
