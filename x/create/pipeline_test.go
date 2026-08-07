package create

import (
	"context"
	"errors"
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
	writeManifest := func(name string, info ManifestInfo) error {
		gotName, gotConfig, gotLayers = name, info.Config, info.Layers
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
	if len(gotLayers) != 3 {
		t.Fatalf("manifest layers = %d, want 3 (2 tensors + config.json)", len(gotLayers))
	}
	for _, n := range []string{"model.embed_tokens.weight", "model.norm.weight", "config.json"} {
		if _, ok := store.blobs[n]; !ok {
			t.Errorf("missing written blob %q (have %v)", n, store.names())
		}
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

func TestCreatePipelineRejectsNilContext(t *testing.T) {
	var ctx context.Context
	err := Create(ctx, "mymodel", t.TempDir(), "", newCaptureStore(), func(string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || err.Error() != "nil context" {
		t.Fatalf("Create() error = %v, want nil context", err)
	}
}
