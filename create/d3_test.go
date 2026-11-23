package create

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func writeD3Source(t *testing.T, dir string) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3_5Model"],"model_type":"qwen3_5","text_config":{"hidden_size":4},"vision_config":{}}`)
	decision := `{"format_version":1,"format_id":"d3-code-readout-v1","prompt":"d3","attention_mode":"noncausal_full_attention","pooling":"last","readout_dtype":"float32","temperature":1,"codes":[` + strings.TrimSuffix(strings.Repeat(`"A",`, 255), ",") + `],"token_ids":[` + strings.TrimSuffix(strings.Repeat(`32,`, 255), ",") + `]}`
	if err := os.WriteFile(filepath.Join(dir, "decision_config.json"), []byte(decision), 0o600); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 4}, make([]byte, 64)),
	})
	createTestSafetensors(t, filepath.Join(dir, "readout.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("weight", "F32", []int32{255, 4}, make([]byte, 4080)),
	})
}

func TestD3ImportRetainsFP32ReadoutAndExcludesPublisherManifest(t *testing.T) {
	dir := t.TempDir()
	writeD3Source(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "MODEL_MANIFEST.json"), []byte(`{"publisher":"vllm-sr"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	files, err := SafetensorsConfigFiles(dir)
	if err != nil || slices.Contains(files, "MODEL_MANIFEST.json") {
		t.Fatalf("imported publisher manifest: %v, %v", files, err)
	}
	store := newCaptureStore()
	var renderer string
	err = Create(t.Context(), "d3-mini-test", dir, testPipelineOptions(), store, func(_ context.Context, _ string, info ManifestInfo) error {
		renderer = info.ModelConfig.Renderer
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if renderer != "d3" || store.blobs["decision_config.json"] == nil || store.blobs["MODEL_MANIFEST.json"] != nil {
		t.Fatalf("renderer = %q; blobs = %v", renderer, store.names())
	}
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	inv, err = prepareD3Inventory(inv)
	if err != nil {
		t.Fatal(err)
	}
	if got := inv.Tensors["weight"].Dtype; got != "F32" {
		t.Fatalf("readout dtype = %q", got)
	}
}

func TestD3RejectsNonFP32ReadoutBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writeD3Source(t, dir)
	createTestSafetensors(t, filepath.Join(dir, "readout.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("weight", "BF16", []int32{255, 4}, make([]byte, 2040)),
	})
	store := newCaptureStore()
	err := Create(t.Context(), "d3-mini-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "FP32 readout") || len(store.blobs) != 0 {
		t.Fatalf("import error = %v; blobs = %v", err, store.names())
	}
}

func TestD3RejectsUnknownDecisionFormatBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writeD3Source(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "decision_config.json"), []byte(`{"format_id":"future-format"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	store := newCaptureStore()
	err := Create(t.Context(), "d3-mini-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), `unsupported Qwen decision format "future-format"`) || len(store.blobs) != 0 {
		t.Fatalf("import error = %v; blobs = %v", err, store.names())
	}
}
