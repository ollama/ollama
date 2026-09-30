package create

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func TestCreateLaya(t *testing.T) {
	dir := t.TempDir()
	configs := map[string]string{
		"rl_agent_config.json":            `{"encoder":"answerdotai/ModernBERT-large","max_len":512,"head_layers":2}`,
		"encoder/config.json":             `{"model_type":"modernbert","hidden_size":1024,"num_hidden_layers":28,"num_attention_heads":16,"vocab_size":50368}`,
		"tokenizer/tokenizer.json":        `{"model":{"type":"BPE"}}`,
		"tokenizer/tokenizer_config.json": `{"cls_token":"[CLS]","sep_token":"[SEP]","mask_token":"[MASK]"}`,
	}
	for name, data := range configs {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("encoder.final_norm.weight", "F16", []int32{8}, make([]byte, 16)),
	})
	if !IsSafetensorsModelDir(dir) {
		t.Fatal("publisher layout was not recognized")
	}
	store := newCaptureStore()
	var info ManifestInfo
	write := func(_ context.Context, _ string, got ManifestInfo) error { info = got; return nil }
	if err := Create(t.Context(), "laya", dir, testPipelineOptions(), store, write, func(string) {}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(info.ModelConfig.Capabilities, []string{"decision"}) {
		t.Fatalf("capabilities: %v", info.ModelConfig.Capabilities)
	}
	for name, data := range configs {
		if !bytes.Equal(store.blobs[name], []byte(data)) {
			t.Errorf("publisher metadata changed: %s", name)
		}
	}
	var descriptor struct {
		Architectures []string `json:"architectures"`
		MaxLen        int      `json:"max_position_embeddings"`
	}
	if err := json.Unmarshal(store.blobs["config.json"], &descriptor); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(descriptor.Architectures, []string{"LayaForDecision"}) || descriptor.MaxLen != 512 {
		t.Fatalf("descriptor: %+v", descriptor)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); !os.IsNotExist(err) {
		t.Fatalf("import changed source: %v", err)
	}
	// A prepared source with the root descriptor still needs its nested metadata.
	if err := os.WriteFile(filepath.Join(dir, "config.json"), store.blobs["config.json"], 0o644); err != nil {
		t.Fatal(err)
	}
	store = newCaptureStore()
	if err := Create(t.Context(), "laya", dir, testPipelineOptions(), store, write, func(string) {}); err != nil {
		t.Fatal(err)
	}
	for name, data := range configs {
		if !bytes.Equal(store.blobs[name], []byte(data)) {
			t.Errorf("prepared source lost metadata: %s", name)
		}
	}
	opts := testPipelineOptions()
	opts.Quantize = "int4"
	store = newCaptureStore()
	if err := Create(t.Context(), "laya", dir, opts, store, write, func(string) {}); err == nil || !strings.Contains(err.Error(), "unquantized") {
		t.Fatalf("quantization: %v", err)
	}
	if len(store.blobs) != 0 {
		t.Fatal("unsupported quantization wrote blobs")
	}
	if err := os.Remove(filepath.Join(dir, "tokenizer/tokenizer.json")); err != nil {
		t.Fatal(err)
	}
	if _, err := SafetensorsConfigFiles(dir); err == nil {
		t.Fatal("accepted incomplete tokenizer")
	}
}
