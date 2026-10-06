package create

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func writeClefSource(t *testing.T, dir string) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"hidden_size":4},"publisher_extension":{"large_integer":123456789012345678901234567890,"scale":1e6}}`)
	for name, data := range map[string]string{
		"joint_head_config.json":       `{"hidden_size":4,"width":2,"heads":1,"layers":1,"routing_layers":1,"feedforward":4}`,
		"model.safetensors.index.json": `{"weight_map":{"model.weight":"model.safetensors"}}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.weight", "BF16", []int32{2, 4}, make([]byte, 16)),
	})
	createTestSafetensors(t, filepath.Join(dir, "joint_head.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("memory_projection.weight", "BF16", []int32{2, 4}, make([]byte, 16)),
		st.NewTensorDataFromBytes("type_embedding.weight", "BF16", []int32{3, 2}, make([]byte, 12)),
	})
}

func TestClefInventory(t *testing.T) {
	for _, tc := range []struct{ name, remove, config, index, wantError string }{
		{name: "complete"},
		{name: "monolithic backbone", remove: "model.safetensors.index.json"},
		{name: "head already indexed", index: `{"weight_map":{"model.weight":"model.safetensors","memory_projection.weight":"joint_head.safetensors","type_embedding.weight":"joint_head.safetensors"}}`},
		{name: "missing config", remove: "joint_head_config.json", wantError: "read Clef joint head config"},
		{name: "missing head", remove: "joint_head.safetensors", wantError: "read Clef joint head"},
		{name: "different backbone", config: `{"architectures":["Qwen3ForCausalLM"]}`, wantError: "unsupported Clef backbone"},
		{name: "different hidden size", config: `{"architectures":["Qwen3_5ForConditionalGeneration"],"text_config":{"hidden_size":8}}`, wantError: "invalid Clef joint head configuration"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeClefSource(t, dir)
			if tc.remove != "" {
				if err := os.Remove(filepath.Join(dir, tc.remove)); err != nil {
					t.Fatal(err)
				}
			}
			if tc.config != "" {
				writeConfigJSON(t, dir, tc.config)
			}
			if tc.index != "" {
				if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(tc.index), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			inv, err := ReadInventory(dir)
			if err == nil {
				inv, err = prepareClefInventory(inv)
			}
			if tc.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("prepareClefInventory() error=%v, want %q", err, tc.wantError)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if inv.Config.Architecture() != "ClefForDecision" || len(inv.Tensors) != 3 || inv.Tensors["memory_projection.weight"].File != "joint_head.safetensors" {
				t.Fatalf("incorrect composite inventory: %+v", inv)
			}
			files, err := SafetensorsWeightFiles(dir)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(files, []string{"joint_head.safetensors", "model.safetensors"}) {
				t.Fatalf("source files=%v", files)
			}
		})
	}
}

func TestClefImportPreservesSourceConfig(t *testing.T) {
	dir := t.TempDir()
	writeClefSource(t, dir)
	original, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ordinary.Config.Architecture() != "Qwen3_5ForConditionalGeneration" || ordinary.Has("memory_projection.weight") {
		t.Fatalf("filenames selected Clef without explicit intent: %+v", ordinary)
	}
	store := newCaptureStore()
	var result ManifestInfo
	err = Create(t.Context(), "clef-test", dir, PipelineOptions{
		Renderer:   "clef",
		Validation: MLXValidationOptions{Force: true, Warning: func(string) {}},
	}, store, func(_ context.Context, _ string, info ManifestInfo) error {
		result = info
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	unchanged, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(unchanged, original) {
		t.Fatal("publisher source config changed during import")
	}
	var source, output map[string]json.RawMessage
	if err := json.Unmarshal(original, &source); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(store.blobs["config.json"], &output); err != nil {
		t.Fatal(err)
	}
	source["architectures"] = json.RawMessage(`["ClefForDecision"]`)
	if !reflect.DeepEqual(output, source) {
		t.Fatalf("output config changed fields other than architectures: %s", store.blobs["config.json"])
	}
	var configLayers int
	for _, layer := range result.Layers {
		if layer.Name == "config.json" {
			configLayers++
		}
	}
	if configLayers != 1 {
		t.Fatalf("got %d config.json layers, want 1", configLayers)
	}
	if result.ModelConfig.Renderer != "clef" || result.ModelConfig.Parser != "" || !slices.Equal(result.ModelConfig.Capabilities, []string{"decision"}) {
		t.Fatalf("incorrect manifest config: %+v", result.ModelConfig)
	}
	if _, ok := store.blobs["memory_projection.weight"]; !ok {
		t.Fatal("head tensor was omitted")
	}
}

func TestClefMetadata(t *testing.T) {
	for _, tc := range []struct {
		name         string
		vision       bool
		capabilities []string
	}{
		{"text", false, []string{"decision"}},
		{"vision", true, []string{"decision", "vision"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeClefSource(t, dir)
			inv, err := ReadInventory(dir)
			if err == nil {
				inv, err = prepareClefInventory(inv)
			}
			if err != nil {
				t.Fatal(err)
			}
			if tc.vision {
				v := map[string]any{}
				inv.Config.VisionConfig = &v
			}
			cfg, err := inferSafetensorsConfig(dir, inv.Config, "", "clef")
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(cfg.Capabilities, tc.capabilities) || cfg.Parser != "" || cfg.Renderer != "clef" {
				t.Fatalf("incorrect Clef metadata: %+v", cfg)
			}
		})
	}
}

func TestClefImportRejectsIncompleteSource(t *testing.T) {
	for _, missing := range []string{"joint_head_config.json", "joint_head.safetensors"} {
		t.Run(missing, func(t *testing.T) {
			dir := t.TempDir()
			writeClefSource(t, dir)
			if err := os.Remove(filepath.Join(dir, missing)); err != nil {
				t.Fatal(err)
			}
			store := newCaptureStore()
			err := Create(t.Context(), "clef-test", dir, PipelineOptions{Renderer: "clef"}, store, func(context.Context, string, ManifestInfo) error {
				t.Fatal("incomplete source reached manifest writer")
				return nil
			}, func(string) {})
			if err == nil || len(store.blobs) != 0 {
				t.Fatalf("incomplete source: error=%v, wrote %d blobs", err, len(store.blobs))
			}
		})
	}
}

func TestClefWeightFiles(t *testing.T) {
	for _, tc := range []struct {
		name, head string
		indexed    bool
		wantError  bool
	}{
		{name: "absent"},
		{name: "unindexed", head: "file"},
		{name: "already indexed", head: "file", indexed: true},
		{name: "directory", head: "directory", wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			index := `{"weight_map":{"model.weight":"model.safetensors"}}`
			if tc.indexed {
				index = `{"weight_map":{"model.weight":"model.safetensors","head.weight":"joint_head.safetensors"}}`
			}
			if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "consolidated.safetensors"), nil, 0o600); err != nil {
				t.Fatal(err)
			}
			head := filepath.Join(dir, "joint_head.safetensors")
			if tc.head == "file" {
				if err := os.WriteFile(head, nil, 0o600); err != nil {
					t.Fatal(err)
				}
			} else if tc.head == "directory" {
				if err := os.Mkdir(head, 0o700); err != nil {
					t.Fatal(err)
				}
			}
			files, err := SafetensorsWeightFiles(dir)
			if tc.wantError {
				if err == nil || !strings.Contains(err.Error(), "regular file") {
					t.Fatalf("got %v, want regular-file error", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := []string{"model.safetensors"}
			if tc.head == "file" {
				want = []string{"joint_head.safetensors", "model.safetensors"}
			}
			if !slices.Equal(files, want) {
				t.Fatalf("files=%v, want %v", files, want)
			}
		})
	}
}
