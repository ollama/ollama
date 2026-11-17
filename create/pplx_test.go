package create

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func writePplxSource(t *testing.T, dir string) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3_5Model"],"model_type":"qwen3_5","text_config":{"hidden_size":4},"vision_config":{}}`)
	if err := os.WriteFile(filepath.Join(dir, "decision_config.json"), []byte(`{"format_version":2,"attention_mode":"noncausal_full_attention","pooling":"last","codes":[`+strings.TrimSuffix(strings.Repeat(`"A",`, 255), ",")+`],"token_ids":[`+strings.TrimSuffix(strings.Repeat(`32,`, 255), ",")+`]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "joint_config.json"), []byte(`{"version":3,"template":"joint-question-inline-readout-v1","marker":"<|fim_suffix|>","readout_offset":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 4}, make([]byte, 64)),
	})
	createTestSafetensors(t, filepath.Join(dir, "readout.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("weight", "BF16", []int32{255, 4}, make([]byte, 2040)),
	})
}

func TestPplxMetadataFilterNeedsDecisionSidecar(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3_5Model"]}`)
	if err := os.WriteFile(filepath.Join(dir, "custom_config.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	files, err := SafetensorsConfigFiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(files, "custom_config.json") {
		t.Fatalf("bare Qwen backbone lost metadata: %v", files)
	}
}

func TestPplxSidecarCheckReportsFilesystemErrors(t *testing.T) {
	if _, err := hasPplxSidecar("\x00"); err == nil {
		t.Fatal("sidecar stat error was treated as an absent sidecar")
	}
}

func TestPplxBareBackboneRejectedBeforeImport(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3_5Model"]}`)
	cfg, _, err := readSourceModelConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := inferSafetensorsConfig(dir, cfg, "", ""); err == nil || !strings.Contains(err.Error(), "requires decision sidecars") {
		t.Fatalf("bare backbone import error = %v", err)
	}
}

func TestPplxInventoryAndImport(t *testing.T) {
	dir := t.TempDir()
	writePplxSource(t, dir)
	// A stray Locate sidecar directory should not alter a v3 import.
	writePplxLocateSource(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "release-manifest.json"), []byte(`{"verified_upstream":"full"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	if inv.Has("weight") {
		t.Fatal("readout was unexpectedly in the backbone shard inventory")
	}
	inv, err = preparePplxInventory(inv)
	if err != nil {
		t.Fatal(err)
	}
	if inv.Tensors["weight"].File != "readout.safetensors" {
		t.Fatalf("readout source = %+v", inv.Tensors["weight"])
	}
	files, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Equal(files, []string{"model.safetensors", "readout.safetensors"}) {
		t.Fatalf("source files = %v, %v", files, err)
	}
	store := newCaptureStore()
	var result ManifestInfo
	err = Create(t.Context(), "pplx-test", dir, testPipelineOptions(), store, func(_ context.Context, _ string, info ManifestInfo) error {
		result = info
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if result.ModelConfig.Renderer != "pplx" || result.ModelConfig.Parser != "" || !slices.Equal(result.ModelConfig.Capabilities, []string{"decision", "vision"}) {
		t.Fatalf("model config = %+v", result.ModelConfig)
	}
	if _, ok := store.blobs["decision_config.json"]; !ok {
		t.Fatal("decision_config.json not imported")
	}
	if _, ok := store.blobs["joint_config.json"]; !ok {
		t.Fatal("joint_config.json not imported")
	}
	if _, ok := store.blobs[pplxLocateConfigFile]; ok {
		t.Fatal("Locate sidecar was imported into a v3 model")
	}
	if _, ok := store.blobs["release-manifest.json"]; ok {
		t.Fatal("release metadata was imported as model configuration")
	}
}

func TestPplxChunkedProtocolInventory(t *testing.T) {
	dir := t.TempDir()
	writePplxSource(t, dir)
	path := filepath.Join(dir, "joint_config.json")
	if err := os.WriteFile(path, []byte(`{"version":4,"template":"joint-chunked-readout-v1","marker":"<|fim_suffix|>","readout_offset":0,"separator":"<|fim_middle|>","usage":"nonpadding_joint_sequence_including_template_and_image_positions","max_questions":null,"max_length":null}`), 0o600); err != nil {
		t.Fatal(err)
	}
	writePplxLocateSource(t, dir)
	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	inv, err = preparePplxInventory(inv)
	if err != nil {
		t.Fatalf("v4 inventory: %v", err)
	}
	// The fixture has one backbone, one readout, one marker and five head projections.
	if len(inv.Tensors) != 8 {
		t.Fatalf("v4 inventory has %d tensors", len(inv.Tensors))
	}
	files, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Contains(files, "locate-span/adapter.safetensors") || !slices.Contains(files, "locate-span/head.safetensors") {
		t.Fatalf("Locate files = %v, %v", files, err)
	}
	configs, err := SafetensorsConfigFiles(dir)
	if err != nil || !slices.Contains(configs, "locate-span/locate_config.json") {
		t.Fatalf("Locate config files = %v, %v", configs, err)
	}
	store := newCaptureStore()
	if err := Create(t.Context(), "pplx-v4-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {}); err != nil {
		t.Fatalf("create v4 fixture: %v", err)
	}
	if _, ok := store.blobs[pplxLocateConfigFile]; !ok {
		t.Fatal("Locate config was not imported into a v4 model")
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"changed":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	base, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := preparePplxInventory(base); err == nil || !strings.Contains(err.Error(), "does not match base tokenizer.json") {
		t.Fatalf("mismatched Locate base error = %v", err)
	}
	if err := os.Remove(filepath.Join(dir, "locate-span", "head.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := SafetensorsWeightFiles(dir); err == nil || !strings.Contains(err.Error(), "head.safetensors") {
		t.Fatalf("missing Locate head error = %v", err)
	}
	if err := os.WriteFile(path, []byte(`{"version":4,"template":"joint-chunked-readout-v1","marker":"<|fim_suffix|>","readout_offset":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := preparePplxInventory(inv); err == nil {
		t.Fatal("v4 without its separator was accepted")
	}
}

func writePplxLocateSource(t *testing.T, dir string) {
	t.Helper()
	locate := filepath.Join(dir, "locate-span")
	if err := os.Mkdir(locate, 0o700); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(locate, "adapter.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("language_model.embed_tokens.locate", "F32", []int32{4}, make([]byte, 16)),
	})
	var head []*st.TensorData
	for _, name := range []string{"ts", "te", "qs", "qe", "qp"} {
		head = append(head, st.NewTensorDataFromBytes(name+".weight", "F32", []int32{128, 4}, make([]byte, 128*4*4)))
	}
	createTestSafetensors(t, filepath.Join(locate, "head.safetensors"), head)
	digests := make(map[string]string)
	for _, name := range []string{"config.json", "decision_config.json", "joint_config.json", "readout.safetensors", "tokenizer.json"} {
		path := filepath.Join(dir, name)
		if name == "tokenizer.json" {
			if err := os.WriteFile(path, []byte(`{}`), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		digests[name] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	config := map[string]any{
		"format": "pplx-decider-locate-span-v1", "marker": "<|locate|>", "marker_id": 7,
		"base": map[string]any{"sha256": digests},
		"lora": map[string]any{"rank": 64, "alpha": 64, "layers": 0, "merged": false},
		"head": map[string]any{"width": 128, "max_span": 128, "proposals": 128},
	}
	data, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(locate, "locate_config.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestPplxMissingDecisionConfigRejectedBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writePplxSource(t, dir)
	if err := os.Remove(filepath.Join(dir, "decision_config.json")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "release-manifest.json"), []byte(`{"review":"local"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	files, err := SafetensorsConfigFiles(dir)
	if err != nil || slices.Contains(files, "release-manifest.json") {
		t.Fatalf("source metadata files = %v, %v", files, err)
	}
	store := newCaptureStore()
	err = Create(t.Context(), "pplx-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "read Perplexity decision config") {
		t.Fatalf("Create() error = %v, want missing decision config", err)
	}
	if len(store.blobs) != 0 {
		t.Fatal("incomplete decision checkpoint wrote blobs")
	}
}

func TestPplxMissingReadoutRejectedBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writePplxSource(t, dir)
	if err := os.Remove(filepath.Join(dir, "readout.safetensors")); err != nil {
		t.Fatal(err)
	}
	store := newCaptureStore()
	err := Create(t.Context(), "pplx-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "open Perplexity readout") {
		t.Fatalf("Create() error = %v", err)
	}
	if len(store.blobs) != 0 {
		t.Fatal("incomplete model wrote blobs")
	}
}

func TestPplxUnsupportedDecisionModeRejectedBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writePplxSource(t, dir)
	path := filepath.Join(dir, "decision_config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data = []byte(strings.Replace(string(data), `"noncausal_full_attention"`, `"causal"`, 1))
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	store := newCaptureStore()
	err = Create(t.Context(), "pplx-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "unsupported Perplexity decision attention mode") {
		t.Fatalf("Create() error = %v, want unsupported-mode rejection", err)
	}
	if len(store.blobs) != 0 {
		t.Fatal("unsupported decision mode wrote blobs")
	}
}

func TestPplxReadoutTransferIsArchitectureScoped(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["LlamaForCausalLM"],"model_type":"llama"}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{1, 1}, make([]byte, 2)),
	})
	createTestSafetensors(t, filepath.Join(dir, "readout.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("weight", "BF16", []int32{1, 1}, make([]byte, 2)),
	})
	files, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Equal(files, []string{"model.safetensors"}) {
		t.Fatalf("unrelated source files = %v, %v", files, err)
	}
}

func TestPplxQuantizationPreservesReadout(t *testing.T) {
	policy, err := newPplxImportTransform(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		want string
	}{
		{"weight", ""},
		{"language_model.layers.0.mlp.up_proj.weight", "nvfp4"},
		{"language_model.layers.0.linear_attn.in_proj_a.weight", ""},
		{"visual.blocks.0.mlp.fc1.weight", ""},
	} {
		if got := policy.quantizationType(tc.name, []int32{5120, 5120}, "nvfp4"); got != tc.want {
			t.Errorf("%s quantization = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestPplxRejectsIncompatibleParserAndRenderer(t *testing.T) {
	for _, tc := range []struct {
		name string
		set  func(*PipelineOptions)
	}{
		{"renderer", func(opts *PipelineOptions) { opts.Renderer = "qwen3.8" }},
		{"parser", func(opts *PipelineOptions) { opts.Parser = "qwen3.5" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writePplxSource(t, dir)
			opts := testPipelineOptions()
			tc.set(&opts)
			store := newCaptureStore()
			err := Create(t.Context(), "pplx-test", dir, opts, store, func(context.Context, string, ManifestInfo) error {
				return nil
			}, func(string) {})
			if err == nil || !strings.Contains(err.Error(), "require renderer") {
				t.Fatalf("Create() error = %v, want incompatible override rejection", err)
			}
			if len(store.blobs) != 0 {
				t.Fatal("invalid metadata wrote blobs")
			}
		})
	}
}
