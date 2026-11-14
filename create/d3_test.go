package create

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func writeD3Source(t *testing.T, dir string) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3_5Model"],"model_type":"qwen3_5","text_config":{"hidden_size":4},"vision_config":{}}`)
	ids := make([]string, 255)
	for i := range ids {
		ids[i] = strconv.Itoa(32 + i)
	}
	decision := `{"format_version":1,"format_id":"` + d3FormatID + `","prompt":"d3","attention_mode":"noncausal_full_attention","pooling":"last","readout_dtype":"float32","temperature":1,"codes":[` + strings.TrimSuffix(strings.Repeat(`"A",`, 255), ",") + `],"token_ids":[` + strings.Join(ids, ",") + `]}`
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

const autoTrustTestVocab = 300

func autoTrustReadoutConfig(formatID, prompt string, firstID int) string {
	ids := make([]string, 255)
	for i := range ids {
		ids[i] = strconv.Itoa(firstID + i)
	}
	codes := strings.TrimSuffix(strings.Repeat(`"A",`, 255), ",")
	return `{"format_version":1,"format_id":"` + formatID + `","prompt":"` + prompt + `","attention_mode":"noncausal_full_attention","pooling":"last","max_length":null,"readout_dtype":"float32","temperature":1,"codes":[` + codes + `],"token_ids":[` + strings.Join(ids, ",") + `]}`
}

func writeAutoTrustSource(t *testing.T, dir string) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5","text_config":{"hidden_size":4,"vocab_size":`+strconv.Itoa(autoTrustTestVocab)+`,"model_type":"qwen3_5_text"},"vision_config":{"hidden_size":8}}`)
	if err := os.WriteFile(filepath.Join(dir, "decision_config.json"), []byte(autoTrustReadoutConfig(autoTrustFormatID, "jev", 0)), 0o600); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.language_model.embed_tokens.weight", "BF16", []int32{autoTrustTestVocab, 4}, make([]byte, autoTrustTestVocab*4*2)),
		st.NewTensorDataFromBytes("model.visual.patch_embed.proj.weight", "BF16", []int32{8, 4}, make([]byte, 64)),
	})
	rows := make([]byte, 0, autoTrustTestVocab*4*2)
	for row := range autoTrustTestVocab {
		for range 4 {
			rows = binary.LittleEndian.AppendUint16(rows, bf16Bits(float32(row)))
		}
	}
	createTestSafetensors(t, filepath.Join(dir, "lm_head.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("lm_head.weight", "BF16", []int32{autoTrustTestVocab, 4}, rows),
	})
	index := `{"weight_map":{"model.language_model.embed_tokens.weight":"model.safetensors","model.visual.patch_embed.proj.weight":"model.safetensors","lm_head.weight":"lm_head.safetensors"}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o600); err != nil {
		t.Fatal(err)
	}
}

func bf16Bits(v float32) uint16 {
	bits := math.Float32bits(v)
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}

func createAutoTrustTestModel(t *testing.T, dir string) (*captureStore, ManifestInfo) {
	t.Helper()
	store := newCaptureStore()
	var info ManifestInfo
	if err := Create(t.Context(), "readout-test", dir, testPipelineOptions(), store, func(_ context.Context, _ string, got ManifestInfo) error {
		info = got
		return nil
	}, func(string) {}); err != nil {
		t.Fatal(err)
	}
	return store, info
}

func TestAutoTrustImportKeepsPublishedReadoutAndVision(t *testing.T) {
	dir := t.TempDir()
	writeAutoTrustSource(t, dir)
	store, info := createAutoTrustTestModel(t, dir)

	if info.ModelConfig.Renderer != "d3" {
		t.Fatalf("renderer = %q", info.ModelConfig.Renderer)
	}
	if got := info.ModelConfig.Capabilities; !slices.Equal(got, []string{"decision", "vision"}) {
		t.Fatalf("capabilities = %v", got)
	}

	names := store.names()
	if !slices.Contains(names, "model.visual.patch_embed.proj.weight") {
		t.Fatalf("vision tower was not packaged: %v", names)
	}
	if slices.Contains(names, readoutTensorName) {
		t.Fatalf("import derived a %s tensor for a folded readout: %v", readoutTensorName, names)
	}
	lmHead, ok := store.blobs[lmHeadReadoutTensor]
	if !ok {
		t.Fatalf("publisher lm_head was not packaged: %v", names)
	}
	if got := readSafetensorsHeaderNames(t, lmHead); !slices.Equal(got, []string{lmHeadReadoutTensor}) {
		t.Fatalf("lm_head blob tensors = %v", got)
	}
	raw := readPackedTensorRaw(t, lmHead, lmHeadReadoutTensor)
	if len(raw) != autoTrustTestVocab*4*2 {
		t.Fatalf("lm_head bytes = %d, want the publisher's BF16 matrix", len(raw))
	}
	for row := range autoTrustTestVocab {
		want := math.Float32frombits(uint32(bf16Bits(float32(row))) << 16)
		for col := range 4 {
			bits := binary.LittleEndian.Uint16(raw[(row*4+col)*2:])
			if v := math.Float32frombits(uint32(bits) << 16); v != want {
				t.Fatalf("lm_head[%d][%d] = %v, want publisher row %v", row, col, v, want)
			}
		}
	}

	var packaged map[string]json.RawMessage
	if err := json.Unmarshal(store.blobs["config.json"], &packaged); err != nil {
		t.Fatal(err)
	}
	if _, ok := packaged["vision_config"]; !ok {
		t.Fatal("packaged config lost vision_config")
	}
	var architectures []string
	if err := json.Unmarshal(packaged["architectures"], &architectures); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(architectures, []string{"Qwen3_5Model"}) {
		t.Fatalf("architectures = %v", architectures)
	}
	if store.blobs["decision_config.json"] == nil {
		t.Fatal("decision_config.json was not imported")
	}
}

func TestAutoTrustImportRejectsMismatchedFormat(t *testing.T) {
	dir := t.TempDir()
	writeAutoTrustSource(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "decision_config.json"), []byte(autoTrustReadoutConfig(d3FormatID, "d3", 0)), 0o600); err != nil {
		t.Fatal(err)
	}
	err := Create(t.Context(), "readout-test", dir, testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "code-readout") && !strings.Contains(err.Error(), "d3 decision checkpoint") {
		t.Fatalf("error = %v", err)
	}
}

func TestAutoTrustImportRejectsOutOfRangeCode(t *testing.T) {
	dir := t.TempDir()
	writeAutoTrustSource(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "decision_config.json"), []byte(autoTrustReadoutConfig(autoTrustFormatID, "jev", autoTrustTestVocab-254)), 0o600); err != nil {
		t.Fatal(err)
	}
	err := Create(t.Context(), "readout-test", dir, testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("error = %v", err)
	}
}

func TestAutoTrustImportRejectsDuplicateCode(t *testing.T) {
	dir := t.TempDir()
	writeAutoTrustSource(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "decision_config.json"), []byte(autoTrustReadoutConfig(autoTrustFormatID, "jev", 0)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "decision_config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	updated := strings.Replace(string(data), `"token_ids":[0,1,`, `"token_ids":[0,0,`, 1)
	if err := os.WriteFile(path, []byte(updated), 0o600); err != nil {
		t.Fatal(err)
	}
	err = Create(t.Context(), "readout-test", dir, testPipelineOptions(), newCaptureStore(), func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("error = %v", err)
	}
}

func TestCodeReadoutImportRejectsInvalidTemperature(t *testing.T) {
	for _, tc := range []struct {
		name  string
		write func(*testing.T, string)
	}{
		{"d3", writeD3Source},
		{"AutoTrust", writeAutoTrustSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			tc.write(t, dir)
			path := filepath.Join(dir, "decision_config.json")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			invalid := strings.Replace(string(data), `"temperature":1`, `"temperature":0`, 1)
			if invalid == string(data) {
				t.Fatal("fixture has no temperature to invalidate")
			}
			if err := os.WriteFile(path, []byte(invalid), 0o600); err != nil {
				t.Fatal(err)
			}
			store := newCaptureStore()
			err = Create(t.Context(), "readout-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
				return nil
			}, func(string) {})
			if err == nil || !strings.Contains(err.Error(), "invalid decision temperature") || len(store.blobs) != 0 {
				t.Fatalf("import error = %v; blobs = %v", err, store.names())
			}
		})
	}
}

func TestAutoTrustImportRejectsCompetingReadout(t *testing.T) {
	dir := t.TempDir()
	writeAutoTrustSource(t, dir)
	createTestSafetensors(t, filepath.Join(dir, "extra.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes(readoutTensorName, "F32", []int32{255, 4}, make([]byte, 255*4*4)),
	})
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	data, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &index); err != nil {
		t.Fatal(err)
	}
	index.WeightMap[readoutTensorName] = "extra.safetensors"
	data, err = json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(indexPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
	store := newCaptureStore()
	err = Create(t.Context(), "readout-test", dir, testPipelineOptions(), store, func(context.Context, string, ManifestInfo) error {
		return nil
	}, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "cannot also ship weight") || len(store.blobs) != 0 {
		t.Fatalf("import error = %v; blobs = %v", err, store.names())
	}
}

func TestAutoTrustFoldedReadoutQuantizationPolicy(t *testing.T) {
	policy := foldedReadoutImportTransform{}
	for _, tc := range []struct {
		name string
		want string
	}{
		{lmHeadReadoutTensor, ""},
		{"model.language_model.layers.0.mlp.up_proj.weight", "nvfp4"},
	} {
		if got := policy.quantizationType(tc.name, []int32{5120, 5120}, "nvfp4"); got != tc.want {
			t.Errorf("%s quantization = %q, want %q", tc.name, got, tc.want)
		}
	}
}
