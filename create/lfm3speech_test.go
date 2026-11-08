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

func writeLFM3AudioSource(t *testing.T, dir string) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["Lfm3AudioForConditionalGeneration"],"model_type":"lfm3_audio","lfm":{"hidden_size":2048,"max_position_embeddings":128000}}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("lfm.model.layers.0.feed_forward.w1.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("depth_linear.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
	})
	codec := filepath.Join(dir, "audio_tokenizer")
	if err := os.Mkdir(codec, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(codec, "config.json"), []byte(`{"model_type":"liquid_codec"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(codec, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("decoder.lin.weight", "BF16", []int32{64, 64}, make([]byte, 64*64*2)),
		st.NewTensorDataFromBytes("decoder.istft.window", "F32", []int32{4}, make([]byte, 16)),
	})
}

func TestLFM3SpeechImportsCodecSubfolder(t *testing.T) {
	dir := t.TempDir()
	writeLFM3AudioSource(t, dir)

	files, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Contains(files, "audio_tokenizer/model.safetensors") {
		t.Fatalf("weight files = %v, %v", files, err)
	}
	configs, err := SafetensorsConfigFiles(dir)
	if err != nil || !slices.Contains(configs, "audio_tokenizer/config.json") {
		t.Fatalf("config files = %v, %v", configs, err)
	}

	inv, err := ReadInventory(dir)
	if err != nil {
		t.Fatal(err)
	}
	if inv, err = prepareLFM3SpeechInventory(inv); err != nil {
		t.Fatal(err)
	}
	if got := inv.Tensors["codec.decoder.lin.weight"]; got.Name != "decoder.lin.weight" || got.File != "audio_tokenizer/model.safetensors" {
		t.Fatalf("codec tensor = %+v", got)
	}
	policy, err := newTensorImportTransform(inv)
	if err != nil {
		t.Fatal(err)
	}
	specs, err := Plan(inv, Classification{Kind: SourceFloat, Quantize: "mxfp8"}, policy)
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range specs {
		for _, ts := range spec.Tensors {
			if want := strings.HasPrefix(ts.Name, "lfm.model."); (ts.Quantize != "") != want {
				t.Errorf("%s quantize = %q", ts.Name, ts.Quantize)
			}
		}
	}

	store := newCaptureStore()
	var got ManifestInfo
	err = Create(t.Context(), "lfm3-audio-test", dir, testPipelineOptions(), store, func(_ context.Context, _ string, info ManifestInfo) error {
		got = info
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"codec.decoder.lin.weight", "codec.decoder.istft.window", "audio_tokenizer/config.json"} {
		if store.blobs[name] == nil {
			t.Errorf("missing blob %s: %v", name, store.names())
		}
	}
	if !slices.Contains(got.ModelConfig.Capabilities, "speech") || !slices.Contains(got.ModelConfig.Capabilities, "completion") {
		t.Errorf("capabilities = %v", got.ModelConfig.Capabilities)
	}
	// The language model config is nested under "lfm"; the manifest records it.
	if !slices.Contains(got.ModelConfig.Voices, "default") || !slices.Contains(got.ModelConfig.Voices, "male-french") {
		t.Errorf("voices %v, want the LFM3 speech voices", got.ModelConfig.Voices)
	}
	if got.ModelConfig.ContextLen != 128000 || got.ModelConfig.EmbedLen != 2048 {
		t.Errorf("context length %d, embedding length %d, want 128000 and 2048", got.ModelConfig.ContextLen, got.ModelConfig.EmbedLen)
	}
}

func TestLFM3SpeechRequiresCodec(t *testing.T) {
	dir := t.TempDir()
	writeLFM3AudioSource(t, dir)
	if err := os.Remove(filepath.Join(dir, "audio_tokenizer", "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := SafetensorsWeightFiles(dir); err == nil || !strings.Contains(err.Error(), "audio_tokenizer/model.safetensors") {
		t.Fatalf("transfer list error = %v", err)
	}
	store := newCaptureStore()
	err := Create(t.Context(), "lfm3-audio-test", dir, testPipelineOptions(), store,
		func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "audio_tokenizer/model.safetensors") || len(store.blobs) != 0 {
		t.Fatalf("import error = %v; blobs = %v", err, store.names())
	}
}
