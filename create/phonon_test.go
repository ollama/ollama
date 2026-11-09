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

func writeGradiumPhononSource(t *testing.T, dir string, voiceShape []int32) {
	t.Helper()
	writeConfigJSON(t, dir, `{"architectures":["GradiumPhononTTSForConditionalGeneration"],"model_type":"gradium_phonon_tts"}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight", "BF16", []int32{512, 512}, make([]byte, 512*512*2)),
	})
	voiceBytes := 4
	for _, size := range voiceShape {
		voiceBytes *= int(size)
	}
	createTestSafetensors(t, filepath.Join(dir, "default-voice.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("speaker_wavs", "F32", voiceShape, make([]byte, voiceBytes)),
	})
}

func TestGradiumPhononImportsPublisherVoices(t *testing.T) {
	dir := t.TempDir()
	writeGradiumPhononSource(t, dir, []int32{1, 512, 2})
	if err := os.Mkdir(filepath.Join(dir, "voices"), 0o755); err != nil {
		t.Fatal(err)
	}
	createTestSafetensors(t, filepath.Join(dir, "voices", "Zoey.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("speaker_wavs", "F32", []int32{1, 512, 2}, make([]byte, 4*512*2)),
	})
	files, err := SafetensorsWeightFiles(dir)
	if err != nil || !slices.Contains(files, "default-voice.safetensors") || !slices.Contains(files, "voices/Zoey.safetensors") {
		t.Fatalf("weight files = %v, %v", files, err)
	}
	store := newCaptureStore()
	var got ManifestInfo
	err = Create(t.Context(), "phonon-test", dir, testPipelineOptions(), store, func(_ context.Context, _ string, info ManifestInfo) error {
		got = info
		return nil
	}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.ModelConfig.Capabilities, []string{"speech"}) || got.ModelConfig.Requires != phononMinOllamaVersion {
		t.Fatalf("model config = %+v", got.ModelConfig)
	}
	if store.blobs["voices.default"] == nil || store.blobs["voices.Zoey"] == nil {
		t.Fatalf("missing voice blobs: %v", store.names())
	}
}

func TestGradiumPhononRejectsBadVoiceBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	writeGradiumPhononSource(t, dir, []int32{1, 256, 2})
	store := newCaptureStore()
	err := Create(t.Context(), "phonon-test", dir, testPipelineOptions(), store,
		func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "unsupported Phonon default voice") || len(store.blobs) != 0 {
		t.Fatalf("import error = %v; blobs = %v", err, store.names())
	}
	if err := os.Remove(filepath.Join(dir, "default-voice.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := SafetensorsWeightFiles(dir); err == nil || !strings.Contains(err.Error(), "requires default-voice.safetensors") {
		t.Fatalf("missing voice transfer error = %v", err)
	}
}

func TestPrereleasePhononImportRejected(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["Phonon"],"model_type":"phonon"}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("flow_lm.bos_emb", "F32", []int32{4}, make([]byte, 16)),
	})
	store := newCaptureStore()
	err := Create(t.Context(), "phonon-legacy", dir, PipelineOptions{}, store,
		func(context.Context, string, ManifestInfo) error { return nil }, func(string) {})
	if err == nil || !strings.Contains(err.Error(), "unsupported MLX architecture") || len(store.blobs) != 0 {
		t.Fatalf("import error = %v; blobs = %v", err, store.names())
	}
}
