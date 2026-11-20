package create

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	st "github.com/ollama/ollama/fs/safetensors"
)

func TestAutoTrustDecisionImportExcludesBuildNotes(t *testing.T) {
	dir := t.TempDir()
	writeConfigJSON(t, dir, `{"architectures":["AutoTrustDecision"],"model_type":"qwen3_5_text"}`)
	createTestSafetensors(t, filepath.Join(dir, "model.safetensors"), []*st.TensorData{
		st.NewTensorDataFromBytes("model.embed_tokens.weight", "BF16", []int32{8, 8}, make([]byte, 128)),
	})
	for name, data := range map[string]string{
		"decision_head.json":             `{"slots":{}}`,
		"calibration.json":               `{"version":1}`,
		"preprocessor_config.json":       `{}`,
		"video_preprocessor_config.json": `{}`,
		"hf_quant_config.json":           `{"quantization":{"quant_algo":"W4A16_NVFP4"}}`,
		"build-notes.json":               `{"source":"/tank/private-models","token":"fixture-secret"}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	store := newCaptureStore()
	var renderer string
	err := Create(t.Context(), "autotrust-decision-fixture", dir, testPipelineOptions(), store,
		func(_ context.Context, _ string, info ManifestInfo) error {
			renderer = info.ModelConfig.Renderer
			return nil
		}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if renderer != "autotrust-decision" {
		t.Fatalf("renderer = %q, want autotrust-decision", renderer)
	}
	if _, ok := store.blobs["build-notes.json"]; ok {
		t.Fatal("internal source notes were packaged")
	}
	for _, name := range []string{"config.json", "decision_head.json", "calibration.json", "preprocessor_config.json", "video_preprocessor_config.json", "hf_quant_config.json"} {
		if _, ok := store.blobs[name]; !ok {
			t.Errorf("required AutoTrust decision metadata %s was omitted", name)
		}
	}
}
