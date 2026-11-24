package decisionqwen

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestDecisionFormatDispatch(t *testing.T) {
	for _, tc := range []struct {
		name, config, wantError string
	}{
		{"Perplexity legacy", `{}`, `Perplexity joint config`},
		{"d3", `{"format_id":"d3-code-readout-v1"}`, `unsupported d3 decision config`},
		{"unknown", `{"format_id":"future-format"}`, `unsupported Qwen decision format "future-format"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			layer, err := manifest.NewLayer(strings.NewReader(tc.config), manifest.MediaTypeImageJSON)
			if err != nil {
				t.Fatal(err)
			}
			layer.Name = "decision_config.json"
			root := &model.Root{Manifest: &manifest.Manifest{Layers: []manifest.Layer{layer}}}
			_, err = newModel(root)
			if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("dispatch error = %v, want %q", err, tc.wantError)
			}
		})
	}
}
