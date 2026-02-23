package client

import (
	"os"
	"path/filepath"
	"testing"
)

func writeConfig(t *testing.T, dir, cfg string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestParserAndRendererInferenceQwen35(t *testing.T) {
	t.Run("qwen3.5 architecture uses qwen3.5 runtime-toggle stack", func(t *testing.T) {
		dir := t.TempDir()
		writeConfig(t, dir, `{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}`)

		if got, want := getParserName(dir), "qwen3.5"; got != want {
			t.Fatalf("getParserName() = %q, want %q", got, want)
		}
		if got, want := getRendererName(dir), "qwen3.5"; got != want {
			t.Fatalf("getRendererName() = %q, want %q", got, want)
		}
	})

	t.Run("qwen3 legacy inference unchanged", func(t *testing.T) {
		dir := t.TempDir()
		writeConfig(t, dir, `{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}`)

		if got, want := getParserName(dir), "qwen3"; got != want {
			t.Fatalf("getParserName() = %q, want %q", got, want)
		}
		if got, want := getRendererName(dir), "qwen3-coder"; got != want {
			t.Fatalf("getRendererName() = %q, want %q", got, want)
		}
	})
}
