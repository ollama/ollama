//go:build slow

package convert

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/llm"
)

func convertFull(t *testing.T, p string) (llm.KV, llm.Tensors) {
	t.Helper()

	mf, err := GetModelFormat(p)
	if err != nil {
		t.Fatal(err)
	}

	params, err := mf.GetParams(p)
	if err != nil {
		t.Fatal(err)
	}

	arch, err := mf.GetModelArch("", p, params)
	if err != nil {
		t.Fatal(err)
	}

	if err := arch.LoadVocab(); err != nil {
		t.Fatal(err)
	}

	if err := arch.GetTensors(); err != nil {
		t.Fatal(err)
	}

	f, err := os.CreateTemp(t.TempDir(), "f16")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := arch.WriteGGUF(f); err != nil {
		t.Fatal(err)
	}

	r, err := os.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	m, _, err := llm.DecodeGGML(r)
	if err != nil {
		t.Fatal(err)
	}

	return m.KV(), m.Tensors()
}

func TestConvertFull(t *testing.T) {
	cases := []struct {
		path    string
		arch    string
		tensors int
		layers  int
	}{
		{"Meta-Llama-3-8B-Instruct", "llama", 291, 35},
		{"Mistral-7B-Instruct-v0.2", "llama", 291, 35},
		{"Mixtral-8x7B-Instruct-v0.1", "llama", 291, 35},
		{"gemma-2b-it", "gemma", 164, 20},
	}

	for _, tt := range cases {
		t.Run(tt.path, func(t *testing.T) {
			p := filepath.Join("testdata", tt.path)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("%s not found", p)
			}

			kv, tensors := convertFull(t, p)

			if kv.Architecture() != tt.arch {
				t.Fatalf("expected llama, got %s", kv.Architecture())
			}

			if kv.FileType().String() != "F16" {
				t.Fatalf("expected F16, got %s", kv.FileType())
			}

			if len(tensors) != tt.tensors {
				t.Fatalf("expected %d tensors, got %d", tt.tensors, len(tensors))
			}

			layers := tensors.Layers()
			if len(layers) != tt.layers {
				t.Fatalf("expected %d layers, got %d", tt.layers, len(layers))
			}
		})
	}
}
