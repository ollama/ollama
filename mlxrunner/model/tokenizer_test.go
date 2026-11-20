package model

import (
	"errors"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
)

func tokenizerTestManifest(t *testing.T, files map[string]string) *manifest.Manifest {
	t.Helper()
	m := &manifest.Manifest{}
	for name, data := range files {
		layer, err := manifest.NewLayer(strings.NewReader(data), manifest.MediaTypeImageJSON)
		if err != nil {
			t.Fatal(err)
		}
		layer.Name = name
		m.Layers = append(m.Layers, layer)
	}
	return m
}

func TestLoadTokenizerOptions(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	newModel := func(*Root) (Model, error) { return nil, nil }
	archs := map[string][]TokenizerOption{
		"TokenizerTestStandard":   nil,
		"TokenizerTestDir":        {WithTokenizerDir("tokenizer")},
		"TokenizerTestGeneration": {WithGenerationConfig([]byte(`{"eos_token_id":[1,2]}`))},
		"TokenizerTestNone":       {WithoutTokenizer()},
	}
	for arch, opts := range archs {
		Register(arch, newModel, opts...)
	}
	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		for arch := range archs {
			delete(registry, arch)
			delete(tokenizerSpecs, arch)
		}
	})

	const tok = `{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2},"merges":[]}}`
	files := func(arch string, extra map[string]string) map[string]string {
		f := map[string]string{
			"config.json":            `{"architectures":["` + arch + `"],"eos_token_id":0}`,
			"tokenizer.json":         tok,
			"generation_config.json": `{"eos_token_id":2}`,
		}
		for k, v := range extra {
			if v == "" {
				delete(f, k)
			} else {
				f[k] = v
			}
		}
		return f
	}

	for _, tc := range []struct {
		name       string
		files      map[string]string
		wantEOS    []int32
		wantAddBOS bool
		wantErr    error
	}{
		{"generation config wins", files("TokenizerTestStandard", nil), []int32{2}, false, nil},
		{"config.json fallback", files("TokenizerTestStandard", map[string]string{"generation_config.json": ""}), []int32{0}, false, nil},
		{"tokenizer dir", files("TokenizerTestDir", map[string]string{"tokenizer.json": "", "tokenizer/tokenizer.json": tok, "tokenizer/tokenizer_config.json": `{"add_bos_token":true}`}), []int32{2}, true, nil},
		{"replacement generation config", files("TokenizerTestGeneration", nil), []int32{1, 2}, false, nil},
		{"without tokenizer", files("TokenizerTestNone", nil), nil, false, ErrNoTokenizer},
		{"unregistered", files("TokenizerTestUnknown", nil), nil, false, ErrNoTokenizer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := LoadTokenizer(tokenizerTestManifest(t, tc.files))
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.AddBOS() != tc.wantAddBOS {
				t.Fatalf("AddBOS = %t, want %t", got.AddBOS(), tc.wantAddBOS)
			}
			if !slices.Equal(got.EOSTokens(), tc.wantEOS) {
				t.Fatalf("EOS = %v, want %v", got.EOSTokens(), tc.wantEOS)
			}
			if ids := got.Encode("abc", false); !slices.Equal(ids, []int32{0, 1, 2}) {
				t.Fatalf("Encode = %v", ids)
			}
		})
	}

	t.Run("listed config unreadable", func(t *testing.T) {
		m := tokenizerTestManifest(t, files("TokenizerTestStandard", nil))
		layer, _ := m.ConfigLayer("generation_config.json")
		blob, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(blob); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadTokenizer(m); err == nil || !strings.Contains(err.Error(), "generation_config.json") {
			t.Fatalf("err = %v, want generation_config.json read error", err)
		}
	})
}
