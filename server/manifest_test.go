package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func createManifest(t *testing.T, path, name string) {
	t.Helper()

	p := filepath.Join(path, "manifests", name)
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}

	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := json.NewEncoder(f).Encode(ManifestV2{}); err != nil {
		t.Fatal(err)
	}
}

func TestManifests(t *testing.T) {
	cases := map[string][]string{
		"empty": {},
		"single": {
			filepath.Join("host", "namespace", "model", "tag"),
		},
		"multiple": {
			filepath.Join("registry.ollama.ai", "library", "llama3", "latest"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q4_0"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q4_1"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q8_0"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q5_0"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q5_1"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q2_K"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_S"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_M"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q3_K_L"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q4_K_S"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q4_K_M"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q5_K_S"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q5_K_M"),
			filepath.Join("registry.ollama.ai", "library", "llama3", "q6_K"),
		},
		"hidden": {
			filepath.Join("host", "namespace", "model", "tag"),
			filepath.Join("host", "namespace", "model", ".hidden"),
		},
	}

	for n, wants := range cases {
		t.Run(n, func(t *testing.T) {
			d := t.TempDir()
			t.Setenv("OLLAMA_MODELS", d)

			for _, want := range wants {
				createManifest(t, d, want)
			}

			ms, err := Manifests()
			if err != nil {
				t.Fatal(err)
			}

			var ns []model.Name
			for k := range ms {
				ns = append(ns, k)
			}

			for _, want := range wants {
				n := model.ParseNameFromFilepath(want)
				if !n.IsValid() && slices.Contains(ns, n) {
					t.Errorf("unexpected invalid name: %s", want)
				} else if n.IsValid() && !slices.Contains(ns, n) {
					t.Errorf("missing valid name: %s", want)
				}
			}
		})
	}
}
