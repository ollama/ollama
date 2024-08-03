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
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}

	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := json.NewEncoder(f).Encode(Manifest{}); err != nil {
		t.Fatal(err)
	}
}

func TestManifests(t *testing.T) {
	cases := map[string]struct {
		ps               []string
		wantValidCount   int
		wantInvalidCount int
	}{
		"empty": {},
		"single": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag"),
			},
			wantValidCount: 1,
		},
		"multiple": {
			ps: []string{
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
			wantValidCount: 15,
		},
		"hidden": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag"),
				filepath.Join("host", "namespace", "model", ".hidden"),
			},
			wantValidCount:   1,
			wantInvalidCount: 1,
		},
		"subdir": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "tag", "one"),
				filepath.Join("host", "namespace", "model", "tag", "another", "one"),
			},
			wantInvalidCount: 2,
		},
		"upper tag": {
			ps: []string{
				filepath.Join("host", "namespace", "model", "TAG"),
			},
			wantValidCount: 1,
		},
		"upper model": {
			ps: []string{
				filepath.Join("host", "namespace", "MODEL", "tag"),
			},
			wantValidCount: 1,
		},
		"upper namespace": {
			ps: []string{
				filepath.Join("host", "NAMESPACE", "model", "tag"),
			},
			wantValidCount: 1,
		},
		"upper host": {
			ps: []string{
				filepath.Join("HOST", "namespace", "model", "tag"),
			},
			wantValidCount: 1,
		},
	}

	for n, wants := range cases {
		t.Run(n, func(t *testing.T) {
			d := t.TempDir()
			t.Setenv("OLLAMA_MODELS", d)

			for _, p := range wants.ps {
				createManifest(t, d, p)
			}

			ms, err := Manifests()
			if err != nil {
				t.Fatal(err)
			}

			var ns []model.Name
			for k := range ms {
				ns = append(ns, k)
			}

			var gotValidCount, gotInvalidCount int
			for _, p := range wants.ps {
				n := model.ParseNameFromFilepath(p)
				if n.IsValid() {
					gotValidCount++
				} else {
					gotInvalidCount++
				}

				if !n.IsValid() && slices.Contains(ns, n) {
					t.Errorf("unexpected invalid name: %s", p)
				} else if n.IsValid() && !slices.Contains(ns, n) {
					t.Errorf("missing valid name: %s", p)
				}
			}

			if gotValidCount != wants.wantValidCount {
				t.Errorf("got valid count %d, want %d", gotValidCount, wants.wantValidCount)
			}

			if gotInvalidCount != wants.wantInvalidCount {
				t.Errorf("got invalid count %d, want %d", gotInvalidCount, wants.wantInvalidCount)
			}
		})
	}
}
