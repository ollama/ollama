package manifest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
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

			ms, err := Manifests(true)
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

func TestColonHostManifestRoundTrip(t *testing.T) {
	d := t.TempDir()
	t.Setenv("OLLAMA_MODELS", d)

	n := model.ParseName("localhost:3000/library/tmp:latest")
	if err := WriteManifest(n, Layer{}, nil); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(n.Filepath(), ":") {
		t.Fatalf("Filepath %q still contains a colon", n.Filepath())
	}

	encoded := filepath.Join(d, "manifests", n.Filepath())
	if _, err := os.Stat(encoded); err != nil {
		t.Fatalf("encoded manifest %s: %v", encoded, err)
	}

	if _, err := ParseNamedManifest(n); err != nil {
		t.Fatalf("ParseNamedManifest: %v", err)
	}

	ms, err := Manifests(false)
	if err != nil {
		t.Fatal(err)
	}
	if !manifestListHas(ms, n) {
		t.Fatalf("manifest list missing %s", n)
	}
}

func TestLegacyColonHostManifestStillOpens(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("a directory name containing ':' cannot be created on Windows")
	}

	d := t.TempDir()
	t.Setenv("OLLAMA_MODELS", d)

	n := model.ParseName("localhost:3000/library/tmp:latest")
	legacy, ok := n.LegacyFilepath()
	if !ok {
		t.Fatal("expected legacy path")
	}
	createManifest(t, d, legacy)

	if _, err := ParseNamedManifest(n); err != nil {
		t.Fatalf("ParseNamedManifest legacy path: %v", err)
	}

	// A later write should update the existing colon directory rather than
	// creating a second encoded copy that list would also see.
	if err := WriteManifest(n, Layer{MediaType: "application/vnd.ollama.image.json"}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(d, "manifests", n.Filepath())); !os.IsNotExist(err) {
		t.Fatalf("write created a second encoded manifest: %v", err)
	}

	ms, err := Manifests(false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 1 {
		t.Fatalf("got %d manifests, want 1", len(ms))
	}
	if !manifestListHas(ms, n) {
		t.Fatalf("manifest list missing %s", n)
	}
}

func manifestListHas(ms map[model.Name]*Manifest, n model.Name) bool {
	for got := range ms {
		if got.EqualFold(n) {
			return true
		}
	}
	return false
}
