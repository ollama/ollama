package hindsight

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	"github.com/ollama/ollama/types/model"
)

// The Qwen3.5 tokenizer hs-decide-1-flash ships with; every qwen3.5 tag in
// the library carries the same file.
const qwen35TokenizerDigest = "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"

func qwen35Tokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	for _, name := range []string{"qwen3.5:0.8b-mxfp8", "qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b"} {
		m, err := manifest.ParseNamedManifestForRunner(model.ParseName(name), manifest.RunnerMLX)
		if errors.Is(err, os.ErrNotExist) || errors.Is(err, manifest.ErrNoCompatibleManifest) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		layer, ok := m.ConfigLayer("tokenizer.json")
		if !ok || layer.Digest != qwen35TokenizerDigest {
			continue
		}
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		tok, err := tokenizer.LoadFromBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		return tok
	}
	t.Skip("no installed qwen3.5 model carries the Qwen3.5 tokenizer; pull qwen3.5:0.8b to run")
	return nil
}

type renderFixture struct {
	Name    string          `json:"name"`
	Request json.RawMessage `json:"request"`
	Tokens  int             `json:"tokens"`
	SHA256  string          `json:"sha256"`
	Head    []int32         `json:"head"`
	Tail    []int32         `json:"tail"`
	Slots   []int           `json:"slots"`
	Groups  [][][]int32     `json:"groups"`
}

// The fixtures are the reference Python renderer's output for these
// requests, so the Go port must match it token for token.
func TestRenderMatchesReferenceRenderer(t *testing.T) {
	data, err := os.ReadFile("testdata/render.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures []renderFixture
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	m := &Model{}
	if err := m.initCodes(qwen35Tokenizer(t)); err != nil {
		t.Fatal(err)
	}
	for _, f := range fixtures {
		t.Run(f.Name, func(t *testing.T) {
			var body struct {
				State     json.RawMessage                          `json:"state"`
				Questions *orderedmap.Map[string, json.RawMessage] `json:"questions"`
			}
			if err := json.Unmarshal(f.Request, &body); err != nil {
				t.Fatal(err)
			}
			native := &llm.NativeDecision{State: body.State}
			for name, q := range body.Questions.All() {
				native.Questions = append(native.Questions, llm.NativeQuestion{Name: name, Question: q})
			}
			r, err := m.render(native)
			if err != nil {
				t.Fatal(err)
			}
			if len(r.ids) != f.Tokens {
				t.Fatalf("%d tokens, want %d", len(r.ids), f.Tokens)
			}
			if !slices.Equal(r.ids[:len(f.Head)], f.Head) || !slices.Equal(r.ids[len(r.ids)-len(f.Tail):], f.Tail) {
				t.Fatalf("head %v tail %v, want head %v tail %v", r.ids[:len(f.Head)], r.ids[len(r.ids)-len(f.Tail):], f.Head, f.Tail)
			}
			if got := idsDigest(r.ids); got != f.SHA256 {
				t.Fatalf("token digest %s, want %s", got, f.SHA256)
			}
			if !slices.Equal(r.slots, f.Slots) {
				t.Fatalf("slots %v, want %v", r.slots, f.Slots)
			}
			for i := range f.Groups {
				if !slices.EqualFunc(r.groups[i], f.Groups[i], slices.Equal) {
					t.Fatalf("question %d option tokens %v, want %v", i, r.groups[i], f.Groups[i])
				}
			}
		})
	}
}

func idsDigest(ids []int32) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.Itoa(int(id))
	}
	return fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(parts, ","))))
}

func TestPairCodes(t *testing.T) {
	m := &Model{}
	if err := m.initCodes(qwen35Tokenizer(t)); err != nil {
		t.Fatal(err)
	}
	if len(m.codes) != maxChoices || m.codes[0] != "A" || m.codes[25] != "Z" || len(m.codes[26]) != 2 {
		t.Fatalf("codes = %v", m.codes)
	}
	for _, code := range m.codes {
		if m.singleToken(code) < 0 || m.singleToken(" "+code) < 0 {
			t.Fatalf("code %q is not one token bare and after a space", code)
		}
	}
}
