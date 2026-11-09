package parser

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// The directory holding a Modelfile is a literal path even when the FROM or
// DRAFT argument contains a glob. A sibling matching that literal name as a
// pattern must never be used as the root of the search.
func TestCreateRequestGlobUsesLiteralModelfileDirectory(t *testing.T) {
	names := []struct{ dir, sibling string }{
		{"models[Q4]", "modelsQ"},
		{"models[", "sibling"},
	}
	if runtime.GOOS != "windows" {
		names = append(names, struct{ dir, sibling string }{"models?", "modelsx"})
	}
	for _, tc := range names {
		t.Run(tc.dir, func(t *testing.T) {
			root := t.TempDir()
			modelDir := filepath.Join(root, tc.dir)
			sibling := filepath.Join(root, tc.sibling)
			for _, dir := range []string{modelDir, sibling} {
				if err := os.Mkdir(dir, 0o700); err != nil {
					t.Fatal(err)
				}
				for _, name := range []string{"model-1.gguf", "model-2.gguf", "model-extra.gguf", "draft-1.gguf"} {
					if err := os.WriteFile(filepath.Join(dir, name), []byte(dir), 0o600); err != nil {
						t.Fatal(err)
					}
				}
			}
			mf, err := ParseFile(strings.NewReader("FROM ./model-[12].gguf\nDRAFT ./draft-*.gguf\n"))
			if err != nil {
				t.Fatal(err)
			}
			req, err := mf.CreateRequest(modelDir)
			if err != nil {
				t.Fatal(err)
			}
			if len(req.Files) != 2 {
				t.Fatalf("expected two model shards in %q; got %v", modelDir, req.Files)
			}
			for _, name := range []string{"model-1.gguf", "model-2.gguf"} {
				if _, ok := req.Files[filepath.Join(modelDir, name)]; !ok {
					t.Errorf("missing intended model shard %q; got %v", name, req.Files)
				}
			}
			if len(req.DraftFiles) != 1 {
				t.Fatalf("expected one draft shard in %q; got %v", modelDir, req.DraftFiles)
			}
			if _, ok := req.DraftFiles[filepath.Join(modelDir, "draft-1.gguf")]; !ok {
				t.Errorf("missing intended draft shard; got %v", req.DraftFiles)
			}
		})
	}
}

func TestCreateRequestGlobParentOfLiteralModelfileDirectory(t *testing.T) {
	root := t.TempDir()
	parent := filepath.Join(root, "models[Q4]")
	sibling := filepath.Join(root, "modelsQ")
	for _, dir := range []string{parent, sibling} {
		if err := os.MkdirAll(filepath.Join(dir, "nested"), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "model-1.gguf"), []byte(dir), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	mf, err := ParseFile(strings.NewReader("FROM ../model-*.gguf\n"))
	if err != nil {
		t.Fatal(err)
	}
	req, err := mf.CreateRequest(filepath.Join(parent, "nested"))
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(parent, "model-1.gguf")
	if len(req.Files) != 1 {
		t.Fatalf("expected only %q; got %v", want, req.Files)
	}
	if _, ok := req.Files[want]; !ok {
		t.Fatalf("expected %q; got %v", want, req.Files)
	}
}
