package build

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/x/encoding/gguf"
	"github.com/ollama/ollama/x/model"
)

const qualifiedRef = "x/y/z:latest+Q4_0"

func TestServerBuildErrors(t *testing.T) {
	dir := t.TempDir()

	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("unqualified ref", func(t *testing.T) {
		err := s.Build("x", model.File{})
		if !errors.Is(err, ErrIncompleteRef) {
			t.Fatalf("Build() err = %v; want unqualified ref", err)
		}
	})

	t.Run("FROM pragma missing", func(t *testing.T) {
		err := s.Build(qualifiedRef, model.File{})
		var e *model.FileError
		if !errors.As(err, &e) {
			t.Fatalf("unexpected error: %v", err)
		}
		if e.Pragma != "FROM" {
			t.Errorf("e.Pragma = %s; want FROM", e.Pragma)
		}
		if e.Message != "missing" {
			t.Errorf("e.Message = %s; want missing", e.Message)
		}
	})

	t.Run("FROM file not found", func(t *testing.T) {
		err := s.Build(qualifiedRef, model.File{From: "bar"})
		if !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("Build() err = %v; want file not found", err)
		}
	})

	t.Run("FROM gguf", func(t *testing.T) {
		w := newWorkDir(t)
		// Write a gguf file without general.file_type metadata.
		w.write("gguf", ""+
			"GGUF"+ // magic
			"\x03\x00\x00\x00"+ // version
			"\x00\x00\x00\x00\x00\x00\x00\x00"+ // numMetaValues
			"\x00\x00\x00\x00\x00\x00\x00\x00"+ // numTensors
			"",
		)

		err := s.Build(qualifiedRef, model.File{From: w.fileName("gguf")})
		if !errors.Is(err, ErrMissingFileType) {
			t.Fatalf("Build() err = %#v; want missing file type", err)
		}
	})

	t.Run("FROM obscure dir", func(t *testing.T) {
		w := newWorkDir(t)
		w.mkdirAll("unknown")
		if err := s.Build(qualifiedRef, model.File{From: w.fileName("unknown")}); err != ErrUnsupportedModelFormat {
			t.Fatalf("Build() err = %#v; want unsupported model type", err)
		}
	})

	t.Run("FROM unsupported model type", func(t *testing.T) {
		w := newWorkDir(t)
		from := w.write("unknown", "unknown content")
		err := s.Build(qualifiedRef, model.File{From: from})
		if !errors.Is(err, ErrUnsupportedModelFormat) {
			t.Fatalf("Build() err = %#v; want unsupported model type", err)
		}
	})
}

func TestBuildBasicGGUF(t *testing.T) {
	w := newWorkDir(t)
	w.write("gguf", ""+
		"GGUF"+ // magic
		"\x03\x00\x00\x00"+ // version
		"\x00\x00\x00\x00\x00\x00\x00\x00"+ // numTensors
		"\x01\x00\x00\x00\x00\x00\x00\x00"+ // numMetaValues

		// general.file_type key
		"\x11\x00\x00\x00\x00\x00\x00\x00"+ // key length
		"general.file_type"+ // key
		"\x04\x00\x00\x00"+ // type (uint32)
		"\x02\x00\x00\x00\x00\x00\x00\x00"+ // uint32 value
		"",
	)

	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Build(qualifiedRef, model.File{From: w.fileName("gguf")}); err != nil {
		t.Fatal(err)
	}

	filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		t.Logf("file: %s", p)
		return nil
	})

	_, err = s.WeightsFile("unknown/y/z:latest+Q4_0")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("WeightsFile() err = %v; want not found", err)
	}

	path, err := s.WeightsFile("x/y/z:latest+Q4_0")
	if err != nil {
		t.Fatal(err)
	}

	info, err := gguf.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.FileType != gguf.TypeQ4_0 {
		t.Errorf("info.FileType = %d; want 1", info.FileType)
	}
}

type work struct {
	t   testing.TB
	dir string
}

func newWorkDir(t *testing.T) work {
	return work{t: t, dir: t.TempDir()}
}

func (w work) write(name, content string) (path string) {
	w.t.Helper()
	path = w.fileName(name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		w.t.Fatal(err)
	}
	return path
}

func (w work) fileName(name string) string {
	w.t.Helper()
	return filepath.Join(w.dir, name)
}

func (w work) mkdirAll(path string) {
	w.t.Helper()
	if err := os.MkdirAll(filepath.Join(w.dir, path), 0755); err != nil {
		w.t.Fatal(err)
	}
}
