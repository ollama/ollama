package llm

import (
	"os"
	"testing"
)

func TestStatusWriterCapturesErrorLine(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "status-writer")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	w := NewStatusWriter(f)
	if _, err := w.Write([]byte("llama_init_from_model: failed to initialize the context: failed to initialize Metal backend\n")); err != nil {
		t.Fatal(err)
	}

	if got, want := w.LastError(), "llama_init_from_model: failed to initialize the context: failed to initialize Metal backend"; got != want {
		t.Fatalf("LastError = %q, want %q", got, want)
	}
}

func TestStatusWriterAccumulatesErrorLines(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "status-writer")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	w := NewStatusWriter(f)
	if _, err := w.Write([]byte("error: failed to initialize the Metal library\n")); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("GGML_ASSERT([rsets->data count] == 0) failed\n")); err != nil {
		t.Fatal(err)
	}

	want := "error: failed to initialize the Metal library\nGGML_ASSERT([rsets->data count] == 0) failed"
	if got := w.LastError(); got != want {
		t.Fatalf("LastError = %q, want %q", got, want)
	}
}
