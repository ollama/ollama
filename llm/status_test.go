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

func TestStatusWriterCapturesPanicHeader(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "status-writer")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	w := NewStatusWriter(f)
	log := "time=2026-05-01T15:36:45.053Z level=INFO source=pipeline.go:71 msg=\"peak memory\" size=\"8.26 GiB\"\n" +
		"panic: mlx: Failed to compile kernel: nvrtc: error: invalid value for --gpu-architecture (-arch)\n" +
		"\t. at /go/src/github.com/ollama/ollama/build/_deps/mlx-c-src/mlx/c/transforms.cpp:15\n\n" +
		"goroutine 31 [running]:\n" +
		"golang.org/x/sync/errgroup.(*Group).Go.func1()\n" +
		"\tgolang.org/x/sync@v0.17.0/errgroup/errgroup.go:93 +0x50\n"
	if _, err := w.Write([]byte(log)); err != nil {
		t.Fatal(err)
	}

	want := "panic: mlx: Failed to compile kernel: nvrtc: error: invalid value for --gpu-architecture (-arch)"
	if got := w.LastError(); got != want {
		t.Fatalf("LastError = %q, want %q", got, want)
	}
}
