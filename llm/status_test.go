package llm

import (
	"io"
	"testing"
)

func TestStatusWriterCapturesErrorLine(t *testing.T) {
	tests := []struct {
		name string
		log  string
		want string
	}{
		{
			name: "llama init",
			log:  "llama_init_from_model: failed to initialize the context: failed to initialize Metal backend\n",
			want: "llama_init_from_model: failed to initialize the context: failed to initialize Metal backend",
		},
		{
			name: "cobra error",
			log:  "Error: foo baz bar\n",
			want: "Error: foo baz bar",
		},
		{
			name: "uppercase mlx",
			log:  "MLX: there was an error\n",
			want: "MLX: there was an error",
		},
		{
			name: "panic header",
			log: "time=2026-05-01T15:36:45.053Z level=INFO source=pipeline.go:71 msg=\"peak memory\" size=\"8.26 GiB\"\n" +
				"panic: mlx: Failed to compile kernel: nvrtc: error: invalid value for --gpu-architecture (-arch)\n" +
				"\t. at /go/src/github.com/ollama/ollama/build/_deps/mlx-c-src/mlx/c/transforms.cpp:15\n\n" +
				"goroutine 31 [running]:\n" +
				"golang.org/x/sync/errgroup.(*Group).Go.func1()\n" +
				"\tgolang.org/x/sync@v0.17.0/errgroup/errgroup.go:93 +0x50\n",
			want: "panic: mlx: Failed to compile kernel: nvrtc: error: invalid value for --gpu-architecture (-arch)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := NewStatusWriter(io.Discard)
			if _, err := w.Write([]byte(tt.log)); err != nil {
				t.Fatal(err)
			}

			if got := w.LastError(); got != tt.want {
				t.Fatalf("LastError = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestStatusWriterAccumulatesErrorLines(t *testing.T) {
	w := NewStatusWriter(io.Discard)
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
