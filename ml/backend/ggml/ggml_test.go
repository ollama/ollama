package ggml

import (
	"bytes"
	"log/slog"
	"os"
	"slices"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

func TestMain(m *testing.M) {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	os.Exit(m.Run())
}

func setup(tb testing.TB) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, ggml.KV{
		"general.architecture": "test",
		"test.block_count":     uint32(1),
	}, []*ggml.Tensor{
		{Name: "blk.0.weight", Shape: []uint64{1}, WriterTo: bytes.NewBuffer(slices.Repeat([]byte{0}, 4))},
	}); err != nil {
		tb.Fatal(err)
	}

	b, err := New(f.Name(), ml.BackendParams{NumGPULayers: 1})
	if err != nil {
		tb.Fatal(err)
	}

	return b
}

// initContextOrSkip takes a testing.T and true for GPU
// If GPUs are not available, the current test is skipped
// gpu=false will always succed
func initContextOrSkip(t *testing.T, b ml.Backend, gpu bool) ml.Context {
	if gpu && len(b.(*Backend).schedBackends) == 1 {
		t.Skip("No GPU detected, skipping GPU test case")
	}
	ctx := b.NewContext()
	t.Cleanup(func() { ctx.Close() })
	if gpu {
		return ctx.Layer(0)
	}
	return ctx.Input()
}
