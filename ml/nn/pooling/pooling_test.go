package pooling_test

import (
	"bytes"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	fsggml "github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/backend/ggml"
	"github.com/ollama/ollama/ml/nn/pooling"
)

func setup(tb testing.TB, n int) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := fsggml.WriteGGUF(f, fsggml.KV{
		"general.architecture": "test",
		"test.block_count":     uint32(1),
	}, []*fsggml.Tensor{
		{Name: "blk.0.weight", Shape: []uint64{1}, WriterTo: bytes.NewBuffer(make([]byte, 4))},
	}); err != nil {
		tb.Fatal(err)
	}

	b, err := ggml.New(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		tb.Fatal(err)
	}

	return b
}

func TestForward(t *testing.T) {
	cases := map[pooling.Type][]float32{
		pooling.TypeMean: {4, 5, 6, 7, 8, 9, 10, 11},
		pooling.TypeCLS:  {0, 1, 2, 3, 4, 5, 6, 7},
		pooling.TypeLast: {8, 9, 10, 11, 12, 13, 14, 15},
	}
	for typ, want := range cases {
		t.Run(typ.String(), func(t *testing.T) {
			b := setup(t, 99)
			defer b.Close()

			ctx := b.NewContext()
			defer ctx.Close()

			tt := ctx.Input().Arange(0, 16, 1, ml.DTypeF32).Reshape(ctx, 8, 2)
			tt = typ.Forward(ctx, tt)

			ctx.Forward(tt).Compute(tt)
			if diff := cmp.Diff(want, tt.Floats()); diff != "" {
				t.Error(diff)
			}
		})
	}
}
