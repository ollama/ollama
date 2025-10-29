package ggml

import (
	"errors"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

func setup(tb testing.TB) ml.Context {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, ggml.KV{"general.architecture": "test"}, nil); err != nil {
		tb.Fatal(err)
	}

	b, err := ml.NewBackend(f.Name(), ml.BackendParams{})
	if err != nil {
		tb.Fatal(err)
	}

	ctx := b.NewContext().Input()

	tb.Cleanup(func() {
		ctx.Close()
		b.Close()
	})

	return ctx
}

func TestInferShape(t *testing.T) {
	cases := []struct {
		name  string
		input []int
		want  []int
		err   error
	}{
		{
			name:  "no inferred shape",
			input: []int{2, 3, 4},
			want:  []int{2, 3, 4},
		},
		{
			name:  "infer begin",
			input: []int{-1, 3, 4},
			want:  []int{2, 3, 4},
		},
		{
			name:  "infer mid",
			input: []int{2, -1, 4},
			want:  []int{2, 3, 4},
		},
		{
			name:  "infer end",
			input: []int{2, 3, -1},
			want:  []int{2, 3, 4},
		},
		{
			name:  "too many inferred dims",
			input: []int{-1, 3, -1},
			err:   errors.New("only one dimension can be inferred"),
		},
		{
			name:  "infer gather",
			input: []int{2, -1},
			want:  []int{2, 12},
		},
		{
			name:  "infer gather all",
			input: []int{-1},
			want:  []int{24},
		},
		{
			name:  "infer split",
			input: []int{2, -1, 3, 2},
			want:  []int{2, 2, 3, 2},
		},
		{
			name:  "indivisible infer",
			input: []int{2, -1, 2, 4},
			err:   errors.New("cannot infer dimension"),
		},
		{
			name:  "infer zero dim",
			input: []int{2, 0, 4},
			err:   errors.New("dimension cannot be zero"),
		},
	}

	ctx := setup(t)
	tensor, ok := ctx.Empty(ml.DTypeF32, 2, 3, 4).(*Tensor)
	if !ok {
		t.Fatal("expected *Tensor")
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil && tt.err == nil {
					// all good
				} else if r != nil && tt.err == nil {
					t.Errorf("unexpected panic: %v", r)
				} else if r == nil && tt.err != nil {
					t.Errorf("expected panic but did not get one: %v", tt.err)
				} else if errStr, ok := r.(string); ok && errStr != tt.err.Error() {
					t.Errorf("expected panic %q but got %q", tt.err.Error(), errStr)
				}
			}()

			inferShape(tensor, tt.input)
			if diff := cmp.Diff(tt.want, tt.input); diff != "" {
				t.Errorf("%s: shape mismatch (-want +got):\n%s", tt.name, diff)
			}
		})
	}
}
