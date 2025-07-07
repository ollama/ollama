package convert

import (
	"bytes"
	"encoding/binary"
	"io"
	"iter"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/pdevine/tensor"
)

type fakeTensor struct {
	name  string
	shape []uint64
	data  []float32

	repacker Repacker
}

func (f fakeTensor) Name() string {
	return f.name
}

func (f fakeTensor) Shape() []uint64 {
	return f.shape
}

func (f fakeTensor) Kind() uint32 {
	return 0
}

func (f *fakeTensor) SetRepacker(fn Repacker) {
	f.repacker = fn
}

func (f fakeTensor) Clone() Tensor {
	return &fakeTensor{
		name:     f.name,
		shape:    slices.Clone(f.shape),
		data:     slices.Clone(f.data),
		repacker: f.repacker,
	}
}

func (f fakeTensor) WriteTo(w io.Writer) (n int64, err error) {
	data := f.data
	if f.repacker != nil {
		data, err = f.repacker(f.name, data, f.shape)
		if err != nil {
			return 0, err
		}
	}

	if err := binary.Write(w, binary.LittleEndian, data); err != nil {
		return 0, err
	}

	return int64(len(data) * 4), nil
}

func mul(shape []uint64) int {
	n := 1
	for _, dim := range shape {
		n *= int(dim)
	}
	return n
}

func TestSplitDim(t *testing.T) {
	r := fakeTensor{
		name:  "a.b",
		shape: []uint64{3, 4},
		data:  []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
	}

	t.Run("no split", func(t *testing.T) {
		for tt := range splitDim(&r, 0, split{Replacer: strings.NewReplacer("a", "x")}) {
			if tt.Name != "x.b" {
				t.Fatalf("expected name 'x', got '%s'", tt.Name)
			}

			if !slices.Equal(tt.Shape, []uint64{3, 4}) {
				t.Fatalf("expected shape [3, 4], got %v", tt.Shape)
			}

			var b bytes.Buffer
			if _, err := tt.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, mul(tt.Shape))
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(f32s, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) {
				t.Fatalf("expected data [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], got %v", f32s)
			}
		}
	})

	t.Run("even split", func(t *testing.T) {
		next, stop := iter.Pull(splitDim(&r, 1,
			split{Replacer: strings.NewReplacer("a", "x")},
			split{Replacer: strings.NewReplacer("b", "y")},
		))
		defer stop()

		{
			tt, ok := next()
			if !ok {
				t.Fatal("expected at least one split")
			}

			if tt.Name != "x.b" {
				t.Fatal("expected name 'x.b', got", tt.Name)
			}

			if !slices.Equal(tt.Shape, []uint64{3, 2}) {
				t.Fatal("expected shape [3, 2], got", tt.Shape)
			}

			var b bytes.Buffer
			if _, err := tt.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, mul(tt.Shape))
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(f32s, []float32{0, 1, 4, 5, 8, 9}) {
				t.Fatal("expected data [0, 1, 4, 5, 8, 9], got", f32s)
			}
		}

		{
			tt, ok := next()
			if !ok {
				t.Fatal("expected at least one split")
			}

			if tt.Name != "a.y" {
				t.Fatal("expected name 'a.y', got", tt.Name)
			}

			if !slices.Equal(tt.Shape, []uint64{3, 2}) {
				t.Fatal("expected shape [3, 2], got", tt.Shape)
			}

			var b bytes.Buffer
			if _, err := tt.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, mul(tt.Shape))
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(f32s, []float32{2, 3, 6, 7, 10, 11}) {
				t.Fatal("expected data [2, 3, 6, 7, 10, 11], got", f32s)
			}
		}
	})

	t.Run("uneven split", func(t *testing.T) {
		next, stop := iter.Pull(splitDim(&r, 0,
			split{Replacer: strings.NewReplacer("a", "x"), dim: 2},
			split{Replacer: strings.NewReplacer("b", "y"), dim: 1},
		))
		defer stop()

		{
			tt, ok := next()
			if !ok {
				t.Fatal("expected at least one split")
			}

			if tt.Name != "x.b" {
				t.Fatal("expected name 'x.b', got", tt.Name)
			}

			if !slices.Equal(tt.Shape, []uint64{2, 4}) {
				t.Fatal("expected shape [2, 4], got", tt.Shape)
			}

			var b bytes.Buffer
			if _, err := tt.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, mul(tt.Shape))
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(f32s, []float32{0, 1, 2, 3, 4, 5, 6, 7}) {
				t.Fatal("expected data [0, 1, 2, 3, 4, 5, 6, 7], got", f32s)
			}
		}

		{
			tt, ok := next()
			if !ok {
				t.Fatal("expected at least one split")
			}

			if tt.Name != "a.y" {
				t.Fatal("expected name 'a.y', got", tt.Name)
			}

			if !slices.Equal(tt.Shape, []uint64{1, 4}) {
				t.Fatal("expected shape [1, 4], got", tt.Shape)
			}

			var b bytes.Buffer
			if _, err := tt.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, mul(tt.Shape))
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(f32s, []float32{8, 9, 10, 11}) {
				t.Fatal("expected data [8, 9, 10, 11], got", f32s)
			}
		}
	})

	t.Run("split with transpose", func(t *testing.T) {
		next, stop := iter.Pull(splitDim(&r, 1,
			split{Replacer: strings.NewReplacer("a", "x")},
			split{Replacer: strings.NewReplacer("b", "y"), fn: func(tt tensor.Tensor) (tensor.Tensor, error) {
				return tensor.Transpose(tt, 1, 0)
			}},
		))
		defer stop()

		{
			tt, ok := next()
			if !ok {
				t.Fatal("expected at least one split")
			}

			if tt.Name != "x.b" {
				t.Fatal("expected name 'x.b', got", tt.Name)
			}

			if !slices.Equal(tt.Shape, []uint64{3, 2}) {
				t.Fatal("expected shape [3, 2], got", tt.Shape)
			}

			var b bytes.Buffer
			if _, err := tt.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, mul(tt.Shape))
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(f32s, []float32{0, 1, 4, 5, 8, 9}) {
				t.Fatal("expected data [0, 1, 4, 5, 8, 9], got", f32s)
			}
		}

		{
			tt, ok := next()
			if !ok {
				t.Fatal("expected at least one split")
			}

			if tt.Name != "a.y" {
				t.Fatal("expected name 'a.y', got", tt.Name)
			}

			if !slices.Equal(tt.Shape, []uint64{3, 2}) {
				t.Fatal("expected shape [3, 2], got", tt.Shape)
			}

			var b bytes.Buffer
			if _, err := tt.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, mul(tt.Shape))
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(f32s, []float32{2, 6, 10, 3, 7, 11}) {
				t.Fatal("expected data [2, 6, 10, 3, 7, 11], got", f32s)
			}
		}
	})
}

func TestMerge(t *testing.T) {
	unmatched := []Tensor{
		&fakeTensor{
			name:  "a.0.b",
			shape: []uint64{5, 2},
			data:  []float32{10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
		},
		&fakeTensor{
			name:  "a.1.b",
			shape: []uint64{5, 2},
			data:  []float32{20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
		},
		&fakeTensor{
			name:  "c.0.d",
			shape: []uint64{5, 2},
			data:  []float32{30, 31, 32, 33, 34, 35, 36, 37, 38, 39},
		},
		&fakeTensor{
			name:  "c.1.d",
			shape: []uint64{5, 2},
			data:  []float32{40, 41, 42, 43, 44, 45, 46, 47, 48, 49},
		},
		&fakeTensor{
			name:  "e.0.f",
			shape: []uint64{5, 2},
			data:  []float32{50, 51, 52, 53, 54, 55, 56, 57, 58, 59},
		},
	}

	checkMatched := func(t *testing.T, n int, matched []*ggml.Tensor) {
		for i := range n {
			got := matched[i]
			if diff := cmp.Diff([]uint64{2, 5, 2}, got.Shape); diff != "" {
				t.Errorf("unexpected (-want +got):\n%s", diff)
			}

			var b bytes.Buffer
			if _, err := got.WriteTo(&b); err != nil {
				t.Fatal(err)
			}

			f32s := make([]float32, 20)
			if err := binary.Read(&b, binary.LittleEndian, &f32s); err != nil {
				t.Fatal(err)
			}

			offset := 10 + (i * 20)
			want := make([]float32, 20)
			for j := range 20 {
				want[j] = float32(offset + j)
			}

			if diff := cmp.Diff(want, f32s); diff != "" {
				t.Errorf("unexpected data (-want +got):\n%s", diff)
			}
		}
	}

	t.Run("single merge", func(t *testing.T) {
		matched, unmatched := mergeTensors(unmatched, merge{"a.*.b", "a.b"})
		if len(unmatched) != 3 {
			t.Error("expected 3 remaining tensors, got", len(unmatched))
		}

		if len(matched) != 1 {
			t.Error("expected 1 merged tensor, got", len(matched))
		}

		checkMatched(t, 1, matched)
	})

	t.Run("multiple merges", func(t *testing.T) {
		matched, unmatched := mergeTensors(unmatched, merge{"a.*.b", "a.b"}, merge{"c.*.d", "c.d"})
		if len(unmatched) != 1 {
			t.Error("expected 1 remaining tensors, got", len(unmatched))
		}

		if len(matched) != 2 {
			t.Error("expected 2 merged tensor, got", len(matched))
		}

		checkMatched(t, 2, matched)
	})

	t.Run("no match", func(t *testing.T) {
		matched, unmatched := mergeTensors(unmatched, merge{"x.*.y", "x.y"})
		if len(unmatched) != 5 {
			t.Error("expected 5 remaining tensors, got", len(unmatched))
		}

		if len(matched) != 0 {
			t.Error("expected no merged tensors, got", len(matched))
		}
	})
}
