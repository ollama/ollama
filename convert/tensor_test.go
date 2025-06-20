package convert

import (
	"bytes"
	"encoding/binary"
	"io"
	"iter"
	"slices"
	"strings"
	"testing"

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
