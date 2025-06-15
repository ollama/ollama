package ggml

import (
	"bytes"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWriteGGUF(t *testing.T) {
	w, err := os.CreateTemp(t.TempDir(), "*.bin")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	if err := WriteGGUF(w, KV{
		"general.alignment": uint32(16),
	}, []*Tensor{
		{Name: "test.0", Shape: []uint64{2, 3}, WriterTo: bytes.NewBuffer(slices.Repeat([]byte{0}, 2*3*4))},
		{Name: "test.1", Shape: []uint64{2, 3}, WriterTo: bytes.NewBuffer(slices.Repeat([]byte{0}, 2*3*4))},
		{Name: "test.2", Shape: []uint64{2, 3}, WriterTo: bytes.NewBuffer(slices.Repeat([]byte{0}, 2*3*4))},
		{Name: "test.3", Shape: []uint64{2, 3}, WriterTo: bytes.NewBuffer(slices.Repeat([]byte{0}, 2*3*4))},
		{Name: "test.4", Shape: []uint64{2, 3}, WriterTo: bytes.NewBuffer(slices.Repeat([]byte{0}, 2*3*4))},
		{Name: "test.5", Shape: []uint64{2, 3}, WriterTo: bytes.NewBuffer(slices.Repeat([]byte{0}, 2*3*4))},
	}); err != nil {
		t.Fatal(err)
	}

	r, err := os.Open(w.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	ff, err := Decode(r, 0)
	if err != nil {
		t.Fatal(err)
	}

	if diff := cmp.Diff(ff.KV(), KV{
		"general.alignment":       uint32(16),
		"general.parameter_count": uint64(36),
	}); diff != "" {
		t.Errorf("Mismatch (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff(ff.Tensors(), Tensors{
		Offset: 336,
		items: []*Tensor{
			{Name: "test.0", Offset: 0, Shape: []uint64{2, 3}},
			{Name: "test.1", Offset: 32, Shape: []uint64{2, 3}},
			{Name: "test.2", Offset: 64, Shape: []uint64{2, 3}},
			{Name: "test.3", Offset: 96, Shape: []uint64{2, 3}},
			{Name: "test.4", Offset: 128, Shape: []uint64{2, 3}},
			{Name: "test.5", Offset: 160, Shape: []uint64{2, 3}},
		},
	}, cmp.AllowUnexported(Tensors{})); diff != "" {
		t.Errorf("Mismatch (-want +got):\n%s", diff)
	}
}

func BenchmarkReadArray(b *testing.B) {
	b.ReportAllocs()

	create := func(tb testing.TB, kv KV) string {
		tb.Helper()
		f, err := os.CreateTemp(b.TempDir(), "")
		if err != nil {
			b.Fatal(err)
		}
		defer f.Close()

		if err := WriteGGUF(f, kv, nil); err != nil {
			b.Fatal(err)
		}

		return f.Name()
	}

	cases := map[string]any{
		"int32":   slices.Repeat([]int32{42}, 1_000_000),
		"uint32":  slices.Repeat([]uint32{42}, 1_000_000),
		"float32": slices.Repeat([]float32{42.}, 1_000_000),
		"string":  slices.Repeat([]string{"42"}, 1_000_000),
	}

		for name, bb := range cases {
	for _, maxArraySize := range []int{-1, 0, 1024} {
			b.Run(name+"-maxArraySize="+strconv.Itoa(maxArraySize), func(b *testing.B) {
				p := create(b, KV{"array": bb})
				for b.Loop() {
					f, err := os.Open(p)
					if err != nil {
						b.Fatal(err)
					}
					if _, err := Decode(f, maxArraySize); err != nil {
						b.Fatal(err)
					}
					f.Close()
				}
			})
		}
	}
}
