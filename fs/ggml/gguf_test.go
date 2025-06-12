package ggml

import (
	"bytes"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWriteGGUF(t *testing.T) {
	b := bytes.NewBuffer(make([]byte, 2*3))
	for range 8 {
		t.Run("shuffle", func(t *testing.T) {
			t.Parallel()

			ts := []*Tensor{
				{Name: "token_embd.weight", Shape: []uint64{2, 3}, WriterTo: b},
				{Name: "blk.0.ffn_norm.weight", Shape: []uint64{2, 3}, WriterTo: b},
				{Name: "blk.0.attn_norm.weight", Shape: []uint64{2, 3}, WriterTo: b},
				{Name: "blk.1.ffn_up.weight", Shape: []uint64{2, 3}, WriterTo: b},
				{Name: "blk.2.ffn_norm.weight", Shape: []uint64{2, 3}, WriterTo: b},
				{Name: "blk.1.ffn_down.weight", Shape: []uint64{2, 3}, WriterTo: b},
				{Name: "blk.0.attn_k.weight", Shape: []uint64{2, 3}, WriterTo: b},
				{Name: "output_norm.weight", Shape: []uint64{3, 2}, WriterTo: b},
				{Name: "output.weight", Shape: []uint64{3, 2}, WriterTo: b},
			}

			rand.Shuffle(len(ts), func(i, j int) {
				ts[i], ts[j] = ts[j], ts[i]
			})

			w, err := os.CreateTemp(t.TempDir(), strings.ReplaceAll(t.Name(), "/", "_")+"*.bin")
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()

			if err := WriteGGUF(w, KV{
				"general.alignment": uint32(16),
			}, ts); err != nil {
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

			if diff := cmp.Diff(KV{
				"general.alignment":       uint32(16),
				"general.parameter_count": uint64(54),
			}, ff.KV()); diff != "" {
				t.Errorf("Mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(Tensors{
				Offset: 592,
				items: []*Tensor{
					{Name: "blk.0.attn_k.weight", Offset: 0, Shape: []uint64{2, 3}},
					{Name: "blk.0.attn_norm.weight", Offset: 32, Shape: []uint64{2, 3}},
					{Name: "blk.0.ffn_norm.weight", Offset: 64, Shape: []uint64{2, 3}},
					{Name: "blk.1.ffn_down.weight", Offset: 96, Shape: []uint64{2, 3}},
					{Name: "blk.1.ffn_up.weight", Offset: 128, Shape: []uint64{2, 3}},
					{Name: "blk.2.ffn_norm.weight", Offset: 160, Shape: []uint64{2, 3}},
					{Name: "output.weight", Offset: 192, Shape: []uint64{3, 2}},
					{Name: "output_norm.weight", Offset: 224, Shape: []uint64{3, 2}},
					{Name: "token_embd.weight", Offset: 256, Shape: []uint64{2, 3}},
				},
			}, ff.Tensors(), cmp.AllowUnexported(Tensors{})); diff != "" {
				t.Errorf("Mismatch (-want +got):\n%s", diff)
			}
		})
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
