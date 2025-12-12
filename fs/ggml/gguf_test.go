package ggml

import (
	"bytes"
	"math/rand/v2"
	"os"
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
				"general.architecture": "test",
				"general.alignment":    uint32(16),
				"test.key":             "value",
				"test.int32_key":       int32(-42),
				"test.int64_key":       int64(-9223372036854775808),
				"test.int32_array":     []int32{-1, 0, 1, 2147483647, -2147483648},
				"test.int64_array":     []int64{-1, 0, 1, 9223372036854775807, -9223372036854775808},
				"attention.key":        "value2",
				"tokenizer.key":        "value3",
				"adapter.key":          "value4",
			}, ts); err != nil {
				t.Fatal(err)
			}

			r, err := os.Open(w.Name())
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()

			ff, err := Decode(r, -1)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(KV{
				"general.architecture":    "test",
				"general.alignment":       uint32(16),
				"general.parameter_count": uint64(54),
				"test.key":                "value",
				"test.int32_key":          int32(-42),
				"test.int64_key":          int64(-9223372036854775808),
				"test.int32_array":        &array[int32]{size: 5, values: []int32{-1, 0, 1, 2147483647, -2147483648}},
				"test.int64_array":        &array[int64]{size: 5, values: []int64{-1, 0, 1, 9223372036854775807, -9223372036854775808}},
				"test.attention.key":      "value2",
				"tokenizer.key":           "value3",
				"adapter.key":             "value4",
			}, ff.KV(), cmp.AllowUnexported(array[int32]{}, array[int64]{})); diff != "" {
				t.Errorf("Mismatch (-want +got):\n%s", diff)
			}

			if diff := cmp.Diff(Tensors{
				Offset: 992,
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
