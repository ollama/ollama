package ggml

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestWriteGGUF(t *testing.T) {
	tensorData := make([]byte, 2*3*4) // 6 F32 elements = 24 bytes
	for range 8 {
		t.Run("shuffle", func(t *testing.T) {
			t.Parallel()

			ts := []*Tensor{
				{Name: "token_embd.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.0.ffn_norm.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.0.attn_norm.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.1.ffn_up.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.2.ffn_norm.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.1.ffn_down.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "blk.0.attn_k.weight", Shape: []uint64{2, 3}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "output_norm.weight", Shape: []uint64{3, 2}, WriterTo: bytes.NewReader(tensorData)},
				{Name: "output.weight", Shape: []uint64{3, 2}, WriterTo: bytes.NewReader(tensorData)},
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

	t.Run("truncated_tensor_data", func(t *testing.T) {
		t.Parallel()

		ts := []*Tensor{
			{Name: "blk.0.attn.weight", Kind: 0, Shape: []uint64{512, 2}, WriterTo: bytes.NewBuffer(make([]byte, 32))},
		}

		w, err := os.CreateTemp(t.TempDir(), "truncated_*.bin")
		if err != nil {
			t.Fatal(err)
		}
		defer w.Close()

		if err := WriteGGUF(w, KV{"general.architecture": "test"}, ts); err != nil {
			t.Fatal(err)
		}

		r, err := os.Open(w.Name())
		if err != nil {
			t.Fatal(err)
		}
		defer r.Close()

		if _, err := Decode(r, -1); err == nil {
			t.Error("Decode should reject GGUF files where tensor data extends beyond file size")
		}
	})
}

// TestDecodeGGUFCorruptInputs verifies that Decode returns descriptive errors
// (rather than panicking) when presented with malformed or corrupt GGUF data.
//
// On unpatched code, the oversized-string and oversized-array cases below
// trigger a runtime panic "makeslice: len out of range": the large uint64
// length values wrap to -1 when cast to int, and make([]T, -1) panics.
// The too-many-dims case would cause an out-of-memory allocation for
// sufficiently large dim counts on corrupt files.
func TestDecodeGGUFCorruptInputs(t *testing.T) {
	// writeMinimalGGUF writes a minimal GGUF v3 file with one KV pair of the
	// given raw key/value bytes (caller is responsible for correct framing).
	// The KV byte slice must include the 8-byte length-prefixed key and 4-byte
	// type followed by whatever value bytes the caller provides.  n_kv is set
	// to 1; n_tensors is set to 0.
	writeMinimalGGUF := func(t *testing.T, kvBytes []byte) *os.File {
		t.Helper()
		f, err := os.CreateTemp(t.TempDir(), "minimal_*.bin")
		if err != nil {
			t.Fatal(err)
		}
		le := binary.LittleEndian
		binary.Write(f, le, []byte("GGUF")) // magic
		binary.Write(f, le, uint32(3))      // version
		binary.Write(f, le, uint64(0))      // n_tensors
		binary.Write(f, le, uint64(1))      // n_kv
		f.Write(kvBytes)
		return f
	}

	// writeKVString builds the raw bytes for a single string KV pair where the
	// value length is rawLength (written as a uint64).
	writeKVString := func(key string, rawLength uint64) []byte {
		var buf bytes.Buffer
		le := binary.LittleEndian
		binary.Write(&buf, le, uint64(len(key)))
		buf.WriteString(key)
		binary.Write(&buf, le, uint32(ggufTypeString)) // value type = string
		binary.Write(&buf, le, rawLength)              // string length (possibly invalid)
		// Omit the string payload so the file ends here; Decode will either
		// error on the oversized length before any read or on an unexpected EOF.
		return buf.Bytes()
	}

	t.Run("oversized_string_length_returns_error", func(t *testing.T) {
		t.Parallel()

		// Without the fix: int(math.MaxUint64) == -1 on 64-bit platforms,
		// so make([]byte, -1) panics with "makeslice: len out of range".
		for _, rawLen := range []uint64{
			math.MaxUint64,
			math.MaxInt64 + 1,
			1<<30 + 1, // just over the 1 GiB cap
		} {
			f := writeMinimalGGUF(t, writeKVString("general.architecture", rawLen))
			f.Seek(0, 0)
			_, err := Decode(f, -1)
			if err == nil {
				t.Errorf("rawLength=%d: expected error, got nil", rawLen)
			}
		}
	})

	t.Run("oversized_array_length_returns_error", func(t *testing.T) {
		t.Parallel()

		// Without the fix: int(math.MaxUint64) == -1, so newArray(-1, ...) panics.
		var buf bytes.Buffer
		le := binary.LittleEndian
		key := "tokenizer.ggml.tokens"
		binary.Write(&buf, le, uint64(len(key)))
		buf.WriteString(key)
		binary.Write(&buf, le, uint32(ggufTypeArray))  // value type = array
		binary.Write(&buf, le, uint32(ggufTypeString)) // array element type = string
		binary.Write(&buf, le, uint64(math.MaxUint64)) // n elements (huge)
		// No element data follows.

		f := writeMinimalGGUF(t, buf.Bytes())
		f.Seek(0, 0)
		_, err := Decode(f, -1)
		if err == nil {
			t.Error("expected error for oversized array, got nil")
		}
	})

	t.Run("too_many_tensor_dims_returns_error", func(t *testing.T) {
		t.Parallel()

		// Write a GGUF with one tensor whose dims field is 5 (> GGML_MAX_DIMS=4).
		f, err := os.CreateTemp(t.TempDir(), "bad_dims_*.bin")
		if err != nil {
			t.Fatal(err)
		}
		le := binary.LittleEndian
		binary.Write(f, le, []byte("GGUF"))
		binary.Write(f, le, uint32(3)) // version
		binary.Write(f, le, uint64(1)) // n_tensors = 1
		binary.Write(f, le, uint64(1)) // n_kv = 1

		// Write a minimal KV pair: general.architecture = "test"
		arch := "test"
		binary.Write(f, le, uint64(len("general.architecture")))
		f.WriteString("general.architecture")
		binary.Write(f, le, uint32(ggufTypeString))
		binary.Write(f, le, uint64(len(arch)))
		f.WriteString(arch)

		// Write one tensor info with dims = 5.
		name := "blk.0.weight"
		binary.Write(f, le, uint64(len(name)))
		f.WriteString(name)
		binary.Write(f, le, uint32(5)) // dims = 5, exceeds GGML_MAX_DIMS
		for i := 0; i < 5; i++ {
			binary.Write(f, le, uint64(4)) // shape dimension
		}
		binary.Write(f, le, uint32(0)) // kind = F32
		binary.Write(f, le, uint64(0)) // offset

		f.Seek(0, 0)
		_, err = Decode(f, -1)
		if err == nil {
			t.Error("expected error for tensor with >4 dims, got nil")
		}
	})
}
