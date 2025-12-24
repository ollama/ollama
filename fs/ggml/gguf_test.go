package ggml

import (
	"bytes"
	"encoding/binary"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// FuzzGGUFDecode tests the GGUF binary format parser with random inputs
func FuzzGGUFDecode(f *testing.F) {
	// Add seed corpus with valid GGUF structures

	// Minimal valid GGUF v3 with no KVs and no tensors
	minimalV3 := makeGGUFHeader(3, 0, 0)
	f.Add(minimalV3)

	// GGUF v3 with 1 KV (string type)
	withKV := makeGGUFWithStringKV(3, "test.key", "value")
	f.Add(withKV)

	// GGUF v3 with 1 tensor
	withTensor := makeGGUFWithTensor(3)
	f.Add(withTensor)

	// GGUF v2 minimal
	minimalV2 := makeGGUFHeader(2, 0, 0)
	f.Add(minimalV2)

	// GGUF v1 minimal
	minimalV1 := makeGGUFHeaderV1(1, 0, 0)
	f.Add(minimalV1)

	// Invalid magic
	invalidMagic := []byte{0x00, 0x00, 0x00, 0x00}
	f.Add(invalidMagic)

	// Truncated header
	f.Add([]byte{0x47, 0x47, 0x55, 0x46}) // Just magic

	// Big endian magic
	beMagic := []byte{0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	f.Add(beMagic)

	// Very large numKV (potential DoS)
	largeKV := makeGGUFHeader(3, 0, 0)
	binary.LittleEndian.PutUint64(largeKV[16:24], 0xFFFFFFFFFFFFFFFF)
	f.Add(largeKV)

	// Very large numTensor (potential DoS)
	largeTensor := makeGGUFHeader(3, 0, 0)
	binary.LittleEndian.PutUint64(largeTensor[8:16], 0xFFFFFFFFFFFFFFFF)
	f.Add(largeTensor)

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 4 {
			return
		}

		r := bytes.NewReader(data)

		// Try to decode - we're looking for panics, hangs, or memory issues
		_, _ = Decode(r, 1024) // Limit array size to prevent OOM
	})
}

// Helper functions to create valid GGUF structures for corpus

func makeGGUFHeader(version uint32, numTensor, numKV uint64) []byte {
	buf := new(bytes.Buffer)
	// Magic (little endian)
	binary.Write(buf, binary.LittleEndian, uint32(FILE_MAGIC_GGUF_LE))
	// Version
	binary.Write(buf, binary.LittleEndian, version)
	// NumTensor (v2+ uses uint64)
	binary.Write(buf, binary.LittleEndian, numTensor)
	// NumKV
	binary.Write(buf, binary.LittleEndian, numKV)
	return buf.Bytes()
}

func makeGGUFHeaderV1(version uint32, numTensor, numKV uint32) []byte {
	buf := new(bytes.Buffer)
	// Magic (little endian)
	binary.Write(buf, binary.LittleEndian, uint32(FILE_MAGIC_GGUF_LE))
	// Version
	binary.Write(buf, binary.LittleEndian, version)
	// NumTensor (v1 uses uint32)
	binary.Write(buf, binary.LittleEndian, numTensor)
	// NumKV
	binary.Write(buf, binary.LittleEndian, numKV)
	return buf.Bytes()
}

func makeGGUFWithStringKV(version uint32, key, value string) []byte {
	buf := new(bytes.Buffer)
	// Header
	binary.Write(buf, binary.LittleEndian, uint32(FILE_MAGIC_GGUF_LE))
	binary.Write(buf, binary.LittleEndian, version)
	binary.Write(buf, binary.LittleEndian, uint64(0)) // numTensor
	binary.Write(buf, binary.LittleEndian, uint64(1)) // numKV

	// Key string (length + data)
	binary.Write(buf, binary.LittleEndian, uint64(len(key)))
	buf.WriteString(key)

	// Type (string = 8)
	binary.Write(buf, binary.LittleEndian, uint32(ggufTypeString))

	// Value string (length + data)
	binary.Write(buf, binary.LittleEndian, uint64(len(value)))
	buf.WriteString(value)

	return buf.Bytes()
}

func makeGGUFWithTensor(version uint32) []byte {
	buf := new(bytes.Buffer)
	// Header
	binary.Write(buf, binary.LittleEndian, uint32(FILE_MAGIC_GGUF_LE))
	binary.Write(buf, binary.LittleEndian, version)
	binary.Write(buf, binary.LittleEndian, uint64(1)) // numTensor
	binary.Write(buf, binary.LittleEndian, uint64(0)) // numKV

	// Tensor name
	tensorName := "test.weight"
	binary.Write(buf, binary.LittleEndian, uint64(len(tensorName)))
	buf.WriteString(tensorName)

	// Tensor dims
	binary.Write(buf, binary.LittleEndian, uint32(2)) // 2 dimensions
	binary.Write(buf, binary.LittleEndian, uint64(4)) // dim 0
	binary.Write(buf, binary.LittleEndian, uint64(4)) // dim 1

	// Tensor kind (F32 = 0)
	binary.Write(buf, binary.LittleEndian, uint32(0))

	// Tensor offset
	binary.Write(buf, binary.LittleEndian, uint64(0))

	return buf.Bytes()
}

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
