package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

const (
	testGGUFTypeUint16 uint32 = 2
	testGGUFTypeUint32 uint32 = 4
)

func TestGGUFEndianAndAlignment(t *testing.T) {
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		t.Run(order.String(), func(t *testing.T) {
			byteOrder := "little"
			if order == binary.BigEndian {
				byteOrder = "big"
			}
			var b bytes.Buffer
			write := func(v any) {
				t.Helper()
				if err := binary.Write(&b, order, v); err != nil {
					t.Fatal(err)
				}
			}
			writeString := func(s string) { write(uint64(len(s))); b.WriteString(s) }
			if order == binary.LittleEndian {
				b.WriteString("GGUF")
			} else {
				b.WriteString("FUGG")
			}
			write(uint32(3))
			write(uint64(1))
			write(uint64(1))
			writeString("general.alignment")
			write(testGGUFTypeUint32)
			write(uint32(128))
			writeString("weight")
			write(uint32(1))
			write(uint64(2))
			write(uint32(ggml.TensorTypeF32))
			write(uint64(0))
			b.Write(make([]byte, (128-b.Len()%128)%128))
			dataOffset := b.Len()
			b.Write([]byte{0, 0, 128, 63, 0, 0, 0, 64})

			path := t.TempDir() + "/model.gguf"
			if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
			inv, err := inspect(t.Context(), path, "")
			if err != nil {
				t.Fatal(err)
			}
			tensor := inv.tensors[tensorKey("model", "weight")]
			if tensor == nil || tensor.Offset != int64(dataOffset) || tensor.Bytes != 8 {
				t.Fatalf("wrong GGUF tensor inventory: %+v", tensor)
			}
			if tensor.ByteOrder != byteOrder {
				t.Fatalf("ByteOrder = %q, want %q", tensor.ByteOrder, byteOrder)
			}
		})
	}
}

func TestRejectPartialGGUF(t *testing.T) {
	// ggml.WriteGGUF treats unrecognized split.* keys as architecture-relative,
	// so construct this valid split header directly rather than testing an
	// invalid llama.split.count artifact produced by that writer.
	var b bytes.Buffer
	write := func(v any) {
		t.Helper()
		if err := binary.Write(&b, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(s string) { write(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	write(uint32(3))
	write(uint64(1))
	write(uint64(1))
	writeString("split.count")
	write(testGGUFTypeUint16)
	write(uint16(2))
	writeString("weight")
	write(uint32(1))
	write(uint64(1))
	write(uint32(0))
	write(uint64(0))
	b.Write(make([]byte, (32-b.Len()%32)%32))
	b.Write([]byte{0, 0, 128, 63})
	path := t.TempDir() + "/split.gguf"
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Compare(t.Context(), path, path, Options{}); err == nil || !strings.Contains(err.Error(), "split GGUF") {
		t.Fatalf("partial model accepted: %v", err)
	}
}

func TestGGUFPayloadAndMetadata(t *testing.T) {
	var paths []string
	for i := range 2 {
		f, err := os.CreateTemp(t.TempDir(), "*.gguf")
		if err != nil {
			t.Fatal(err)
		}
		kv := ggml.KV{"general.architecture": "llama", "general.description": []string{"left", "right"}[i], "general.parameter_count": uint64(99 + i), "tokenizer.ggml.tokens": []string{"a", " b"}}
		tensors := []*ggml.Tensor{{Name: "weight", Kind: uint32(ggml.TensorTypeQ4_K), Shape: []uint64{256, 1}, WriterTo: bytes.NewReader(append(make([]byte, 143), byte(i)))}, {Name: "scalar", Kind: uint32(ggml.TensorTypeF32), Shape: []uint64{1}, WriterTo: bytes.NewReader([]byte{0, 0, 128, 63})}}
		if err := ggml.WriteGGUF(f, kv, tensors); err != nil {
			t.Fatal(err)
		}
		f.Close()
		paths = append(paths, f.Name())
	}
	r, err := Compare(t.Context(), paths[0], paths[1], Options{})
	if err != nil {
		t.Fatal(err)
	}
	wantBytesHashed := int64(2 * (ggml.TensorTypeQ4_K.TypeSize() + ggml.TensorTypeF32.TypeSize()))
	if r.Equal || r.Summary.Changed != 1 || r.Summary.Equal != 1 || r.Summary.BytesHashed != wantBytesHashed {
		t.Fatalf("wrong GGUF comparison: %+v", r)
	}
	if len(r.Metadata) != 1 || r.Metadata[0].Path != "/model.gguf/general.description" {
		t.Fatalf("GGUF metadata comparison is wrong: %+v", r.Metadata)
	}
}
