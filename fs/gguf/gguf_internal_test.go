package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

func TestReadStringRejectsOversizedLength(t *testing.T) {
	var b bytes.Buffer
	writeInternalRaw(t, &b, uint64(MaxStringLength+1))

	_, err := readString(testFile(b.Bytes()))
	if err == nil {
		t.Fatal("readString unexpectedly succeeded")
	}
	if !strings.Contains(err.Error(), "string") {
		t.Fatalf("readString error = %q, want string length error", err)
	}
}

func TestReadArrayRejectsOversizedCollectedArray(t *testing.T) {
	var b bytes.Buffer
	writeInternalRaw(t, &b, typeString)
	writeInternalRaw(t, &b, uint64(MaxArraySize+1))

	_, err := readArray(testFile(b.Bytes()))
	if err == nil {
		t.Fatal("readArray unexpectedly succeeded")
	}
	if !strings.Contains(err.Error(), "array size") {
		t.Fatalf("readArray error = %q, want array size error", err)
	}
}

func TestScanKeyValuesRejectsMalformedMetadata(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want string
	}{
		{
			name: "oversized key string",
			data: internalGGUFTestFile(func(b *bytes.Buffer) {
				writeInternalGGUFHeader(t, b, 1)
				writeInternalRaw(t, b, uint64(MaxStringLength+1))
			}),
			want: "string",
		},
		{
			name: "oversized skipped string",
			data: internalGGUFTestFile(func(b *bytes.Buffer) {
				writeInternalGGUFHeader(t, b, 1)
				writeInternalString(t, b, "unused")
				writeInternalRaw(t, b, typeString)
				writeInternalRaw(t, b, uint64(MaxStringLength+1))
			}),
			want: "string",
		},
		{
			name: "oversized skipped array",
			data: internalGGUFTestFile(func(b *bytes.Buffer) {
				writeInternalGGUFHeader(t, b, 1)
				writeInternalString(t, b, "unused")
				writeInternalRaw(t, b, typeArray)
				writeInternalRaw(t, b, typeUint8)
				writeInternalRaw(t, b, uint64(MaxArraySize+1))
			}),
			want: "array size",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ScanKeyValues panicked: %v", r)
				}
			}()

			_, err := ScanKeyValues(writeTempFile(t, tt.data), func(string) bool {
				return false
			})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestTensorInfoRejectsRowNotMultipleOfBlockSize(t *testing.T) {
	ti := TensorInfo{
		Name:  "bad.weight",
		Shape: []uint64{31},
		Type:  TensorTypeQ4_0,
	}

	if _, ok := ti.numBytes(); ok {
		t.Fatal("numBytes unexpectedly succeeded")
	}
}

func TestTensorReaderReturnsLazyParseError(t *testing.T) {
	var b bytes.Buffer
	writeInternalRaw(t, &b, []byte("GGUF"))
	writeInternalRaw(t, &b, uint32(3))
	writeInternalRaw(t, &b, uint64(1)) // tensors
	writeInternalRaw(t, &b, uint64(0)) // key-values
	writeInternalString(t, &b, "bad.weight")
	writeInternalRaw(t, &b, uint32(MaxTensorDims+1))

	p := writeTempFile(t, b.Bytes())
	f, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	_, _, err = f.TensorReader("bad.weight")
	if err == nil {
		t.Fatal("TensorReader unexpectedly succeeded")
	}
	if !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("TensorReader error = %q, want dimensions error", err)
	}
}

func TestTensorReaderRejectsInvalidOffset(t *testing.T) {
	var b bytes.Buffer
	writeInternalRaw(t, &b, []byte("GGUF"))
	writeInternalRaw(t, &b, uint32(3))
	writeInternalRaw(t, &b, uint64(1)) // tensors
	writeInternalRaw(t, &b, uint64(0)) // key-values
	writeInternalString(t, &b, "bad.weight")
	writeInternalRaw(t, &b, uint32(1)) // dimensions
	writeInternalRaw(t, &b, uint64(1))
	writeInternalRaw(t, &b, uint32(TensorTypeF32))
	writeInternalRaw(t, &b, ^uint64(0))

	p := writeTempFile(t, b.Bytes())
	f, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	_, _, err = f.TensorReader("bad.weight")
	if err == nil {
		t.Fatal("TensorReader unexpectedly succeeded")
	}
	if !strings.Contains(err.Error(), "offset") {
		t.Fatalf("TensorReader error = %q, want offset error", err)
	}
}

func TestTensorReaderRejectsMissingTensor(t *testing.T) {
	var b bytes.Buffer
	writeInternalRaw(t, &b, []byte("GGUF"))
	writeInternalRaw(t, &b, uint32(3))
	writeInternalRaw(t, &b, uint64(0)) // tensors
	writeInternalRaw(t, &b, uint64(0)) // key-values

	p := writeTempFile(t, b.Bytes())
	f, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	_, _, err = f.TensorReader("missing.weight")
	if err == nil {
		t.Fatal("TensorReader unexpectedly succeeded")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Fatalf("TensorReader error = %q, want not found error", err)
	}
}

func testFile(data []byte) *File {
	return &File{
		reader: newBufferedReader(bytes.NewReader(data), 32<<10),
		bts:    make([]byte, 4096),
	}
}

func writeInternalRaw(t *testing.T, b *bytes.Buffer, v any) {
	t.Helper()
	if err := binary.Write(b, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
}

func writeInternalString(t *testing.T, b *bytes.Buffer, s string) {
	t.Helper()
	writeInternalRaw(t, b, uint64(len(s)))
	writeInternalRaw(t, b, []byte(s))
}

func internalGGUFTestFile(fn func(*bytes.Buffer)) []byte {
	var b bytes.Buffer
	fn(&b)
	return b.Bytes()
}

func writeInternalGGUFHeader(t *testing.T, b *bytes.Buffer, numKV uint64) {
	t.Helper()
	writeInternalRaw(t, b, []byte("GGUF"))
	writeInternalRaw(t, b, uint32(3))
	writeInternalRaw(t, b, uint64(0))
	writeInternalRaw(t, b, numKV)
}

func writeTempFile(t *testing.T, data []byte) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}
	return f.Name()
}
