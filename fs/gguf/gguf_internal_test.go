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

func TestReadFileMetadataRejectsMalformedValues(t *testing.T) {
	tests := []struct {
		name string
		data func(*testing.T) []byte
		want string
	}{
		{
			name: "version 1 zero-length string",
			data: func(t *testing.T) []byte {
				var b bytes.Buffer
				writeInternalRaw(t, &b, []byte("GGUF"))
				writeInternalRaw(t, &b, uint32(1))
				writeInternalRaw(t, &b, uint32(0))
				writeInternalRaw(t, &b, uint32(1))
				writeInternalRaw(t, &b, uint64(0))
				return b.Bytes()
			},
			want: "version 1 string has zero length",
		},
		{
			name: "zero alignment",
			data: func(t *testing.T) []byte {
				var b bytes.Buffer
				writeInternalRaw(t, &b, []byte("GGUF"))
				writeInternalRaw(t, &b, uint32(3))
				writeInternalRaw(t, &b, uint64(0))
				writeInternalRaw(t, &b, uint64(1))
				writeInternalString(t, &b, "general.alignment")
				writeInternalRaw(t, &b, typeUint32)
				writeInternalRaw(t, &b, uint32(0))
				return b.Bytes()
			},
			want: "alignment 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ReadFileMetadata(writeTempFile(t, tt.data(t)), 0)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ReadFileMetadata() error = %v, want containing %q", err, tt.want)
			}
		})
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

func TestReadArrayRejectsSkippedVersion1StringWithoutTerminator(t *testing.T) {
	var b bytes.Buffer
	writeInternalRaw(t, &b, uint64(4))
	b.WriteString("nope")

	f := testFile(b.Bytes())
	f.Version = 1
	f.maxArraySize = 0
	_, err := readArrayString(f, 1)
	if err == nil || !strings.Contains(err.Error(), "not null terminated") {
		t.Fatalf("readArrayString error = %v, want null-termination error", err)
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
	if got := ti.NumBytes(); got != -1 {
		t.Fatalf("NumBytes() = %d, want -1", got)
	}
}

func TestTensorInfoRejectsElementCountOverflow(t *testing.T) {
	ti := TensorInfo{
		Name:  "bad.weight",
		Shape: []uint64{maxInt64(), 2},
		Type:  TensorTypeF32,
	}

	if got := ti.NumValues(); got != -1 {
		t.Fatalf("NumValues() = %d, want -1", got)
	}
	if got := ti.NumBytes(); got != -1 {
		t.Fatalf("NumBytes() = %d, want -1", got)
	}
}

func TestCurrentTensorTypes(t *testing.T) {
	for _, tt := range []struct {
		value     TensorType
		number    uint32
		name      string
		blockSize int64
		typeSize  int64
	}{
		{value: TensorTypeMXFP4, number: 39, name: "mxfp4", blockSize: 32, typeSize: 17},
		{value: TensorTypeNVFP4, number: 40, name: "nvfp4", blockSize: 64, typeSize: 36},
		{value: TensorTypeQ1_0, number: 41, name: "q1_0", blockSize: 128, typeSize: 18},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := uint32(tt.value); got != tt.number {
				t.Fatalf("value = %d, want %d", got, tt.number)
			}
			if got := tt.value.String(); got != tt.name {
				t.Fatalf("String() = %q, want %q", got, tt.name)
			}
			if got := tt.value.blockSize(); got != tt.blockSize {
				t.Fatalf("blockSize() = %d, want %d", got, tt.blockSize)
			}
			if got := tt.value.typeSize(); got != tt.typeSize {
				t.Fatalf("typeSize() = %d, want %d", got, tt.typeSize)
			}
		})
	}
}

func TestReadFileMetadataLegacyEncoding(t *testing.T) {
	for _, tt := range []struct {
		name      string
		version   uint32
		byteOrder binary.ByteOrder
	}{
		{name: "version_1", version: 1, byteOrder: binary.LittleEndian},
		{name: "big_endian", version: 3, byteOrder: binary.BigEndian},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := writeEncodedGGUF(t, tt.version, tt.byteOrder)
			metadata, err := ReadFileMetadata(path, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got := metadata.Architecture(); got != "llama" {
				t.Fatalf("architecture = %q, want llama", got)
			}
			if got := metadata.ParameterCount(); got != 1 {
				t.Fatalf("parameter count = %d, want 1", got)
			}
		})
	}
}

func TestReadFileMetadataDoesNotPreallocateDeclaredKeyValues(t *testing.T) {
	var b bytes.Buffer
	b.WriteString("GGUF")
	writeInternalRaw(t, &b, uint32(3))
	writeInternalRaw(t, &b, uint64(0))
	writeInternalRaw(t, &b, uint64(maxInt()))

	if _, err := ReadFileMetadata(writeTempFile(t, b.Bytes()), 0); err == nil {
		t.Fatal("ReadFileMetadata unexpectedly succeeded")
	}
}

func writeEncodedGGUF(t *testing.T, version uint32, byteOrder binary.ByteOrder) string {
	t.Helper()

	var b bytes.Buffer
	if byteOrder == binary.BigEndian {
		b.WriteString("FUGG")
	} else {
		b.WriteString("GGUF")
	}
	writeEncodedValue(t, &b, byteOrder, version)
	if version == 1 {
		writeEncodedValue(t, &b, byteOrder, uint32(1))
		writeEncodedValue(t, &b, byteOrder, uint32(1))
	} else {
		writeEncodedValue(t, &b, byteOrder, uint64(1))
		writeEncodedValue(t, &b, byteOrder, uint64(1))
	}
	writeEncodedString(t, &b, byteOrder, version, "general.architecture")
	writeEncodedValue(t, &b, byteOrder, typeString)
	writeEncodedString(t, &b, byteOrder, version, "llama")
	writeEncodedString(t, &b, byteOrder, version, "weight")
	writeEncodedValue(t, &b, byteOrder, uint32(1))
	writeEncodedValue(t, &b, byteOrder, uint64(1))
	writeEncodedValue(t, &b, byteOrder, uint32(TensorTypeF32))
	writeEncodedValue(t, &b, byteOrder, uint64(0))
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	writeEncodedValue(t, &b, byteOrder, float32(0))
	return writeTempFile(t, b.Bytes())
}

func writeEncodedString(t *testing.T, b *bytes.Buffer, byteOrder binary.ByteOrder, version uint32, value string) {
	t.Helper()
	length := len(value)
	if version == 1 {
		length++
	}
	writeEncodedValue(t, b, byteOrder, uint64(length))
	b.WriteString(value)
	if version == 1 {
		b.WriteByte(0)
	}
}

func writeEncodedValue(t *testing.T, b *bytes.Buffer, byteOrder binary.ByteOrder, value any) {
	t.Helper()
	if err := binary.Write(b, byteOrder, value); err != nil {
		t.Fatal(err)
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
		reader:    newBufferedReader(bytes.NewReader(data), 32<<10),
		bts:       make([]byte, 4096),
		byteOrder: binary.LittleEndian,
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
