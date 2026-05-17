package ggml

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"testing"
)

// TestGGUFElementDiskSize pins the per-element on-disk size used to bound
// array allocations. These values must equal the encoded width of each
// fixed-width type (so bounding the count also bounds make([]T, n)), and the
// 8-byte length prefix for strings.
func TestGGUFElementDiskSize(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  uint32
		want int
	}{
		{"uint8", ggufTypeUint8, 1},
		{"int8", ggufTypeInt8, 1},
		{"bool", ggufTypeBool, 1},
		{"uint16", ggufTypeUint16, 2},
		{"int16", ggufTypeInt16, 2},
		{"uint32", ggufTypeUint32, 4},
		{"int32", ggufTypeInt32, 4},
		{"float32", ggufTypeFloat32, 4},
		{"uint64", ggufTypeUint64, 8},
		{"int64", ggufTypeInt64, 8},
		{"float64", ggufTypeFloat64, 8},
		{"string", ggufTypeString, 8},
		{"unknown falls back to 1", uint32(0xdead), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := ggufElementDiskSize(tc.typ); got != tc.want {
				t.Fatalf("ggufElementDiskSize(%d) = %d, want %d", tc.typ, got, tc.want)
			}
		})
	}
}

// TestValidateArrayLength exercises the type-aware bound directly: an array of
// n elements each at least elemSize bytes must physically fit in the file.
func TestValidateArrayLength(t *testing.T) {
	const mb = 1 << 20
	for _, tc := range []struct {
		name     string
		fileSize int64
		n        uint64
		elemSize int
		wantErr  bool
	}{
		// reject: declared element data exceeds the file
		{"float64 1.6MB of data in a 1MB file", mb, 200_000, 8, true},
		{"uint64-max count (uint64->int sign flip)", mb, math.MaxUint64, 8, true},
		{"int32 4KB of data in a 100B file", 100, 1000, 4, true},

		// pass: data physically fits
		{"string array fits in 1MB", mb, 100_000, 8, false},
		{"uint8 512 elements in 1KB", 1 << 10, 512, 1, false},
		{"empty float64 array", mb, 0, 8, false},
		{"empty array, unknown elem size", mb, 0, 0, false},

		// boundary: the bound is inclusive (n*elemSize == fileSize is allowed)
		{"exactly fileSize/elemSize", mb, mb / 8, 8, false},
		{"one past fileSize/elemSize", mb, mb/8 + 1, 8, true},

		// unknown file size: only the sign-flip guard applies
		{"unknown filesize allows large n", 0, 1 << 40, 8, false},
		{"unknown filesize still rejects sign flip", 0, math.MaxUint64, 8, true},

		// elemSize < 1 is defensively treated as 1
		{"elemSize 0 treated as 1, fits", 1 << 10, 1024, 0, false},
		{"elemSize 0 treated as 1, over", 1 << 10, 1025, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &containerGGUF{fileSize: tc.fileSize}
			err := c.validateArrayLength(tc.n, tc.elemSize)
			if tc.wantErr && err == nil {
				t.Fatalf("validateArrayLength(n=%d, elem=%d, fileSize=%d): want error, got nil", tc.n, tc.elemSize, tc.fileSize)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("validateArrayLength(n=%d, elem=%d, fileSize=%d): want nil, got %v", tc.n, tc.elemSize, tc.fileSize, err)
			}
		})
	}
}

// TestDecodeArrayAmplificationRejected is the regression case for the reviewer
// feedback: a typed-array element count that is <= fileSize (so the old
// raw-byte validateLength check would have let it through) but whose backing
// allocation n*sizeof(T) exceeds the file. It must now be rejected end-to-end.
func TestDecodeArrayAmplificationRejected(t *testing.T) {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, uint32(FILE_MAGIC_GGUF_LE))
	binary.Write(&b, binary.LittleEndian, uint32(3))  // version
	binary.Write(&b, binary.LittleEndian, uint64(0))  // numTensor
	binary.Write(&b, binary.LittleEndian, uint64(1))  // numKV
	binary.Write(&b, binary.LittleEndian, uint64(1))  // key length
	b.WriteByte('x')                                  // key "x"
	binary.Write(&b, binary.LittleEndian, uint32(9))  // value type = array
	binary.Write(&b, binary.LittleEndian, uint32(12)) // element type = float64
	binary.Write(&b, binary.LittleEndian, uint64(10)) // count

	fileSize := b.Len() // 49 bytes
	// count 10 <= 49 bytes, so the pre-fix `n > fileSize` check passed it,
	// yet make([]float64, 10) reads/needs 80 bytes — more than the whole file.
	if 10 > fileSize || 10*8 <= fileSize {
		t.Fatalf("test premise broken: fileSize=%d", fileSize)
	}
	if _, err := Decode(bytes.NewReader(b.Bytes()), -1); err == nil {
		t.Fatalf("Decode must reject a 10-element float64 array (80 bytes) in a %d-byte file", fileSize)
	}
}

// fakeSeeker reports a large virtual file size via Seek while only serving a
// small header and then io.EOF. It lets us exercise the "count is within
// fileSize/elemSize so it passes validation" path without writing a 100MB
// file to disk.
type fakeSeeker struct {
	data []byte
	size int64
	pos  int64
}

func (f *fakeSeeker) Read(p []byte) (int, error) {
	if f.pos >= int64(len(f.data)) {
		return 0, io.EOF
	}
	n := copy(p, f.data[f.pos:])
	f.pos += int64(n)
	return n, nil
}

func (f *fakeSeeker) Seek(off int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case io.SeekStart:
		abs = off
	case io.SeekCurrent:
		abs = f.pos + off
	case io.SeekEnd:
		abs = f.size + off
	default:
		return 0, fmt.Errorf("fakeSeeker: bad whence %d", whence)
	}
	if abs < 0 {
		return 0, fmt.Errorf("fakeSeeker: negative position %d", abs)
	}
	f.pos = abs
	return abs, nil
}

// TestDecodeLargeStringArrayBoundedAlloc is the empirical backing for the
// elemSize=8 argument for string arrays. With a 100MiB file a count of 12.5M
// strings passes validateArrayLength (12.5M <= 100MiB/8), so the decoder does
// allocate make([]string, 12.5M) ~= 200MiB of headers — bounded at ~2x the
// file, NOT the pre-fix multi-GB — and then fails cleanly on the first string
// body read (EOF) instead of panicking or OOM-killing the process.
func TestDecodeLargeStringArrayBoundedAlloc(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates ~200MiB to demonstrate the bounded string-array path")
	}

	const fileSize = 100 << 20 // 100 MiB virtual file
	const n = 12_500_000       // <= fileSize/8 (13,107,200): passes validation

	var hdr bytes.Buffer
	binary.Write(&hdr, binary.LittleEndian, uint32(FILE_MAGIC_GGUF_LE))
	binary.Write(&hdr, binary.LittleEndian, uint32(3)) // version
	binary.Write(&hdr, binary.LittleEndian, uint64(0)) // numTensor
	binary.Write(&hdr, binary.LittleEndian, uint64(1)) // numKV
	binary.Write(&hdr, binary.LittleEndian, uint64(1)) // key length
	hdr.WriteByte('x')                                 // key "x"
	binary.Write(&hdr, binary.LittleEndian, uint32(9)) // value type = array
	binary.Write(&hdr, binary.LittleEndian, uint32(8)) // element type = string
	binary.Write(&hdr, binary.LittleEndian, uint64(n)) // array count

	rs := &fakeSeeker{data: hdr.Bytes(), size: fileSize}
	if _, err := Decode(rs, -1); err == nil {
		t.Fatal("Decode should error: the file declares a 12.5M-string array it cannot contain")
	}
}
