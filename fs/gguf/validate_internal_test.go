package gguf

import (
	"math"
	"testing"
)

// TestElementDiskSize pins the per-element on-disk size used to bound the
// unconditional make([]T, n) / make([]string, n) allocations in readArrayData
// and readArrayString (this parser has no maxArraySize cap at all).
func TestElementDiskSize(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  uint32
		want int
	}{
		{"uint8", typeUint8, 1},
		{"int8", typeInt8, 1},
		{"bool", typeBool, 1},
		{"uint16", typeUint16, 2},
		{"int16", typeInt16, 2},
		{"uint32", typeUint32, 4},
		{"int32", typeInt32, 4},
		{"float32", typeFloat32, 4},
		{"uint64", typeUint64, 8},
		{"int64", typeInt64, 8},
		{"float64", typeFloat64, 8},
		{"string", typeString, 8},
		{"unknown falls back to 1", uint32(0xdead), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := elementDiskSize(tc.typ); got != tc.want {
				t.Fatalf("elementDiskSize(%d) = %d, want %d", tc.typ, got, tc.want)
			}
		})
	}
}

// TestValidateArrayLength mirrors the fs/ggml unit test against the lazy
// parser's File.validateArrayLength.
func TestValidateArrayLength(t *testing.T) {
	const mb = 1 << 20
	for _, tc := range []struct {
		name     string
		fileSize int64
		n        uint64
		elemSize int
		wantErr  bool
	}{
		{"float64 1.6MB of data in a 1MB file", mb, 200_000, 8, true},
		{"uint64-max count (uint64->int sign flip)", mb, math.MaxUint64, 8, true},
		{"int32 4KB of data in a 100B file", 100, 1000, 4, true},

		{"string array fits in 1MB", mb, 100_000, 8, false},
		{"uint8 512 elements in 1KB", 1 << 10, 512, 1, false},
		{"empty float64 array", mb, 0, 8, false},
		{"empty array, unknown elem size", mb, 0, 0, false},

		{"exactly fileSize/elemSize", mb, mb / 8, 8, false},
		{"one past fileSize/elemSize", mb, mb/8 + 1, 8, true},

		{"unknown filesize allows large n", 0, 1 << 40, 8, false},
		{"unknown filesize still rejects sign flip", 0, math.MaxUint64, 8, true},

		{"elemSize 0 treated as 1, fits", 1 << 10, 1024, 0, false},
		{"elemSize 0 treated as 1, over", 1 << 10, 1025, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &File{fileSize: tc.fileSize}
			err := f.validateArrayLength(tc.n, tc.elemSize)
			if tc.wantErr && err == nil {
				t.Fatalf("validateArrayLength(n=%d, elem=%d, fileSize=%d): want error, got nil", tc.n, tc.elemSize, tc.fileSize)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("validateArrayLength(n=%d, elem=%d, fileSize=%d): want nil, got %v", tc.n, tc.elemSize, tc.fileSize, err)
			}
		})
	}
}
