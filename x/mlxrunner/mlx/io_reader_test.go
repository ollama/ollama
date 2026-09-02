//go:build cgo

package mlx

import (
	"bytes"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

func TestFileIOReaderReadAtReportsProgressAndData(t *testing.T) {
	path := writeReaderTestFile(t, []byte("abcdefghijklmnopqrstuvwxyz"))

	var completed atomic.Int64
	reader, err := newFileIOReader(path, func(n int64) { completed.Add(n) })
	if err != nil {
		t.Fatalf("newFileIOReader error: %v", err)
	}
	defer reader.close()

	buf := make([]byte, 5)
	read := reader.readAt(unsafe.Pointer(&buf[0]), uint64(len(buf)), 7)
	if read != len(buf) {
		t.Fatalf("readAt read %d bytes, want %d", read, len(buf))
	}
	if string(buf) != "hijkl" {
		t.Fatalf("readAt data = %q, want %q", string(buf), "hijkl")
	}
	if got := completed.Load(); got != int64(len(buf)) {
		t.Fatalf("progress bytes = %d, want %d", got, len(buf))
	}
	if got := reader.tell(); got != 0 {
		t.Fatalf("tell after readAt = %d, want 0", got)
	}
}

func TestFileIOReaderSequentialReadSeekEnd(t *testing.T) {
	path := writeReaderTestFile(t, []byte("abcdefghijklmnopqrstuvwxyz"))

	reader, err := newFileIOReader(path, nil)
	if err != nil {
		t.Fatalf("newFileIOReader error: %v", err)
	}
	defer reader.close()

	reader.seek(-3, io.SeekEnd)
	buf := make([]byte, 3)
	reader.read(unsafe.Pointer(&buf[0]), uint64(len(buf)))

	if string(buf) != "xyz" {
		t.Fatalf("read data = %q, want %q", string(buf), "xyz")
	}
	if got := reader.tell(); got != 26 {
		t.Fatalf("tell = %d, want 26", got)
	}
}

func TestFileIOReaderConcurrentReadAt(t *testing.T) {
	data := bytes.Repeat([]byte("0123456789abcdef"), 1024)
	path := writeReaderTestFile(t, data)

	var completed atomic.Int64
	reader, err := newFileIOReader(path, func(n int64) { completed.Add(n) })
	if err != nil {
		t.Fatalf("newFileIOReader error: %v", err)
	}
	defer reader.close()

	const goroutines = 16
	const iterations = 128
	const chunk = 64

	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := range iterations {
				offset := (g*iterations + i) * chunk % (len(data) - chunk)
				buf := make([]byte, chunk)
				read := reader.readAt(unsafe.Pointer(&buf[0]), uint64(len(buf)), uint64(offset))
				if read != len(buf) {
					t.Errorf("readAt read %d bytes, want %d", read, len(buf))
					return
				}
				if !bytes.Equal(buf, data[offset:offset+chunk]) {
					t.Errorf("readAt mismatch at offset %d", offset)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	if err := reader.Err(); err != nil {
		t.Fatalf("reader error: %v", err)
	}
	if got, want := completed.Load(), int64(goroutines*iterations*chunk); got != want {
		t.Fatalf("progress bytes = %d, want %d", got, want)
	}
}

func TestFileIOReaderShortReadRecordsError(t *testing.T) {
	path := writeReaderTestFile(t, []byte("abc"))

	reader, err := newFileIOReader(path, nil)
	if err != nil {
		t.Fatalf("newFileIOReader error: %v", err)
	}
	defer reader.close()

	buf := make([]byte, 4)
	read := reader.readAt(unsafe.Pointer(&buf[0]), uint64(len(buf)), 0)
	if read != 3 {
		t.Fatalf("readAt read %d bytes, want 3", read)
	}
	if err := reader.Err(); err == nil {
		t.Fatal("expected short read error")
	}
	if reader.good() {
		t.Fatal("reader is good after short read")
	}
}

func TestLoadSafetensorsWithProgressUsesReaderCallbacks(t *testing.T) {
	withMLXThread(t, func() {
		payload := []byte{3, 1, 4, 1}
		path := writeSafetensorsTestFile(t, payload)

		var completed atomic.Int64
		sf, err := LoadSafetensorsWithProgress(path, func(n int64) { completed.Add(n) })
		if err != nil {
			t.Fatalf("LoadSafetensorsWithProgress error: %v", err)
		}
		defer sf.Free()

		arr := sf.Get("tensor")
		if arr == nil {
			t.Fatal("tensor not found")
		}
		if got := arr.DType(); got != DTypeUint8 {
			t.Fatalf("dtype = %v, want %v", got, DTypeUint8)
		}
		if got, want := arr.Dims(), []int{len(payload)}; !slices.Equal(got, want) {
			t.Fatalf("dims = %v, want %v", got, want)
		}

		arr32 := arr.AsType(DTypeInt32)
		Eval(arr32)
		if got, want := arr32.Ints(), []int32{3, 1, 4, 1}; !slices.Equal(got, want) {
			t.Fatalf("values = %v, want %v", got, want)
		}
		if got := completed.Load(); got < int64(len(payload)) {
			t.Fatalf("progress bytes = %d, want at least tensor payload %d", got, len(payload))
		}
	})
}

func writeReaderTestFile(t *testing.T, data []byte) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "test.safetensors")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write test file: %v", err)
	}
	return path
}

func writeSafetensorsTestFile(t *testing.T, data []byte) string {
	t.Helper()

	header := []byte(`{"tensor":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}`)
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(header))); err != nil {
		t.Fatalf("write header length: %v", err)
	}
	buf.Write(header)
	buf.Write(data)

	return writeReaderTestFile(t, buf.Bytes())
}
