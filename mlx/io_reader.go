package mlx

// #include <stdbool.h>
// #include <stdint.h>
// #include <stddef.h>
// #include <stdlib.h>
import "C"

import (
	"io"
	"math"
	"runtime/cgo"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ioReader adapts Go random-access I/O to MLX's stateful reader vtable. File
// policy, parallelism, and accounting belong to the caller's ReaderAt.
type ioReader struct {
	reader SafetensorsReader
	label  *C.char

	offsetMu sync.Mutex // guards offset and serializes stateful seek/read calls
	offset   int64
	closed   atomic.Bool
}

func ioReaderFromHandle(desc unsafe.Pointer) *ioReader {
	if desc == nil {
		return nil
	}
	handle := *(*cgo.Handle)(desc)
	reader, _ := handle.Value().(*ioReader)
	return reader
}

func (r *ioReader) seek(off int64, whence int) bool {
	r.offsetMu.Lock()
	defer r.offsetMu.Unlock()

	var base int64
	switch whence {
	case io.SeekStart:
	case io.SeekCurrent:
		base = r.offset
	case io.SeekEnd:
		base = r.reader.Size()
	default:
		return false
	}
	if off > 0 && base > math.MaxInt64-off {
		return false
	}
	next := base + off
	if next < 0 {
		return false
	}
	r.offset = next
	return true
}

func (r *ioReader) read(data unsafe.Pointer, n uint64) int {
	r.offsetMu.Lock()
	defer r.offsetMu.Unlock()

	read := r.readAt(data, n, uint64(r.offset))
	r.offset += int64(read)
	return read
}

func (r *ioReader) readAt(data unsafe.Pointer, n, off uint64) int {
	if n == 0 {
		return 0
	}
	if r.closed.Load() || data == nil || n > uint64(math.MaxInt) || off > math.MaxInt64 || n > uint64(math.MaxInt64)-off {
		return 0
	}

	buf := unsafe.Slice((*byte)(data), int(n))
	read, _ := r.reader.ReadAt(buf, int64(off))
	return read
}

func (r *ioReader) close() {
	if r == nil || r.closed.Swap(true) {
		return
	}
	_ = r.reader.Close()
}

//export goMLXReaderIsOpen
func goMLXReaderIsOpen(desc unsafe.Pointer) C.bool {
	reader := ioReaderFromHandle(desc)
	return C.bool(reader != nil && !reader.closed.Load())
}

//export goMLXReaderTell
func goMLXReaderTell(desc unsafe.Pointer) C.size_t {
	reader := ioReaderFromHandle(desc)
	if reader == nil {
		return 0
	}
	reader.offsetMu.Lock()
	defer reader.offsetMu.Unlock()
	return C.size_t(reader.offset)
}

//export goMLXReaderSeek
func goMLXReaderSeek(desc unsafe.Pointer, off C.int64_t, whence C.int) C.int {
	reader := ioReaderFromHandle(desc)
	if reader == nil || !reader.seek(int64(off), int(whence)) {
		return -1
	}
	return 0
}

//export goMLXReaderRead
func goMLXReaderRead(desc unsafe.Pointer, data *C.char, n C.size_t) C.size_t {
	reader := ioReaderFromHandle(desc)
	if reader == nil {
		return 0
	}
	return C.size_t(reader.read(unsafe.Pointer(data), uint64(n)))
}

//export goMLXReaderReadAtOffset
func goMLXReaderReadAtOffset(desc unsafe.Pointer, data *C.char, n C.size_t, off C.size_t) C.size_t {
	reader := ioReaderFromHandle(desc)
	if reader == nil {
		return 0
	}
	return C.size_t(reader.readAt(unsafe.Pointer(data), uint64(n), uint64(off)))
}

//export goMLXReaderFree
func goMLXReaderFree(desc unsafe.Pointer) {
	if desc == nil {
		return
	}

	handle := *(*cgo.Handle)(desc)
	if reader, ok := handle.Value().(*ioReader); ok {
		reader.close()
		C.free(unsafe.Pointer(reader.label))
	}
	handle.Delete()
	C.free(desc)
}

//export goMLXReaderLabel
func goMLXReaderLabel(desc unsafe.Pointer) *C.char {
	reader := ioReaderFromHandle(desc)
	if reader == nil {
		return nil
	}
	return reader.label
}
