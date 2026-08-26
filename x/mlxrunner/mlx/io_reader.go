package mlx

// #include <stdbool.h>
// #include <stdint.h>
// #include <stddef.h>
// #include <stdlib.h>
// #include "generated.h"
//
// extern bool goMLXReaderIsOpen(void*);
// extern bool goMLXReaderGood(void*);
// extern size_t goMLXReaderTell(void*);
// extern void goMLXReaderSeek(void*, int64_t, int);
// extern void goMLXReaderRead(void*, char*, size_t);
// extern void goMLXReaderReadAtOffset(void*, char*, size_t, size_t);
// extern void goMLXReaderWrite(void*, const char*, size_t);
// extern const char* goMLXReaderLabel(void*);
// extern void goMLXReaderFree(void*);
//
// static mlx_io_vtable go_mlx_reader_vtable(void) {
// 	mlx_io_vtable vtable = {0};
// 	vtable.is_open = goMLXReaderIsOpen;
// 	vtable.good = goMLXReaderGood;
// 	vtable.tell = goMLXReaderTell;
// 	vtable.seek = goMLXReaderSeek;
// 	vtable.read = goMLXReaderRead;
// 	vtable.read_at_offset = goMLXReaderReadAtOffset;
// 	vtable.write = (void (*)(void*, const char*, size_t))goMLXReaderWrite;
// 	vtable.label = goMLXReaderLabel;
// 	vtable.free = goMLXReaderFree;
// 	return vtable;
// }
//
// static mlx_io_reader go_mlx_reader_new(void* desc) {
// 	return mlx_io_reader_new(desc, go_mlx_reader_vtable());
// }
import "C"

import (
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/cgo"
	"sync"
	"sync/atomic"
	"unsafe"
)

const maxInt = int(^uint(0) >> 1)
const readerBatchSize = 1 << 25

// Match MLX's native ParallelFileReader shape for large tensor materialization:
// one disk-wide bounded pool, not one unbounded pool per safetensors blob.
var readerBatchTokens = make(chan struct{}, readerBatchParallelism())

func readerBatchParallelism() int {
	return min(max(runtime.NumCPU()/2, 4), 16)
}

type fileIOReader struct {
	path     string
	size     int64
	label    unsafe.Pointer
	progress func(int64)

	filesMu sync.Mutex
	idle    []*os.File

	offsetMu sync.Mutex
	offset   int64

	errMu sync.Mutex
	err   error

	closed atomic.Bool
}

func newFileIOReader(path string, progress func(int64)) (*fileIOReader, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}

	info, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("stat safetensors: %w", err)
	}

	return &fileIOReader{
		path:     path,
		size:     info.Size(),
		label:    unsafe.Pointer(C.CString(path)),
		progress: progress,
		idle:     []*os.File{file},
	}, nil
}

func (r *fileIOReader) newCReader() C.mlx_io_reader {
	payload := (*cgo.Handle)(C.malloc(C.size_t(unsafe.Sizeof(cgo.Handle(0)))))
	if payload == nil {
		panic("mlx: failed to allocate IO reader handle")
	}
	*payload = cgo.NewHandle(r)
	return C.go_mlx_reader_new(unsafe.Pointer(payload))
}

func (r *fileIOReader) isOpen() bool {
	return r != nil && !r.closed.Load()
}

func (r *fileIOReader) good() bool {
	return r.isOpen() && r.Err() == nil
}

func (r *fileIOReader) tell() uint64 {
	r.offsetMu.Lock()
	defer r.offsetMu.Unlock()
	if r.offset < 0 {
		return 0
	}
	return uint64(r.offset)
}

func (r *fileIOReader) seek(off int64, whence int) {
	r.offsetMu.Lock()
	defer r.offsetMu.Unlock()

	var next int64
	switch whence {
	case io.SeekStart:
		next = off
	case io.SeekCurrent:
		next = r.offset + off
	case io.SeekEnd:
		next = r.size + off
	default:
		r.setErr(fmt.Errorf("invalid seek whence %d", whence))
		return
	}
	if next < 0 {
		r.setErr(fmt.Errorf("invalid negative seek offset %d", next))
		return
	}
	r.offset = next
}

func (r *fileIOReader) read(data unsafe.Pointer, n uint64) {
	r.offsetMu.Lock()
	defer r.offsetMu.Unlock()

	read := r.readAt(data, n, uint64(r.offset))
	r.offset += int64(read)
}

func (r *fileIOReader) readAt(data unsafe.Pointer, n, off uint64) int {
	if n == 0 {
		return 0
	}
	if !r.isOpen() {
		r.setErr(errors.New("read from closed safetensors reader"))
		return 0
	}
	if data == nil {
		r.setErr(errors.New("read into nil safetensors buffer"))
		return 0
	}
	if n > uint64(maxInt) {
		r.setErr(fmt.Errorf("safetensors read too large: %d", n))
		return 0
	}
	if off > uint64(maxInt)-n {
		r.setErr(fmt.Errorf("safetensors read offset too large: %d", off))
		return 0
	}
	if n >= readerBatchSize {
		return r.readAtBatched(data, n, off)
	}

	buf := unsafe.Slice((*byte)(data), int(n))
	file, err := r.acquireFile()
	if err != nil {
		r.setErr(err)
		return 0
	}
	defer r.releaseFile(file)

	read, err := file.ReadAt(buf, int64(off))
	if read > 0 && r.progress != nil {
		r.progress(int64(read))
	}
	if err != nil && !(errors.Is(err, io.EOF) && read == len(buf)) {
		r.setErr(fmt.Errorf("%s: %w", r.path, err))
	}
	if read != len(buf) {
		r.setErr(fmt.Errorf("%s: %w", r.path, io.ErrUnexpectedEOF))
	}
	return read
}

func (r *fileIOReader) readAtBatched(data unsafe.Pointer, n, off uint64) int {
	type result struct {
		read int
		err  error
	}

	chunks := int((n + readerBatchSize - 1) / readerBatchSize)
	results := make(chan result, chunks)
	for offset := uint64(0); offset < n; offset += readerBatchSize {
		size := min(uint64(readerBatchSize), n-offset)
		go func(offset, size uint64) {
			readerBatchTokens <- struct{}{}
			defer func() { <-readerBatchTokens }()

			file, err := r.acquireFile()
			if err != nil {
				results <- result{err: err}
				return
			}
			defer r.releaseFile(file)

			buf := unsafe.Slice((*byte)(unsafe.Add(data, offset)), int(size))
			read, err := file.ReadAt(buf, int64(off+offset))
			if err != nil && !(errors.Is(err, io.EOF) && read == len(buf)) {
				err = fmt.Errorf("%s: %w", r.path, err)
			} else if read != len(buf) {
				err = fmt.Errorf("%s: %w", r.path, io.ErrUnexpectedEOF)
			}
			results <- result{read: read, err: err}
		}(offset, size)
	}

	total := 0
	for range chunks {
		res := <-results
		total += res.read
		if res.read > 0 && r.progress != nil {
			r.progress(int64(res.read))
		}
		if res.err != nil {
			r.setErr(res.err)
		}
	}
	return total
}

func (r *fileIOReader) acquireFile() (*os.File, error) {
	r.filesMu.Lock()
	defer r.filesMu.Unlock()
	if r.closed.Load() {
		return nil, errors.New("read from closed safetensors reader")
	}
	last := len(r.idle) - 1
	if last >= 0 {
		file := r.idle[last]
		r.idle[last] = nil
		r.idle = r.idle[:last]
		return file, nil
	}
	file, err := os.Open(r.path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	return file, nil
}

func (r *fileIOReader) releaseFile(file *os.File) {
	r.filesMu.Lock()
	defer r.filesMu.Unlock()
	if r.closed.Load() {
		if err := file.Close(); err != nil {
			r.setErr(err)
		}
		return
	}
	r.idle = append(r.idle, file)
}

func (r *fileIOReader) setErr(err error) {
	if err == nil {
		return
	}

	r.errMu.Lock()
	defer r.errMu.Unlock()
	if r.err == nil {
		r.err = err
	}
}

func (r *fileIOReader) Err() error {
	r.errMu.Lock()
	defer r.errMu.Unlock()
	return r.err
}

func (r *fileIOReader) close() {
	if r == nil || r.closed.Swap(true) {
		return
	}

	r.filesMu.Lock()
	files := r.idle
	r.idle = nil
	r.filesMu.Unlock()

	for _, file := range files {
		if file == nil {
			continue
		}
		if err := file.Close(); err != nil {
			r.setErr(err)
		}
	}
	if r.label != nil {
		C.free(r.label)
		r.label = nil
	}
}
