package mlx

// #include <stdbool.h>
// #include <stddef.h>
// #include <stdlib.h>
// #include <stdint.h>
import "C"

import (
	"fmt"
	"runtime/cgo"
	"unsafe"
)

func readerFromHandle(desc unsafe.Pointer) *fileIOReader {
	if desc == nil {
		return nil
	}
	handle := *(*cgo.Handle)(desc)
	reader, _ := handle.Value().(*fileIOReader)
	return reader
}

//export goMLXReaderIsOpen
func goMLXReaderIsOpen(desc unsafe.Pointer) C.bool {
	reader := readerFromHandle(desc)
	return C.bool(reader != nil && reader.isOpen())
}

//export goMLXReaderGood
func goMLXReaderGood(desc unsafe.Pointer) C.bool {
	reader := readerFromHandle(desc)
	return C.bool(reader != nil && reader.good())
}

//export goMLXReaderTell
func goMLXReaderTell(desc unsafe.Pointer) C.size_t {
	reader := readerFromHandle(desc)
	if reader == nil {
		return 0
	}
	return C.size_t(reader.tell())
}

//export goMLXReaderSeek
func goMLXReaderSeek(desc unsafe.Pointer, off C.int64_t, whence C.int) {
	if reader := readerFromHandle(desc); reader != nil {
		reader.seek(int64(off), int(whence))
	}
}

//export goMLXReaderRead
func goMLXReaderRead(desc unsafe.Pointer, data *C.char, n C.size_t) {
	if reader := readerFromHandle(desc); reader != nil {
		reader.read(unsafe.Pointer(data), uint64(n))
	}
}

//export goMLXReaderReadAtOffset
func goMLXReaderReadAtOffset(desc unsafe.Pointer, data *C.char, n C.size_t, off C.size_t) {
	if reader := readerFromHandle(desc); reader != nil {
		reader.readAt(unsafe.Pointer(data), uint64(n), uint64(off))
	}
}

//export goMLXReaderWrite
func goMLXReaderWrite(desc unsafe.Pointer, data *C.char, n C.size_t) {
	if reader := readerFromHandle(desc); reader != nil {
		_, _ = data, n
		reader.setErr(fmt.Errorf("write on read-only safetensors reader"))
	}
}

//export goMLXReaderLabel
func goMLXReaderLabel(desc unsafe.Pointer) *C.char {
	reader := readerFromHandle(desc)
	if reader == nil || reader.label == nil {
		return nil
	}
	return (*C.char)(reader.label)
}

//export goMLXReaderFree
func goMLXReaderFree(desc unsafe.Pointer) {
	if desc == nil {
		return
	}

	handle := *(*cgo.Handle)(desc)
	if reader, ok := handle.Value().(*fileIOReader); ok {
		reader.close()
	}
	handle.Delete()
	C.free(desc)
}
