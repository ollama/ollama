package llm

// #cgo CPPFLAGS: -Illama.cpp/ggml/include
// #cgo LDFLAGS: -lllama -lggml -lstdc++ -lpthread
// #cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/build/darwin/arm64_static -L${SRCDIR}/build/darwin/arm64_static/src -L${SRCDIR}/build/darwin/arm64_static/ggml/src -framework Accelerate -framework Metal
// #cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/build/darwin/x86_64_static -L${SRCDIR}/build/darwin/x86_64_static/src -L${SRCDIR}/build/darwin/x86_64_static/ggml/src
// #cgo windows,amd64 LDFLAGS: -static-libstdc++ -static-libgcc -static -L${SRCDIR}/build/windows/amd64_static -L${SRCDIR}/build/windows/amd64_static/src -L${SRCDIR}/build/windows/amd64_static/ggml/src
// #cgo windows,arm64 LDFLAGS: -static-libstdc++ -static-libgcc -static -L${SRCDIR}/build/windows/arm64_static -L${SRCDIR}/build/windows/arm64_static/src -L${SRCDIR}/build/windows/arm64_static/ggml/src
// #cgo linux,amd64 LDFLAGS: -L${SRCDIR}/build/linux/x86_64_static -L${SRCDIR}/build/linux/x86_64_static/src -L${SRCDIR}/build/linux/x86_64_static/ggml/src
// #cgo linux,arm64 LDFLAGS: -L${SRCDIR}/build/linux/arm64_static -L${SRCDIR}/build/linux/arm64_static/src -L${SRCDIR}/build/linux/arm64_static/ggml/src
// #include <stdlib.h>
// #include <stdatomic.h>
// #include "llama.h"
// bool update_quantize_progress(float progress, void* data) {
//	atomic_int* atomicData = (atomic_int*)data;
//  int intProgress = *((int*)&progress);
//  atomic_store(atomicData, intProgress);
//  return true;
// }
import "C"
import (
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/ollama/ollama/api"
)

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}

func Quantize(infile, outfile string, ftype fileType, fn func(resp api.ProgressResponse), tensorCount int) error {
	cinfile := C.CString(infile)
	defer C.free(unsafe.Pointer(cinfile))

	coutfile := C.CString(outfile)
	defer C.free(unsafe.Pointer(coutfile))
	params := C.llama_model_quantize_default_params()
	params.nthread = -1
	params.ftype = ftype.Value()

	// Initialize "global" to store progress
	store := (*int32)(C.malloc(C.sizeof_int))
	defer C.free(unsafe.Pointer(store))

	// Initialize store value, e.g., setting initial progress to 0
	atomic.StoreInt32(store, 0)

	params.quantize_callback_data = unsafe.Pointer(store)
	params.quantize_callback = (C.llama_progress_callback)(C.update_quantize_progress)

	ticker := time.NewTicker(30 * time.Millisecond)
	done := make(chan struct{})
	defer close(done)

	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				progressInt := atomic.LoadInt32(store)
				progress := *(*float32)(unsafe.Pointer(&progressInt))
				fn(api.ProgressResponse{
					Status: fmt.Sprintf("quantizing model %d/%d", int(progress), tensorCount),
					Type:   "quantize",
				})
			case <-done:
				fn(api.ProgressResponse{
					Status: fmt.Sprintf("quantizing model %d/%d", tensorCount, tensorCount),
					Type:   "quantize",
				})
				return
			}
		}
	}()

	if rc := C.llama_model_quantize(cinfile, coutfile, &params); rc != 0 {
		return fmt.Errorf("failed to quantize model. This model architecture may not be supported, or you may need to upgrade Ollama to the latest version")
	}

	return nil
}
