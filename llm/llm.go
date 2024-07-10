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
// #include "llama.h"
// bool update_quantize_progress(float progress, void* data) {
//  *((float*)data) = progress;
// 	return true;
// }
import "C"
import (
	"fmt"
	"unsafe"
	"time"

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
	store := C.malloc(C.sizeof_float)
	defer C.free(unsafe.Pointer(store))

	// Initialize store value, e.g., setting initial progress to 0
	*(*C.float)(store) = 0.0

	params.quantize_callback_data = store
	params.quantize_callback = (C.llama_progress_callback)(C.update_quantize_progress)

	ticker := time.NewTicker(60 * time.Millisecond)
	done := make(chan struct{})
	defer close(done)

	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				fn(api.ProgressResponse{
					Status:   fmt.Sprintf("quantizing model tensors %d/%d", int(*((*C.float)(store))), tensorCount),
					Quantize: "quant",
				})			
				fmt.Println("Progress: ", *((*C.float)(store)))
			case <-done:
				fn(api.ProgressResponse{
					Status:   fmt.Sprintf("quantizing model tensors %d/%d", tensorCount, tensorCount),
					Quantize: "quant",
				})
				return
			}
		}
	}()

	if rc := C.llama_model_quantize(cinfile, coutfile, &params); rc != 0 {
		return fmt.Errorf("llama_model_quantize: %d", rc)
	}

	return nil
}
