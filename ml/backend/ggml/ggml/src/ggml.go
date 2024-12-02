package ggml

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -DNDEBUG -DGGML_USE_CPU
// #cgo CPPFLAGS: -I${SRCDIR}/../include -I${SRCDIR}/ggml-cpu
// #include <stdlib.h>
// #include "ggml-backend.h"
// extern void sink(int level, char *text, void *user_data);
import "C"

import (
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	_ "github.com/ollama/ollama/ml/backend/ggml/ggml/src/ggml-cpu"
)

func init() {
	C.ggml_log_set((C.ggml_log_callback)(C.sink), nil)
}

//export sink
func sink(level C.int, text *C.char, _ unsafe.Pointer) {
	msg := strings.TrimSpace(C.GoString(text))
	switch level {
	case C.GGML_LOG_LEVEL_DEBUG:
		slog.Debug(msg)
	case C.GGML_LOG_LEVEL_INFO:
		slog.Info(msg)
	case C.GGML_LOG_LEVEL_WARN:
		slog.Warn(msg)
	case C.GGML_LOG_LEVEL_ERROR:
		slog.Error(msg)
	}
}

var OnceLoad = sync.OnceFunc(func() {
	var lib struct{ name, defaultValue string }
	switch runtime.GOOS {
	case "darwin", "linux":
		lib.name = "LD_LIBRARY_PATH"
		lib.defaultValue = "/usr/local/lib:/usr/lib"
	case "windows":
		lib.name = "PATH"
		lib.defaultValue = "."
	default:
		return
	}

	paths, ok := os.LookupEnv(lib.name)
	if !ok {
		paths = lib.defaultValue
	}

	for _, path := range filepath.SplitList(paths) {
		func() {
			cpath := C.CString(path)
			defer C.free(unsafe.Pointer(cpath))
			C.ggml_backend_load_all_from_path(cpath)
		}()
	}
})
