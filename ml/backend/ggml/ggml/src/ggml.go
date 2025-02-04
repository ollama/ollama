package ggml

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -DNDEBUG -DGGML_USE_CPU
// #cgo CPPFLAGS: -I${SRCDIR}/../include -I${SRCDIR}/ggml-cpu
// #cgo windows LDFLAGS: -lmsvcrt -static -static-libgcc -static-libstdc++
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
	var name, value string
	switch runtime.GOOS {
	case "darwin":
		name = "DYLD_LIBRARY_PATH"
		value = "."
	case "windows":
		name = "PATH"
		value = "."
	default:
		name = "LD_LIBRARY_PATH"
		value = "/usr/local/lib:/usr/lib"
	}

	paths, ok := os.LookupEnv(name)
	if !ok {
		paths = value
	}

	split := filepath.SplitList(paths)
	visited := make(map[string]struct{}, len(split))
	for _, path := range split {
		abspath, _ := filepath.Abs(path)
		if _, ok := visited[abspath]; !ok {
			func() {
				slog.Debug("Loading backend from", "path", abspath)
				cpath := C.CString(abspath)
				defer C.free(unsafe.Pointer(cpath))
				C.ggml_backend_load_all_from_path(cpath)
			}()

			visited[abspath] = struct{}{}
		}
	}
})
