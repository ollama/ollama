package ggml

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -DNDEBUG -DGGML_USE_CPU
// #cgo CPPFLAGS: -I${SRCDIR}/../include -I${SRCDIR}/ggml-cpu
// #include <stdlib.h>
// #include "ggml-backend.h"
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

var OnceLoad = sync.OnceFunc(func() {
	var lib struct{ name, pattern, defaultValue string }
	switch runtime.GOOS {
	case "darwin":
		lib.name = "LD_LIBRARY_PATH"
		lib.pattern = "libggml-*.dylib"
		lib.defaultValue = "/usr/local/lib:/usr/lib"
	case "linux":
		lib.name = "LD_LIBRARY_PATH"
		lib.pattern = "libggml-*.so"
		lib.defaultValue = "/usr/local/lib:/usr/lib"
	case "windows":
		lib.name = "PATH"
		lib.pattern = "ggml-*.dll"
		lib.defaultValue = "."
	default:
		return
	}

	paths, ok := os.LookupEnv(lib.name)
	if !ok {
		paths = lib.defaultValue
	}

	for _, path := range filepath.SplitList(paths) {
		matches, err := filepath.Glob(filepath.Join(path, lib.pattern))
		if err != nil {
			slog.Error("failed to glob", "path", path, "error", err)
			continue
		}

		for _, match := range matches {
			if base := filepath.Base(match); strings.HasPrefix(base, "ggml-base") ||
				strings.HasPrefix(base, "libggml-base") {
				continue
			}

			func() {
				cmatch := C.CString(match)
				defer C.free(unsafe.Pointer(cmatch))

				C.ggml_backend_load(cmatch)
			}()
		}
	}
})
