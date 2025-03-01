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
	"context"
	"fmt"
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
	C.ggml_log_set(C.ggml_log_callback(C.sink), nil)
}

//export sink
func sink(level C.int, text *C.char, _ unsafe.Pointer) {
	// slog levels zeros INFO and are multiples of 4
	if slog.Default().Enabled(context.TODO(), slog.Level(int(level-C.GGML_LOG_LEVEL_INFO)*4)) {
		fmt.Fprint(os.Stderr, C.GoString(text))
	}
}

var OnceLoad = sync.OnceFunc(func() {
	exe, err := os.Executable()
	if err != nil {
		slog.Warn("failed to get executable path", "error", err)
		exe = "."
	}

	// PATH, LD_LIBRARY_PATH, and DYLD_LIBRARY_PATH are often
	// set by the parent process, however, use a default value
	// if the environment variable is not set.
	var name, value string
	switch runtime.GOOS {
	case "darwin":
		// On macOS, DYLD_LIBRARY_PATH is often not set, so
		// we use the directory of the executable as the default.
		name = "DYLD_LIBRARY_PATH"
		value = filepath.Dir(exe)
	case "windows":
		name = "PATH"
		value = filepath.Join(filepath.Dir(exe), "lib", "ollama")
	default:
		name = "LD_LIBRARY_PATH"
		value = filepath.Join(filepath.Dir(exe), "..", "lib", "ollama")
	}

	paths, ok := os.LookupEnv(name)
	if !ok {
		paths = value
	}

	split := filepath.SplitList(paths)
	visited := make(map[string]struct{}, len(split))
	for _, path := range split {
		abspath, err := filepath.Abs(path)
		if err != nil {
			slog.Error("failed to get absolute path", "error", err)
			continue
		}

		if abspath != filepath.Dir(exe) && !strings.Contains(abspath, filepath.FromSlash("lib/ollama")) {
			slog.Debug("skipping path which is not part of ollama", "path", abspath)
			continue
		}

		if _, ok := visited[abspath]; !ok {
			func() {
				slog.Debug("ggml backend load all from path", "path", abspath)
				cpath := C.CString(abspath)
				defer C.free(unsafe.Pointer(cpath))
				C.ggml_backend_load_all_from_path(cpath)
			}()

			visited[abspath] = struct{}{}
		}
	}
})
