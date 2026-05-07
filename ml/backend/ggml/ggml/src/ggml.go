package ggml

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -DNDEBUG -DGGML_USE_CPU -DGGML_VERSION=0x0 -DGGML_COMMIT=0x0
// #cgo CPPFLAGS: -I${SRCDIR}/../include -I${SRCDIR}/ggml-cpu
// #cgo windows CFLAGS: -Wno-dll-attribute-on-redeclaration
// #cgo windows LDFLAGS: -lmsvcrt -static -static-libgcc -static-libstdc++
// #include <stdlib.h>
// #include "ggml-backend.h"
// extern void sink(int level, char *text, void *user_data);
// static struct ggml_backend_feature * first_feature(ggml_backend_get_features_t fp, ggml_backend_reg_t reg) { return fp(reg); }
// static struct ggml_backend_feature * next_feature(struct ggml_backend_feature * feature) { return &feature[1]; }
/*
typedef enum { COMPILER_CLANG, COMPILER_GNUC, COMPILER_UNKNOWN } COMPILER;
static COMPILER compiler_name(void) {
#if defined(__clang__)
	return COMPILER_CLANG;
#elif defined(__GNUC__)
	return COMPILER_GNUC;
#else
	return COMPILER_UNKNOWN;
#endif
}
*/
import "C"

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/ollama/ollama/envconfig"
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

	var value string
	switch runtime.GOOS {
	case "darwin":
		value = filepath.Dir(exe)
	case "windows":
		value = filepath.Join(filepath.Dir(exe), "lib", "ollama")
	default:
		value = filepath.Join(filepath.Dir(exe), "..", "lib", "ollama")
	}

	// Avoid potentially loading incompatible GGML libraries
	paths, ok := os.LookupEnv("OLLAMA_LIBRARY_PATH")
	if !ok {
		slog.Debug("OLLAMA_LIBRARY_PATH not set, falling back to default", "search", value)
		paths = value
	}
	allowExternalLibraryPath := ok && envconfig.AllowExternalLibraryPath()

	libPaths = filepath.SplitList(paths)
	visited := make(map[string]struct{}, len(libPaths))
	for _, path := range libPaths {
		abspath, err := filepath.Abs(path)
		if err != nil {
			slog.Error("failed to get absolute path", "error", err)
			continue
		}

		isOllamaPath := abspath == filepath.Dir(exe) || strings.Contains(abspath, filepath.FromSlash("lib/ollama"))
		if !allowExternalLibraryPath && !isOllamaPath {
			slog.Debug("skipping path which is not part of ollama", "path", abspath)
			continue
		}

		if _, ok := visited[abspath]; !ok {
			if allowExternalLibraryPath && !isOllamaPath {
				loadExternalBackends(abspath)
			} else {
				slog.Debug("ggml backend load all from path", "path", abspath)
				cpath := C.CString(abspath)
				defer C.free(unsafe.Pointer(cpath))
				C.ggml_backend_load_all_from_path(cpath)
			}

			visited[abspath] = struct{}{}
		}
	}

	slog.Info("system", "", system{})
})

var libPaths []string

func loadExternalBackends(dir string) {
	path := filepath.Join(dir, backendLibraryName("sycl"))
	if _, err := os.Stat(path); err != nil {
		slog.Debug("external ggml backend not found", "path", path, "error", err)
		return
	}

	slog.Debug("ggml external backend load", "path", path)
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	C.ggml_backend_load(cpath)
}

func backendLibraryName(name string) string {
	switch runtime.GOOS {
	case "windows":
		return "ggml-" + name + ".dll"
	case "darwin":
		return "libggml-" + name + ".dylib"
	default:
		return "libggml-" + name + ".so"
	}
}

func LibPaths() []string {
	return libPaths
}

type system struct{}

func (system) LogValue() slog.Value {
	var attrs []slog.Attr
	if envconfig.AllowExternalLibraryPath() {
		return slog.GroupValue(append(attrs, compilerAttr())...)
	}

	names := make(map[string]int)
	for i := range C.ggml_backend_dev_count() {
		r := C.ggml_backend_dev_backend_reg(C.ggml_backend_dev_get(i))

		func() {
			fName := C.CString("ggml_backend_get_features")
			defer C.free(unsafe.Pointer(fName))

			if fn := C.ggml_backend_reg_get_proc_address(r, fName); fn != nil {
				var features []any
				for f := C.first_feature(C.ggml_backend_get_features_t(fn), r); f.name != nil; f = C.next_feature(f) {
					features = append(features, C.GoString(f.name), C.GoString(f.value))
				}

				name := C.GoString(C.ggml_backend_reg_name(r))
				attrs = append(attrs, slog.Group(name+"."+strconv.Itoa(names[name]), features...))
				names[name] += 1
			}
		}()
	}

	attrs = append(attrs, compilerAttr())
	return slog.GroupValue(attrs...)
}

func compilerAttr() slog.Attr {
	switch C.compiler_name() {
	case C.COMPILER_CLANG:
		return slog.String("compiler", "cgo(clang)")
	case C.COMPILER_GNUC:
		return slog.String("compiler", "cgo(gcc)")
	default:
		return slog.String("compiler", "cgo(unknown)")
	}
}
