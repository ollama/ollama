// TODO: embed the metal library
//
//go:generate bash metal.sh
package llama

// #cgo CFLAGS: -I.
// #cgo CXXFLAGS: -std=c++11 -DGGML_USE_METAL
// #cgo darwin,arm64 LDFLAGS: -framework Foundation -framework Metal -framework MetalKit -framework Accelerate
// #include <stdlib.h>
// #include "llama.h"
import "C"

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}
