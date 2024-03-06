package llm

// #cgo CFLAGS: -Illama.cpp
// #cgo LDFLAGS: -lstdc++ -lllama
// #cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/build/darwin/arm64/metal -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
// #cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/build/darwin/x86_64/cpu
// #cgo windows,amd64 LDFLAGS: -L${SRCDIR}/build/windows/amd64/cpu/bin/Release
// #cgo linux,amd64 LDFLAGS: -L${SRCDIR}/build/linux/x86_64/cpu
// #cgo linux,arm64 LDFLAGS: -L${SRCDIR}/build/linux/arm64/cpu
// #include "llama.h"
import "C"

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}
