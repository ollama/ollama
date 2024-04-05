package llm

// #cgo CFLAGS: -Illama.cpp
// #cgo darwin,arm64 LDFLAGS: ${SRCDIR}/build/darwin/arm64_static/libllama.a -lstdc++
// #cgo darwin,amd64 LDFLAGS: ${SRCDIR}/build/darwin/x86_64_static/libllama.a -lstdc++
// #cgo windows,amd64 LDFLAGS: ${SRCDIR}/build/windows/amd64_static/libllama.a -static -lstdc++
// #cgo linux,amd64 LDFLAGS: ${SRCDIR}/build/linux/x86_64_static/libllama.a -lstdc++
// #cgo linux,arm64 LDFLAGS: ${SRCDIR}/build/linux/arm64_static/libllama.a -lstdc++
// #include "llama.h"
import "C"

// SystemInfo is an unused example of calling llama.cpp functions using CGo
func SystemInfo() string {
	return C.GoString(C.llama_print_system_info())
}
