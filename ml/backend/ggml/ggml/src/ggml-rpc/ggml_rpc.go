package rpc

// CGo compiles ggml-rpc.cpp and transport.cpp in this directory.
// Include paths:
//   ${SRCDIR}           → transport.h (same dir)
//   ${SRCDIR}/..        → ggml-impl.h, ggml-backend-impl.h, ggml-cpp.h (src/)
//   ${SRCDIR}/../../include → ggml-rpc.h, ggml-backend.h (include/)

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -DNDEBUG -DGGML_USE_RPC -DGGML_VERSION=0x0 -DGGML_COMMIT=0x0
// #cgo CPPFLAGS: -I${SRCDIR} -I${SRCDIR}/.. -I${SRCDIR}/../../include
// #cgo windows CFLAGS: -Wno-dll-attribute-on-redeclaration
// #cgo windows LDFLAGS: -lws2_32
// #include "ggml-rpc.h"
// #include "ggml-backend.h"
// #include <stdlib.h>
import "C"
import (
	"errors"
	"unsafe"
)

// AddServer registers a remote RPC server as a ggml backend device.
// endpoint is "host:port", e.g. "192.168.1.50:50052".
// After this call the remote machine's RAM/GPU appears in ggml_backend_dev_count()
// and will be picked up by EnumerateGPUs() in the llama package.
func AddServer(endpoint string) {
	cs := C.CString(endpoint)
	defer C.free(unsafe.Pointer(cs))
	C.ggml_backend_rpc_add_server(cs)
}

// GetDeviceMemory queries free and total memory on a specific device of a remote server.
// device is 0-indexed; for a CPU-only machine use device=0.
func GetDeviceMemory(endpoint string, device uint32) (free, total uint64) {
	cs := C.CString(endpoint)
	defer C.free(unsafe.Pointer(cs))
	var cfree, ctotal C.size_t
	C.ggml_backend_rpc_get_device_memory(cs, C.uint(device), &cfree, &ctotal)
	return uint64(cfree), uint64(ctotal)
}

// StartServer runs a blocking RPC server that exposes local compute devices to
// remote callers. This is the function called on the Raspberry Pi side.
// endpoint is "host:port" to bind on, e.g. "0.0.0.0:50052".
// cacheDir is a path for tensor caching (use "" for no caching).
// nThreads controls CPU thread count (0 = auto).
// Returns an error if no local compute devices are available to expose.
func StartServer(endpoint, cacheDir string, nThreads int) error {
	cEndpoint := C.CString(endpoint)
	defer C.free(unsafe.Pointer(cEndpoint))

	var cCacheDir *C.char
	if cacheDir != "" {
		cCacheDir = C.CString(cacheDir)
		defer C.free(unsafe.Pointer(cCacheDir))
	}

	// Enumerate local compute devices (CPU on the Pi, plus any GPUs). Skip
	// devices contributed by the RPC backend itself to avoid recursive exposure.
	n := C.ggml_backend_dev_count()
	devices := make([]C.ggml_backend_dev_t, 0, int(n))
	for i := C.size_t(0); i < n; i++ {
		dev := C.ggml_backend_dev_get(i)
		switch C.ggml_backend_dev_type(dev) {
		case C.GGML_BACKEND_DEVICE_TYPE_CPU,
			C.GGML_BACKEND_DEVICE_TYPE_GPU,
			C.GGML_BACKEND_DEVICE_TYPE_IGPU:
			devices = append(devices, dev)
		}
	}
	if len(devices) == 0 {
		return errors.New("no local compute devices available to expose via RPC")
	}

	C.ggml_backend_rpc_start_server(cEndpoint, cCacheDir,
		C.size_t(nThreads),
		C.size_t(len(devices)),
		(*C.ggml_backend_dev_t)(unsafe.Pointer(&devices[0])))
	return nil
}
