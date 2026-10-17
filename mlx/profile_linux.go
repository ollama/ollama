//go:build linux

// Emit NVTX ranges that Nsight Systems captures. Load libnvToolsExt at first
// use so profiling adds no link-time dependency on the CUDA toolkit.

package mlx

// #cgo LDFLAGS: -ldl
// #include <dlfcn.h>
// #include <stdlib.h>
//
// typedef int (*range_push_t)(const char *);
// typedef int (*range_pop_t)(void);
//
// static range_push_t _range_push = NULL;
// static range_pop_t  _range_pop  = NULL;
// static int          _markers_tried = 0;
//
// // load_markers resolves a push/pop marker pair the first time it is called.
// // Returns 1 if markers are available, 0 otherwise.
// static int load_markers(void) {
//     if (_markers_tried) {
//         return _range_push != NULL && _range_pop != NULL;
//     }
//     _markers_tried = 1;
//
//     // CUDA: NVTX (captured by Nsight Systems).
//     void *h = dlopen("libnvToolsExt.so.1", RTLD_NOW | RTLD_LOCAL);
//     if (h == NULL) {
//         h = dlopen("libnvToolsExt.so", RTLD_NOW | RTLD_LOCAL);
//     }
//     if (h != NULL) {
//         _range_push = (range_push_t)dlsym(h, "nvtxRangePushA");
//         _range_pop  = (range_pop_t)dlsym(h, "nvtxRangePop");
//         if (_range_push != NULL && _range_pop != NULL) {
//             return 1;
//         }
//     }
//
//     _range_push = NULL;
//     _range_pop  = NULL;
//     return 0;
// }
//
// static void marker_push(const char *msg) {
//     if (load_markers()) {
//         _range_push(msg);
//     }
// }
//
// static void marker_pop(void) {
//     if (load_markers()) {
//         _range_pop();
//     }
// }
import "C"

import "unsafe"

func profileMarkersAvailable() bool { return C.load_markers() == 1 }

func profileRangePush(name string) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.marker_push(cName)
}

func profileRangePop() {
	C.marker_pop()
}
