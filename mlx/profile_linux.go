//go:build linux

// On Linux the shipping GPU backend is CUDA, so we emit NVTX ranges that Nsight
// Systems captures. We dlopen libnvToolsExt at first use rather than linking it,
// so the runner has no hard dependency on the CUDA toolkit when profiling is
// off. Backend selection is funneled through load_markers() so a future ROCm
// (roctx) path drops into the marked slot without restructuring.

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
//     void *h = dlopen("libnvToolsExt.so.1", RTLD_NOW | RTLD_GLOBAL);
//     if (h == NULL) {
//         h = dlopen("libnvToolsExt.so", RTLD_NOW | RTLD_GLOBAL);
//     }
//     if (h != NULL) {
//         _range_push = (range_push_t)dlsym(h, "nvtxRangePushA");
//         _range_pop  = (range_pop_t)dlsym(h, "nvtxRangePop");
//         if (_range_push != NULL && _range_pop != NULL) {
//             return 1;
//         }
//     }
//
//     // Future ROCm slot: dlopen("libroctx64.so") + roctxRangePushA/roctxRangePop
//     // (captured by rocprofv3). Not implemented yet.
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

func profileRangePush(name string) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.marker_push(cName)
}

func profileRangePop() {
	C.marker_pop()
}
