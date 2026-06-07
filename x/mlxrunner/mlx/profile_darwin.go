//go:build darwin

// os_signpost intervals land in Instruments / xctrace's "Points of Interest"
// track, letting a Metal System Trace be split by inference phase. We use a
// single static signpost name ("phase") and pass the actual phase name as the
// interval message so prefill/decode show up as labeled sub-intervals. Begin
// returns a generated signpost id which end must match, so we keep a small
// stack; pushes/pops run on the serialized MLX worker thread.

package mlx

// #include <os/log.h>
// #include <os/signpost.h>
// #include <stdlib.h>
//
// static os_log_t _ollama_signpost_log(void) {
//     // Initialized on first use from the single MLX worker thread.
//     static os_log_t log = NULL;
//     if (log == NULL) {
//         log = os_log_create("com.ollama.mlx", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
//     }
//     return log;
// }
//
// static os_signpost_id_t _sp_stack[32];
// static int _sp_top = 0;
//
// static void ollama_signpost_begin(const char *name) {
//     os_log_t log = _ollama_signpost_log();
//     os_signpost_id_t sid = os_signpost_id_generate(log);
//     if (_sp_top < (int)(sizeof(_sp_stack) / sizeof(_sp_stack[0]))) {
//         _sp_stack[_sp_top++] = sid;
//     }
//     os_signpost_interval_begin(log, sid, "phase", "%s", name);
// }
//
// static void ollama_signpost_end(void) {
//     if (_sp_top <= 0) {
//         return;
//     }
//     os_log_t log = _ollama_signpost_log();
//     os_signpost_id_t sid = _sp_stack[--_sp_top];
//     os_signpost_interval_end(log, sid, "phase");
// }
import "C"

import "unsafe"

func profileRangePush(name string) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	C.ollama_signpost_begin(cName)
}

func profileRangePop() {
	C.ollama_signpost_end()
}
