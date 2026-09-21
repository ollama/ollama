#include "webview.h"

// Called by Go only after dispatching to the UI thread and checking the view's
// lifetime. Do not queue another callback holding the native view pointer.
extern "C" void CgoWebViewReturn(webview_t w, const char *id, int status,
                                const char *result) {
  static_cast<webview::webview *>(w)->resolve_on_main_thread(id, status, result);
}
