//go:build darwin || (windows && !winui)

package main

import (
	"fmt"
	"log/slog"
	"runtime"

	"github.com/ollama/ollama/app/webview"
)

func runWebviewEventLoop(wv webview.WebView) {
	// On Darwin, we can't have two threads both running global event loops. On
	// legacy Windows builds the event loops are tied to their windows.
	if runtime.GOOS == "darwin" {
		return
	}
	slog.Debug("starting webview event loop")
	go func() {
		wv.Run()
		slog.Debug("webview event loop exited")
	}()
}

func navigateExistingWebview(wv webview.WebView, _ string, path string) {
	wv.Eval(fmt.Sprintf(`
		history.pushState({}, '', '%s');
	`, path))
}

func terminateWebview(wv webview.WebView) {
	wv.Terminate()
	wv.Destroy()
}
