//go:build windows && winui

package main

import (
	"log/slog"

	"github.com/ollama/ollama/app/webview"
)

func runWebviewEventLoop(webview.WebView) {
	// The WinUI tray owns the primary Windows message loop. WebView2's windows
	// are created on that same thread, so IUP services them too.
}

func navigateExistingWebview(wv webview.WebView, url, path string) {
	// A history-only URL change does not notify the SPA router. Navigate the
	// existing WebView so tray actions reliably switch between chat and settings.
	slog.Debug("navigating existing Ollama UI", "path", path)
	wv.Navigate(url)
}

func terminateWebview(wv webview.WebView) {
	// Destroy WebView2 on the primary UI thread where it was created. IUP's
	// message loop executes this dispatch before the queued tray exit.
	wv.Dispatch(wv.Destroy)
}
