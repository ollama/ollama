//go:build darwin

package main

import "github.com/ollama/ollama/app/webview"

func bindClaudeDesktop(wv webview.WebView) {
	wv.Bind("getClaudeDesktopStatus", func() claudeDesktopStatus {
		return getClaudeDesktopConnectionStatus()
	})

	wv.Bind("setClaudeDesktopConnected", func(enabled bool) claudeDesktopActionResult {
		err := setClaudeDesktopConnection(enabled)
		result := claudeDesktopActionResult{
			Status: getClaudeDesktopConnectionStatus(),
		}
		if err != nil {
			result.Error = err.Error()
		}
		return result
	})

	wv.Bind("prepareClaudeDesktopConnection", func() claudeDesktopActionResult {
		err := prepareClaudeDesktopConnection()
		result := claudeDesktopActionResult{
			Status: getClaudeDesktopConnectionStatus(),
		}
		if err != nil {
			result.Error = err.Error()
		}
		return result
	})

	wv.Bind("openClaudeDesktop", func() string {
		if err := openClaudeDesktopApplication(); err != nil {
			return err.Error()
		}
		return ""
	})

	wv.Bind("installClaudeDesktop", func() claudeDesktopInstallResult {
		return requestClaudeDesktopInstall()
	})

	wv.Bind("restartClaudeDesktop", func(models []string) claudeDesktopActionResult {
		err := restartClaudeDesktopWithModels(models)
		result := claudeDesktopActionResult{Status: getClaudeDesktopConnectionStatus()}
		if err != nil {
			result.Error = err.Error()
		}
		return result
	})

	wv.Bind("getShowAppsInMenu", func() bool {
		return getShowAppsInMenu()
	})

	wv.Bind("setShowAppsInMenu", func(visible bool) {
		setShowAppsInMenu(visible)
	})
}
