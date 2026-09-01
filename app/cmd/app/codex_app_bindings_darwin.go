//go:build darwin

package main

import "github.com/ollama/ollama/app/webview"

func bindCodexDesktop(wv webview.WebView) {
	wv.Bind("getCodexDesktopStatus", func() codexDesktopStatus {
		return getCodexDesktopStatus()
	})
	wv.Bind("getCodexDesktopRequestCount", func() uint64 {
		return codexDesktop.OllamaRequestCount()
	})
	wv.Bind("setCodexDesktopConnected", func(enabled bool) codexDesktopActionResult {
		err := setCodexDesktopConnection(enabled)
		result := codexDesktopActionResult{Status: getCodexDesktopStatus()}
		if err != nil {
			result.Error = err.Error()
		}
		return result
	})
	wv.Bind("installCodexDesktop", func() codexDesktopInstallResult {
		return requestCodexDesktopInstall()
	})
	wv.Bind("getCodexDesktopModelsSettings", func() codexDesktopModelsSettingsResult {
		settings, err := getCodexDesktopModelsSettings()
		result := codexDesktopModelsSettingsResult{Settings: settings}
		if err != nil {
			result.Error = err.Error()
		}
		return result
	})
	wv.Bind("applyCodexDesktopModels", func(models []string) codexDesktopModelsSettingsResult {
		err := applyCodexDesktopModels(models)
		settings, statusErr := getCodexDesktopModelsSettings()
		result := codexDesktopModelsSettingsResult{Settings: settings}
		if err != nil {
			result.Error = err.Error()
		} else if statusErr != nil {
			result.Error = statusErr.Error()
		}
		return result
	})
}
