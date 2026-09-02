//go:build darwin

package main

import (
	"errors"
	"log/slog"

	"github.com/ollama/ollama/app/webview"
)

func bindCodexDesktop(wv webview.WebView) {
	wv.Bind("getCodexDesktopStatus", func() codexDesktopStatus {
		return getCodexDesktopStatus()
	})
	wv.Bind("getCodexDesktopRequestCount", func() uint64 {
		return codexDesktop.OllamaRequestCount()
	})
	wv.Bind("setCodexDesktopConnected", func(enabled, restartConfirmed bool) codexDesktopActionResult {
		err := setCodexDesktopConnection(enabled, restartConfirmed)
		result := codexDesktopActionResult{Status: getCodexDesktopStatus()}
		if errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
			result.RestartConfirmationRequired = true
		} else if err != nil {
			result.Error = err.Error()
			slog.Warn("failed to change ChatGPT integration from Settings", "connected", enabled, "error", err)
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
