//go:build darwin

package main

import (
	"errors"
	"log/slog"

	"github.com/ollama/ollama/app/webview"
)

func codexDesktopModelRefreshError(settings codexDesktopModelsSettings) string {
	if len(settings.Selected) > 0 {
		return "Couldn’t refresh available models. Your saved models are unchanged."
	}
	return "Couldn’t refresh available models. Try again."
}

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
			result.Warning = codexDesktopModelRefreshError(settings)
			slog.Warn("failed to refresh available ChatGPT models", "error", err)
		}
		return result
	})
	wv.Bind("applyCodexDesktopModels", func(models []string, restartConfirmed bool) codexDesktopModelsSettingsResult {
		err := applyCodexDesktopModels(models, restartConfirmed)
		settings, statusErr := getCodexDesktopModelsSettings()
		result := codexDesktopModelsSettingsResult{Settings: settings}
		if errors.Is(err, errCodexDesktopRestartConfirmationRequired) {
			result.RestartConfirmationRequired = true
		} else if err != nil {
			result.Error = err.Error()
		} else if statusErr != nil {
			result.Warning = codexDesktopModelRefreshError(settings)
			slog.Warn("failed to refresh available ChatGPT models after applying settings", "error", statusErr)
		}
		return result
	})
	wv.Bind("resetCodexDesktopModels", func() codexDesktopModelsSettingsResult {
		err := resetCodexDesktopModels()
		settings, statusErr := getCodexDesktopModelsSettings()
		result := codexDesktopModelsSettingsResult{Settings: settings}
		if err != nil {
			result.Error = err.Error()
		} else if statusErr != nil {
			result.Warning = codexDesktopModelRefreshError(settings)
			slog.Warn("failed to refresh available ChatGPT models after resetting settings", "error", statusErr)
		}
		return result
	})
}
