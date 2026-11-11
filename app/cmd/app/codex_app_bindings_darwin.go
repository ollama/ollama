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
	wv.Bind("markCodexDesktopIntegrationUsed", func() string {
		if err := markCodexDesktopIntegrationUsed(); err != nil {
			return err.Error()
		}
		return ""
	})
	wv.Bind("getCodexDesktopStatus", func() codexDesktopStatus {
		return getCodexDesktopStatus()
	})
	wv.Bind("getCodexDesktopRequestCount", func() uint64 {
		return codexDesktop.OllamaRequestCount()
	})
	wv.Bind("setCodexDesktopConnected", changeCodexDesktopConnection)
	wv.Bind("installCodexDesktop", func() codexDesktopInstallResult {
		return requestCodexDesktopInstall()
	})
	wv.Bind("applyCodexDesktopModels", func(models []string, restartConfirmed bool, versions ...string) codexDesktopModelsSettingsResult {
		err := applyCodexDesktopModels(models, restartConfirmed, versions...)
		settings, statusErr := getCodexDesktopModelsSettings()
		result := codexDesktopModelsSettingsResult{Settings: settings}
		var confirmation interface{ ConfirmationVersion() string }
		if errors.As(err, &confirmation) {
			result.RestartConfirmationRequired = true
			result.ConfirmationVersion = confirmation.ConfirmationVersion()
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
