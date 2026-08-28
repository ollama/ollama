//go:build windows

package main

import (
	"errors"

	"github.com/ollama/ollama/app/webview"
	"github.com/ollama/ollama/cmd/launch"
)

// openClaudeDesktopDownloadPage sends the user to Claude's download page so the
// Windows app can be installed before Ollama connects it.
func openClaudeDesktopDownloadPage() error {
	openInBrowser("https://claude.ai/download")
	return nil
}

func bindClaudeDesktop(wv webview.WebView) {
	wv.Bind("getClaudeDesktopStatus", func() claudeDesktopStatus {
		return getClaudeDesktopConnectionStatus()
	})
	wv.Bind("getClaudeDesktopConnectionSummary", func() claudeDesktopStatus {
		return getClaudeDesktopConnectionSummary()
	})
	wv.Bind("getClaudeDesktopRequestCount", func() uint64 {
		return claudeDesktopRequestCount()
	})

	wv.Bind("setClaudeDesktopConnected", func(enabled, restartConfirmed bool) claudeDesktopActionResult {
		err := setClaudeDesktopConnection(enabled, restartConfirmed)
		result := claudeDesktopActionResult{
			Status: getClaudeDesktopConnectionSummary(),
		}
		if err != nil {
			result.Error = err.Error()
		}
		return result
	})

	wv.Bind("prepareClaudeDesktopConnection", func() claudeDesktopActionResult {
		err := prepareClaudeDesktopConnection()
		result := claudeDesktopActionResult{
			Status: getClaudeDesktopConnectionSummary(),
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

	wv.Bind("applyClaudeDesktopMappings", func(mappings map[string]string, restartConfirmed bool) claudeDesktopActionResult {
		applied, err := applyClaudeDesktopMappings(mappings, restartConfirmed)
		result := claudeDesktopActionResult{
			Status:          getClaudeDesktopConnectionStatus(),
			MappingsApplied: applied,
		}
		if err != nil {
			result.Error = err.Error()
			result.RestartConfirmationRequired = errors.Is(err, launch.ErrClaudeDesktopRestartConfirmationRequired)
		}
		return result
	})
	wv.Bind("resetClaudeDesktopMappings", func(restartConfirmed bool) claudeDesktopActionResult {
		applied, err := resetClaudeDesktopMappings(restartConfirmed)
		result := claudeDesktopActionResult{
			Status:          getClaudeDesktopConnectionStatus(),
			MappingsApplied: applied,
		}
		if err != nil {
			result.Error = err.Error()
			result.RestartConfirmationRequired = errors.Is(err, launch.ErrClaudeDesktopRestartConfirmationRequired)
		}
		return result
	})

	wv.Bind("setClaudeDesktopAutoMode", func(enabled, restartConfirmed bool) claudeDesktopActionResult {
		err := setClaudeDesktopAutoMode(enabled, restartConfirmed)
		result := claudeDesktopActionResult{Status: getClaudeDesktopConnectionStatus()}
		if err != nil {
			result.Error = err.Error()
			result.RestartConfirmationRequired = errors.Is(err, launch.ErrClaudeDesktopRestartConfirmationRequired)
		}
		return result
	})

	wv.Bind("getShowAppsInMenu", func() bool {
		return true
	})

	wv.Bind("setShowAppsInMenu", func(visible bool) {})
}

func requestClaudeDesktopInstall() claudeDesktopInstallResult {
	if openErr := openClaudeDesktopDownloadPage(); openErr != nil {
		return claudeDesktopInstallFailed
	}
	return claudeDesktopInstallerOpened
}
