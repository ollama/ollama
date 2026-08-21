//go:build darwin

package main

type claudeDesktopInstallResult string

const (
	claudeDesktopInstallCancelled claudeDesktopInstallResult = "cancelled"
	claudeDesktopInstallerOpened  claudeDesktopInstallResult = "opened"
	claudeDesktopInstallFailed    claudeDesktopInstallResult = "failed"
)

type claudeDesktopStatus struct {
	Installed   bool `json:"installed"`
	Connected   bool `json:"connected"`
	Running     bool `json:"running"`
	StartFailed bool `json:"startFailed"`
}

type claudeDesktopActionResult struct {
	Status claudeDesktopStatus `json:"status"`
	Error  string              `json:"error,omitempty"`
}
