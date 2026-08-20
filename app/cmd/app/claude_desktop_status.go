//go:build darwin

package main

type claudeDesktopInstallResult string

const (
	claudeDesktopInstallCancelled claudeDesktopInstallResult = "cancelled"
	claudeDesktopInstallerOpened  claudeDesktopInstallResult = "opened"
	claudeDesktopInstallFailed    claudeDesktopInstallResult = "failed"
)

type claudeDesktopStatus struct {
	Supported    bool   `json:"supported"`
	Installed    bool   `json:"installed"`
	Connected    bool   `json:"connected"`
	Running      bool   `json:"running"`
	StartFailed  bool   `json:"startFailed"`
	PortConflict bool   `json:"portConflict"`
	GatewayPort  int    `json:"gatewayPort,omitempty"`
	Error        string `json:"error,omitempty"`
}

type claudeDesktopActionResult struct {
	Status claudeDesktopStatus `json:"status"`
	Error  string              `json:"error,omitempty"`
}
