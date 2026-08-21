//go:build windows || darwin

package main

type claudeDesktopInstallResult string

const (
	claudeDesktopInstallCancelled claudeDesktopInstallResult = "cancelled"
	claudeDesktopInstallerOpened  claudeDesktopInstallResult = "opened"
	claudeDesktopInstallFailed    claudeDesktopInstallResult = "failed"
)

type claudeDesktopStatus struct {
	Supported    bool                       `json:"supported"`
	Used         bool                       `json:"used"`
	Installed    bool                       `json:"installed"`
	Connected    bool                       `json:"connected"`
	Running      bool                       `json:"running"`
	StartFailed  bool                       `json:"startFailed"`
	PortConflict bool                       `json:"portConflict"`
	GatewayPort  int                        `json:"gatewayPort,omitempty"`
	Error        string                     `json:"error,omitempty"`
	ModelSource  string                     `json:"modelSource,omitempty"`
	Models       []claudeDesktopModelStatus `json:"models,omitempty"`
}

type claudeDesktopModelStatus struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
	Description string `json:"description,omitempty"`
	Cloud       bool   `json:"cloud"`
	Selected    bool   `json:"selected"`
}

type claudeDesktopActionResult struct {
	Status claudeDesktopStatus `json:"status"`
	Error  string              `json:"error,omitempty"`
}
