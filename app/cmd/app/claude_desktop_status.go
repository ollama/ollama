//go:build darwin

package main

import "github.com/ollama/ollama/internal/proxy"

type claudeDesktopInstallResult string

const (
	claudeDesktopInstallCancelled claudeDesktopInstallResult = "cancelled"
	claudeDesktopInstallerOpened  claudeDesktopInstallResult = "opened"
	claudeDesktopInstallFailed    claudeDesktopInstallResult = "failed"
)

type claudeDesktopStatus struct {
	Supported      bool                         `json:"supported"`
	Used           bool                         `json:"used"`
	Installed      bool                         `json:"installed"`
	Configured     bool                         `json:"configured"`
	Connected      bool                         `json:"connected"`
	Running        bool                         `json:"running"`
	StartFailed    bool                         `json:"startFailed"`
	PortConflict   bool                         `json:"portConflict"`
	GatewayPort    int                          `json:"gatewayPort,omitempty"`
	RoutedRequests uint64                       `json:"routedRequests"`
	Error          string                       `json:"error,omitempty"`
	AutoMode       bool                         `json:"autoMode"`
	ModelSource    string                       `json:"modelSource,omitempty"`
	Models         []claudeDesktopModelStatus   `json:"models,omitempty"`
	Mappings       []claudeDesktopMappingStatus `json:"mappings,omitempty"`
}

type claudeDesktopMappingStatus struct {
	RouteID   string `json:"routeId"`
	RouteName string `json:"routeName"`
	Model     string `json:"model,omitempty"`
}

type claudeDesktopModelStatus struct {
	Name         string                          `json:"name"`
	DisplayName  string                          `json:"displayName"`
	Description  string                          `json:"description,omitempty"`
	Cloud        bool                            `json:"cloud"`
	Selected     bool                            `json:"selected"`
	AutoMode     bool                            `json:"autoMode"`
	Availability proxy.ClaudeDesktopAvailability `json:"availability"`
	Reason       proxy.ClaudeDesktopAccessReason `json:"reason,omitempty"`
	RequiredPlan string                          `json:"requiredPlan,omitempty"`
}

type claudeDesktopActionResult struct {
	Status                      claudeDesktopStatus `json:"status"`
	Error                       string              `json:"error,omitempty"`
	MappingsApplied             bool                `json:"mappingsApplied,omitempty"`
	RestartConfirmationRequired bool                `json:"restartConfirmationRequired,omitempty"`
}
