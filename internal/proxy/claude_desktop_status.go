package proxy

import (
	"encoding/json"
	"net/http"
)

type claudeRuntimeModel struct {
	Model            ClaudeDesktopModel `json:"model"`
	Route            string             `json:"route"`
	EntitlementKnown bool               `json:"entitlement_known"`
}

type claudeDesktopRuntimeStatus struct {
	Models []claudeRuntimeModel `json:"models"`
	Counts ClaudeDesktopCounts  `json:"counts"`
}

func (p *ClaudeDesktop) serveRuntimeStatus(w http.ResponseWriter, models []ClaudeDesktopModel) {
	status := claudeDesktopRuntimeStatus{Counts: p.Counts()}
	for _, model := range models {
		status.Models = append(status.Models, claudeRuntimeModel{model, model.GatewayID(), model.entitlementKnown})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(status)
}
