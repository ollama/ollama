//go:build darwin

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/proxy"
)

// Discovery runs on the HTTP server, not WebKit's native message callback.
func desktopModelSettingsHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()
		catalog := r.URL.Query().Get("catalog") == "true"
		var result any
		switch r.PathValue("integration") {
		case "claude-desktop":
			status := claudeDesktopSettingsSummary()
			if catalog && status.Installed && status.Used {
				status = claudeDesktopConnectionStatus(ctx)
			}
			result = status
		case "chatgpt":
			settings := codexDesktopSettingsSummary()
			response := codexDesktopModelsSettingsResult{Settings: settings}
			if catalog && settings.Installed {
				var err error
				response.Settings, err = codexDesktopSettingsWithInventory(ctx, settings)
				if err != nil {
					response.Warning = codexDesktopModelRefreshError(response.Settings)
				}
			}
			result = response
		default:
			http.NotFound(w, r)
			return
		}
		if ctx.Err() != nil {
			http.Error(w, "Model discovery timed out", http.StatusGatewayTimeout)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		json.NewEncoder(w).Encode(result)
	})
}

func claudeDesktopSettingsSummary() claudeDesktopStatus {
	status := getClaudeDesktopConnectionSummary()
	selected := activeClaudeDesktopModels()
	mappings := proxy.ClaudeDesktopMappings(selected)
	if saved := launch.ClaudeDesktopModelMappings(); len(saved) > 0 {
		mappings = saved
	}
	for _, route := range proxy.ClaudeDesktopRoutes() {
		status.Mappings = append(status.Mappings, claudeDesktopMappingStatus{
			RouteID: route.ID, RouteName: route.DisplayName, Model: mappings[route.ID],
		})
	}
	for _, model := range selected {
		name := model.Name
		if model.Cloud {
			name = model.OllamaModel
		}
		status.Models = append(status.Models, claudeDesktopModelStatus{
			Name: name, DisplayName: name, Cloud: model.Cloud, Selected: true,
			Availability: "unknown", AutoMode: claudeDesktopModelSupportsAutoMode(model),
		})
	}
	for _, mapping := range status.Mappings {
		name := mapping.Model
		known := false
		for _, model := range status.Models {
			known = known || model.Name == name
		}
		if name != "" && !known {
			status.Models = append(status.Models, claudeDesktopModelStatus{
				Name: name, DisplayName: name, Selected: true, Availability: "unknown",
			})
		}
	}
	autoMode, err := launch.ClaudeDesktopAutoModeEnabled()
	status.AutoMode = autoMode
	if err != nil && status.Error == "" {
		status.Error = err.Error()
	}
	return status
}
