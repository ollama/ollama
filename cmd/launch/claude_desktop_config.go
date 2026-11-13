package launch

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"

	"github.com/ollama/ollama/internal/proxy"
)

// ClaudeDesktopMapping is one editable Claude model slot.
type ClaudeDesktopMapping struct {
	ID           string
	Name         string
	Model        string
	DefaultModel string
}

// ClaudeDesktopMappingChoice selects a slot to edit or reset to its default.
// An empty RouteID saves the draft and launches Claude; ErrCancelled discards it.
type ClaudeDesktopMappingChoice struct {
	RouteID string
	Clear   bool
}

var DefaultClaudeDesktopMappingSelector func([]ClaudeDesktopMapping, string) (ClaudeDesktopMappingChoice, error)

func (c *launcherClient) configureClaudeDesktopMappings(ctx context.Context, candidate, defaults map[string]string) (map[string]string, error) {
	if !isInteractiveSession() {
		return nil, errors.New("editing Claude Desktop mappings requires an interactive terminal; use --model <model> --config to change the primary mapping without launching")
	}
	if DefaultClaudeDesktopMappingSelector == nil {
		return nil, errors.New("Claude Desktop mapping selector is unavailable")
	}
	draft := maps.Clone(candidate)
	if draft == nil {
		draft = make(map[string]string)
	}
	current := ""
	for {
		if err := claudeDesktopContextError(ctx); err != nil {
			return nil, err
		}
		var rows []ClaudeDesktopMapping
		for _, route := range proxy.ClaudeDesktopRoutes() {
			rows = append(rows, ClaudeDesktopMapping{ID: route.ID, Name: route.DisplayName, Model: draft[route.ID], DefaultModel: defaults[route.ID]})
		}
		choice, err := DefaultClaudeDesktopMappingSelector(rows, current)
		if err != nil {
			return nil, err
		}
		if err := claudeDesktopContextError(ctx); err != nil {
			return nil, err
		}
		if choice.RouteID == "" {
			effective := maps.Clone(draft)
			for _, route := range proxy.ClaudeDesktopRoutes() {
				if strings.TrimSpace(effective[route.ID]) == "" {
					if defaults[route.ID] != "" {
						effective[route.ID] = defaults[route.ID]
					} else {
						delete(effective, route.ID)
					}
				}
			}
			models := claudeDesktopMappingModels(effective)
			if len(models) == 0 {
				return nil, errors.New("map at least one Claude Desktop model before saving")
			}
			return effective, nil
		}
		var selected *ClaudeDesktopMapping
		for i := range rows {
			if rows[i].ID == choice.RouteID {
				selected = &rows[i]
				break
			}
		}
		if selected == nil {
			return nil, fmt.Errorf("unknown Claude Desktop route %q", choice.RouteID)
		}
		current = selected.ID
		if choice.Clear {
			delete(draft, selected.ID)
			continue
		}
		currentModel := selected.Model
		if currentModel == "" {
			currentModel = selected.DefaultModel
		}
		model, err := c.selectSingleModelWithSelectorReady(ctx, "Select Ollama model for "+selected.Name+":", currentModel, DefaultSingleSelector, true, "Claude Desktop", claudeDesktopIntegrationName)
		if errors.Is(err, ErrCancelled) {
			continue
		}
		if err != nil {
			return nil, err
		}
		draft[selected.ID] = model
	}
}

func claudeDesktopMappingModels(mappings map[string]string) []string {
	var models []string
	for _, route := range proxy.ClaudeDesktopRoutes() {
		if model := strings.TrimSpace(mappings[route.ID]); model != "" {
			models = append(models, model)
		}
	}
	return models
}
