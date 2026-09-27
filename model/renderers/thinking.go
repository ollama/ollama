package renderers

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// ThinkingForRenderer returns the controls of the selected renderer variant.
// Unknown or invalid descriptors are omitted.
func ThinkingForRenderer(name string) *model.Thinking {
	if name == "harmony" {
		return &model.Thinking{
			Values:  []any{"low", "medium", "high"},
			Default: "medium",
		}
	}
	r := rendererForName(name)
	if r == nil {
		return nil
	}
	thinking := r.Thinking()
	if !thinking.Valid() {
		return nil
	}
	return thinking.Clone()
}

// ResolveThinking preserves explicit booleans and supported names. Omission or
// an unsupported name uses the default. Unknown metadata preserves legacy handling.
func ResolveThinking(requestedThink *api.ThinkValue, thinking *model.Thinking) *api.ThinkValue {
	if !thinking.Valid() {
		return requestedThink
	}
	if requestedThink != nil {
		switch value := requestedThink.Value.(type) {
		case bool:
			return requestedThink
		case string:
			if thinking.Supports(value) {
				return requestedThink
			}
		case int:
			// A thinking-token budget asks for thinking and names no level;
			// the sampler enforces the budget, not the renderer.
			return thinkingOn(thinking)
		}
	}
	return &api.ThinkValue{Value: thinking.Default}
}

// thinkingOn is the control that turns thinking on without choosing a level:
// true when the model states it, else the default when that thinks, else the
// first stated value that thinks. A model that cannot think keeps its default.
func thinkingOn(thinking *model.Thinking) *api.ThinkValue {
	if thinking.Supports(true) {
		return &api.ThinkValue{Value: true}
	}
	if thinking.Default != false {
		return &api.ThinkValue{Value: thinking.Default}
	}
	for _, value := range thinking.Values {
		if value != false {
			return &api.ThinkValue{Value: value}
		}
	}
	return &api.ThinkValue{Value: thinking.Default}
}
