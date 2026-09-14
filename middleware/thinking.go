package middleware

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// ThinkingLookup identifies models whose generic renderer resolves named efforts.
// A nil result keeps the existing compatibility conversion for that model.
type ThinkingLookup func(string) *model.Thinking

func modelThinking(lookups []ThinkingLookup, name string) bool {
	return len(lookups) > 0 && lookups[0] != nil && lookups[0](name).Valid()
}

func requestedThinking(effort string) *api.ThinkValue {
	switch effort {
	case "":
		return nil
	case "none":
		return &api.ThinkValue{Value: false}
	default:
		return &api.ThinkValue{Value: effort}
	}
}
