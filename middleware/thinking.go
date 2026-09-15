package middleware

import (
	"github.com/ollama/ollama/types/model"
)

// ThinkingLookup identifies models whose generic renderer resolves named efforts.
// A nil result keeps the existing compatibility conversion for that model.
type ThinkingLookup func(string) *model.Thinking

func modelThinking(lookups []ThinkingLookup, name string) *model.Thinking {
	if len(lookups) == 0 || lookups[0] == nil {
		return nil
	}
	return lookups[0](name)
}
