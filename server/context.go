package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

func (s *Server) projectedContextLength(model api.ListModelResponse) int {
	// This bucket remains the fallback when no runner-specific preflight estimate is available.
	projected := int(envconfig.ContextLength())
	if projected == 0 {
		projected = s.defaultNumCtx
	}
	if model.Details.ContextLength > 0 {
		projected = min(projected, model.Details.ContextLength)
	}
	return projected
}
