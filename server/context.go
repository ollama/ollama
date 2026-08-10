package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

func (s *Server) recommendedContextLength(model api.ListModelResponse) int {
	// This bucket remains the fallback when no runner-specific preflight estimate is available.
	recommended := int(envconfig.ContextLength())
	if recommended == 0 {
		recommended = s.defaultNumCtx
	}
	if model.Details.ContextLength > 0 {
		recommended = min(recommended, model.Details.ContextLength)
	}
	return recommended
}
