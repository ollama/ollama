package server

import (
	"context"
	"net/http"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

const (
	capabilityThinking = "thinking"
)

// TokenizeHandler counts tokens for the given input.
func (s *Server) TokenizeHandler(c *gin.Context) {
	var req api.TokenizeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	// Get model info
	m, err := GetModel(req.Model)
	if err != nil {
		switch {
		case c.GetBool("remote"):
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		}
		return
	}

	// Check if cloud model - use estimation
	isCloud := m.Config.RemoteHost != "" || m.Config.RemoteModel != ""
	caps := m.Capabilities()
	var hasThinking bool
	for _, c := range caps {
		if string(c) == capabilityThinking {
			hasThinking = true
			break
		}
	}

	// Determine input: prefer messages, fallback to prompt
	var input string
	if len(req.Messages) > 0 {
		// Render messages to get prompt text
		tmpl, err := renderPrompt(m, req.Messages, nil, nil)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		input = tmpl
	} else {
		input = req.Prompt
	}

	if input == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "messages or prompt is required"})
		return
	}

	inputTokens := len(input) / 4 // Default estimate
	var outputTokens int

	// For cloud models, use simple estimation
	if isCloud {
		// Cloud models: rough estimation
		outputTokens = inputTokens / 4
		resp := api.TokenizeResponse{
			Model:        req.Model,
			Tokens:       inputTokens,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
		}
		c.JSON(http.StatusOK, resp)
		return
	}

	// For thinking models, estimate higher output
	if hasThinking {
		outputTokens = inputTokens * 2 // Thinking models output more
	}

	// Schedule runner to get tokenizer
	runner, _, _, err := s.scheduleRunner(c.Request.Context(), req.Model, nil, nil, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	defer runner.Close()

	// Tokenize using the runner
	ctx := context.Background()
	tokenIDs, err := runner.Tokenize(ctx, input)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	inputTokens = len(tokenIDs)

	// Reduce output token estimate for efficiency
	if outputTokens == 0 {
		outputTokens = inputTokens / 4 // Conservative estimate
		if hasThinking {
			outputTokens = inputTokens / 2 // Thinking: more output
		}
	}

	resp := api.TokenizeResponse{
		Model:        req.Model,
		Tokens:       inputTokens,
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
	}

	c.JSON(http.StatusOK, resp)
}
