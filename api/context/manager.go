package context

import (
	"context"
	"fmt"
	"github.com/ollama/ollama/api/providers"
)

// ContextManager manages conversation context
type ContextManager struct {
	maxContextTokens    int
	warningThreshold    float64
	autoSummarize       bool
	summarizationModel  string
	summarizationPrompt string
}

// NewContextManager creates a new context manager
func NewContextManager(maxTokens int, warningThreshold float64, autoSummarize bool) *ContextManager {
	return &ContextManager{
		maxContextTokens: maxTokens,
		warningThreshold: warningThreshold,
		autoSummarize:    autoSummarize,
		summarizationModel: "claude-haiku-4-5",
		summarizationPrompt: `Summarize the following conversation concisely, preserving key information and context:

<conversation>
%s
</conversation>

Provide a clear, informative summary that can be used as context for continuing the conversation.`,
	}
}

// CheckContext checks if context is within limits
func (cm *ContextManager) CheckContext(messages []providers.Message, currentTokens int) (*ContextStatus, error) {
	status := &ContextStatus{
		CurrentTokens:   currentTokens,
		MaxTokens:       cm.maxContextTokens,
		UsagePercentage: float64(currentTokens) / float64(cm.maxContextTokens),
		NeedsAction:     false,
	}

	if status.UsagePercentage >= cm.warningThreshold {
		status.NeedsAction = true

		if status.UsagePercentage >= 0.95 {
			status.Action = "truncate"
		} else {
			status.Action = "warn"
		}
	}

	return status, nil
}

// SummarizeMessages summarizes old messages to free up context
func (cm *ContextManager) SummarizeMessages(ctx context.Context, messages []providers.Message, provider providers.Provider) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages to summarize")
	}

	var conversationText string
	for _, msg := range messages {
		conversationText += fmt.Sprintf("%s: %s\n\n", msg.Role, msg.Content)
	}

	prompt := fmt.Sprintf(cm.summarizationPrompt, conversationText)

	req := providers.ChatRequest{
		Model: cm.summarizationModel,
		Messages: []providers.Message{
			{Role: "user", Content: prompt},
		},
	}

	maxTokens := 1000
	req.MaxTokens = &maxTokens

	resp, err := provider.ChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	return resp.Message.Content, nil
}

// TruncateMessages truncates old messages
func (cm *ContextManager) TruncateMessages(messages []providers.Message, targetTokens int) []providers.Message {
	if len(messages) <= 2 {
		return messages
	}

	startIdx := 0
	if messages[0].Role == "system" {
		startIdx = 1
	}

	estimatedTokensPerMessage := 200
	messagesToKeep := targetTokens / estimatedTokensPerMessage

	if messagesToKeep >= len(messages) {
		return messages
	}

	if startIdx == 1 {
		keepFrom := len(messages) - messagesToKeep + 1
		if keepFrom < 1 {
			keepFrom = 1
		}
		return append([]providers.Message{messages[0]}, messages[keepFrom:]...)
	}

	keepFrom := len(messages) - messagesToKeep
	if keepFrom < 0 {
		keepFrom = 0
	}
	return messages[keepFrom:]
}

// ContextStatus represents current context status
type ContextStatus struct {
	CurrentTokens   int     `json:"current_tokens"`
	MaxTokens       int     `json:"max_tokens"`
	UsagePercentage float64 `json:"usage_percentage"`
	NeedsAction     bool    `json:"needs_action"`
	Action          string  `json:"action"`
}

// Legacy Manager struct for backwards compatibility
type Manager struct{}
func NewManager() *Manager { return &Manager{} }
