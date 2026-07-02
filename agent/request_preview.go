package agent

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

// ChatRequestPreview is the request body plus the estimated prompt tokens for it.
type ChatRequestPreview struct {
	Request      api.ChatRequest
	PromptTokens int
}

// BuildChatRequestPreview builds the chat request shape used for a run and its estimated prompt tokens.
func BuildChatRequestPreview(opts RunOptions, messages []api.Message, tools api.Tools) ChatRequestPreview {
	return ChatRequestPreview{
		Request:      buildChatRequest(opts, messages, tools),
		PromptTokens: EstimateChatRequestPromptTokens(opts, messages, tools),
	}
}

// EstimateChatRequestPromptTokens estimates the prompt tokens for a chat request before sending it.
func EstimateChatRequestPromptTokens(opts RunOptions, messages []api.Message, tools api.Tools) int {
	return estimateCompactionRequestTokens(CompactionRequest{
		SystemPrompt: opts.SystemPrompt,
		Messages:     sanitizeMessagesForRequest(messages),
		Tools:        tools,
		Format:       opts.Format,
		Options:      opts.Options,
	})
}

func buildChatRequest(opts RunOptions, messages []api.Message, tools api.Tools) api.ChatRequest {
	requestMessages := sanitizeMessagesForRequest(messages)
	if strings.TrimSpace(opts.SystemPrompt) != "" {
		withSystem := make([]api.Message, 0, len(requestMessages)+1)
		withSystem = append(withSystem, api.Message{Role: "system", Content: opts.SystemPrompt})
		requestMessages = append(withSystem, requestMessages...)
	}

	req := api.ChatRequest{
		Model:    opts.Model,
		Messages: requestMessages,
		Format:   json.RawMessage(chatRequestFormat(opts.Format)),
		Options:  opts.Options,
		Think:    opts.Think,
	}
	if opts.KeepAlive != nil {
		req.KeepAlive = opts.KeepAlive
	}
	if len(tools) > 0 {
		req.Tools = tools
	}
	return req
}

func chatRequestFormat(format string) string {
	if format == "json" {
		return `"` + format + `"`
	}
	return format
}
