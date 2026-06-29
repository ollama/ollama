package agent

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
)

func estimateChatRequestTokens(opts RunOptions, messages []api.Message, tools api.Tools) int {
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
