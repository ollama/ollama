package middleware

import (
	"context"

	"github.com/ollama/ollama/api"
)

const maxWebSearchLoops = 3

// doFollowUpChat sends a non-streaming /api/chat request with the accumulated
// messages and tools so the model can continue after a web search result.
func doFollowUpChat(ctx context.Context, model string, messages []api.Message, tools api.Tools, options map[string]any) (api.ChatResponse, error) {
	stream := false
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return api.ChatResponse{}, err
	}
	var chatResponse api.ChatResponse
	request := api.ChatRequest{Model: model, Messages: messages, Stream: &stream, Tools: tools, Options: options}
	if err := client.Chat(ctx, &request, func(response api.ChatResponse) error {
		chatResponse = response
		return nil
	}); err != nil {
		return api.ChatResponse{}, err
	}
	return chatResponse, nil
}

func buildWebSearchAssistantMessage(response api.ChatResponse, webSearchCall api.ToolCall) api.Message {
	assistant := api.Message{
		Role:      "assistant",
		ToolCalls: []api.ToolCall{webSearchCall},
	}
	assistant.Content = response.Message.Content
	assistant.Thinking = response.Message.Thinking
	return assistant
}

func findWebSearchToolCall(toolCalls []api.ToolCall) (api.ToolCall, bool, bool) {
	var webSearchCall api.ToolCall
	var hasWebSearch, hasOtherTools bool

	for _, toolCall := range toolCalls {
		if toolCall.Function.Name == "web_search" {
			if !hasWebSearch {
				webSearchCall = toolCall
				hasWebSearch = true
			}
			continue
		}
		hasOtherTools = true
	}

	return webSearchCall, hasWebSearch, hasOtherTools
}

func extractQueryFromToolCall(toolCall *api.ToolCall) string {
	query, ok := toolCall.Function.Arguments.Get("query")
	if !ok {
		return ""
	}
	value, _ := query.(string)
	return value
}
