package middleware

import (
	"context"

	"github.com/ollama/ollama/api"
)

const maxWebSearchLoops = 3

// doFollowUpChat sends a non-streaming /api/chat request with the accumulated
// messages and tools so the model can continue after a web search result.
func doFollowUpChat(ctx context.Context, base api.ChatRequest, messages []api.Message, tools api.Tools) (api.ChatResponse, error) {
	stream := false
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return api.ChatResponse{}, err
	}
	var chatResponse api.ChatResponse
	request := base
	request.Messages = messages
	request.Stream = &stream
	request.Tools = tools
	if err := client.Chat(ctx, &request, func(response api.ChatResponse) error {
		chatResponse = response
		return nil
	}); err != nil {
		return api.ChatResponse{}, err
	}
	return chatResponse, nil
}

// streamFollowUpChat streams the model response after a web search result.
func streamFollowUpChat(ctx context.Context, base api.ChatRequest, messages []api.Message, tools api.Tools, yield func(api.ChatResponse) error) error {
	stream := true
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	request := base
	request.Messages = messages
	request.Stream = &stream
	request.Tools = tools
	return client.Chat(ctx, &request, yield)
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
