package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

const (
	testImage = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`
)

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests)
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

func TestFromMessagesRequest_Basic(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", result.Model)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "user" || result.Messages[0].Content != "Hello" {
		t.Errorf("unexpected message: %+v", result.Messages[0])
	}

	if numPredict, ok := result.Options["num_predict"].(int); !ok || numPredict != 1024 {
		t.Errorf("expected num_predict 1024, got %v", result.Options["num_predict"])
	}
}

func TestFromMessagesRequest_WithSystemPrompt(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		System:    "You are a helpful assistant.",
		Messages: []MessageParam{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "system" || result.Messages[0].Content != "You are a helpful assistant." {
		t.Errorf("unexpected system message: %+v", result.Messages[0])
	}
}

func TestFromMessagesRequest_WithSystemPromptArray(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		System: []any{
			map[string]any{"type": "text", "text": "You are helpful."},
			map[string]any{"type": "text", "text": " Be concise."},
		},
		Messages: []MessageParam{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result.Messages))
	}

	if result.Messages[0].Content != "You are helpful. Be concise." {
		t.Errorf("unexpected system message content: %q", result.Messages[0].Content)
	}
}

func TestFromMessagesRequest_WithOptions(t *testing.T) {
	temp := 0.7
	topP := 0.9
	topK := 40
	req := MessagesRequest{
		Model:         "test-model",
		MaxTokens:     2048,
		Messages:      []MessageParam{{Role: "user", Content: "Hello"}},
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"\n", "END"},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Options["temperature"] != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", result.Options["temperature"])
	}
	if result.Options["top_p"] != 0.9 {
		t.Errorf("expected top_p 0.9, got %v", result.Options["top_p"])
	}
	if result.Options["top_k"] != 40 {
		t.Errorf("expected top_k 40, got %v", result.Options["top_k"])
	}
	if diff := cmp.Diff([]string{"\n", "END"}, result.Options["stop"]); diff != "" {
		t.Errorf("stop sequences mismatch: %s", diff)
	}
}

func TestFromMessagesRequest_WithImage(t *testing.T) {
	imgData, _ := base64.StdEncoding.DecodeString(testImage)

	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{
				Role: "user",
				Content: []any{
					map[string]any{"type": "text", "text": "What's in this image?"},
					map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": "image/png",
							"data":       testImage,
						},
					},
				},
			},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Content != "What's in this image?" {
		t.Errorf("expected content 'What's in this image?', got %q", result.Messages[0].Content)
	}

	if len(result.Messages[0].Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(result.Messages[0].Images))
	}

	if string(result.Messages[0].Images[0]) != string(imgData) {
		t.Error("image data mismatch")
	}
}

func TestFromMessagesRequest_WithToolUse(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{Role: "user", Content: "What's the weather in Paris?"},
			{
				Role: "assistant",
				Content: []any{
					map[string]any{
						"type":  "tool_use",
						"id":    "call_123",
						"name":  "get_weather",
						"input": map[string]any{"location": "Paris"},
					},
				},
			},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result.Messages))
	}

	if len(result.Messages[1].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.Messages[1].ToolCalls))
	}

	tc := result.Messages[1].ToolCalls[0]
	if tc.ID != "call_123" {
		t.Errorf("expected tool call ID 'call_123', got %q", tc.ID)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("expected tool name 'get_weather', got %q", tc.Function.Name)
	}
}

func TestFromMessagesRequest_WithToolResult(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{
				Role: "user",
				Content: []any{
					map[string]any{
						"type":        "tool_result",
						"tool_use_id": "call_123",
						"content":     "The weather in Paris is sunny, 22°C",
					},
				},
			},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}

	msg := result.Messages[0]
	if msg.Role != "tool" {
		t.Errorf("expected role 'tool', got %q", msg.Role)
	}
	if msg.ToolCallID != "call_123" {
		t.Errorf("expected tool_call_id 'call_123', got %q", msg.ToolCallID)
	}
	if msg.Content != "The weather in Paris is sunny, 22°C" {
		t.Errorf("unexpected content: %q", msg.Content)
	}
}

func TestFromMessagesRequest_WithTools(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages:  []MessageParam{{Role: "user", Content: "Hello"}},
		Tools: []Tool{
			{
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
			},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}

	tool := result.Tools[0]
	if tool.Type != "function" {
		t.Errorf("expected type 'function', got %q", tool.Type)
	}
	if tool.Function.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", tool.Function.Name)
	}
	if tool.Function.Description != "Get current weather" {
		t.Errorf("expected description 'Get current weather', got %q", tool.Function.Description)
	}
}

func TestFromMessagesRequest_DropsCustomWebSearchWhenBuiltinPresent(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages:  []MessageParam{{Role: "user", Content: "Hello"}},
		Tools: []Tool{
			{
				Type: "web_search_20250305",
				Name: "web_search",
			},
			{
				Type:        "custom",
				Name:        "web_search",
				Description: "User-defined web search that should be dropped",
				InputSchema: json.RawMessage(`{"type":"invalid"}`),
			},
			{
				Type:        "custom",
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
			},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Tools) != 2 {
		t.Fatalf("expected 2 tools after dropping custom web_search, got %d", len(result.Tools))
	}
	if result.Tools[0].Function.Name != "web_search" {
		t.Fatalf("expected first tool to be built-in web_search, got %q", result.Tools[0].Function.Name)
	}
	if result.Tools[1].Function.Name != "get_weather" {
		t.Fatalf("expected second tool to be get_weather, got %q", result.Tools[1].Function.Name)
	}
}

func TestFromMessagesRequest_KeepsCustomWebSearchWhenBuiltinAbsent(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages:  []MessageParam{{Role: "user", Content: "Hello"}},
		Tools: []Tool{
			{
				Type:        "custom",
				Name:        "web_search",
				Description: "User-defined web search",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`),
			},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 custom tool, got %d", len(result.Tools))
	}
	if result.Tools[0].Function.Name != "web_search" {
		t.Fatalf("expected custom tool name web_search, got %q", result.Tools[0].Function.Name)
	}
	if result.Tools[0].Function.Description != "User-defined web search" {
		t.Fatalf("expected custom description preserved, got %q", result.Tools[0].Function.Description)
	}
}

func TestFromMessagesRequest_WithThinking(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages:  []MessageParam{{Role: "user", Content: "Hello"}},
		Thinking:  &ThinkingConfig{Type: "enabled", BudgetTokens: 1000},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Think == nil {
		t.Fatal("expected Think to be set")
	}
	if v, ok := result.Think.Value.(bool); !ok || !v {
		t.Errorf("expected Think.Value to be true, got %v", result.Think.Value)
	}
}

func TestFromMessagesRequest_ThinkingOnlyBlock(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{Role: "user", Content: "Hello"},
			{
				Role: "assistant",
				Content: []any{
					map[string]any{
						"type":     "thinking",
						"thinking": "Let me think about this...",
					},
				},
			},
		},
	}

	result, err := FromMessagesRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result.Messages))
	}

	assistantMsg := result.Messages[1]
	if assistantMsg.Thinking != "Let me think about this..." {
		t.Errorf("expected thinking content, got %q", assistantMsg.Thinking)
	}
}

func TestFromMessagesRequest_ToolUseMissingID(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{
				Role: "assistant",
				Content: []any{
					map[string]any{
						"type": "tool_use",
						"name": "get_weather",
					},
				},
			},
		},
	}

	_, err := FromMessagesRequest(req)
	if err == nil {
		t.Fatal("expected error for missing tool_use id")
	}
	if err.Error() != "tool_use block missing required 'id' field" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestFromMessagesRequest_ToolUseMissingName(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{
				Role: "assistant",
				Content: []any{
					map[string]any{
						"type": "tool_use",
						"id":   "call_123",
					},
				},
			},
		},
	}

	_, err := FromMessagesRequest(req)
	if err == nil {
		t.Fatal("expected error for missing tool_use name")
	}
	if err.Error() != "tool_use block missing required 'name' field" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestFromMessagesRequest_InvalidToolSchema(t *testing.T) {
	req := MessagesRequest{
		Model:     "test-model",
		MaxTokens: 1024,
		Messages:  []MessageParam{{Role: "user", Content: "Hello"}},
		Tools: []Tool{
			{
				Name:        "bad_tool",
				InputSchema: json.RawMessage(`{invalid json`),
			},
		},
	}

	_, err := FromMessagesRequest(req)
	if err == nil {
		t.Fatal("expected error for invalid tool schema")
	}
}

func TestToMessagesResponse_Basic(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role:    "assistant",
			Content: "Hello there!",
		},
		Done:       true,
		DoneReason: "stop",
		Metrics: api.Metrics{
			PromptEvalCount: 10,
			EvalCount:       5,
		},
	}

	result := ToMessagesResponse("msg_123", resp)

	if result.ID != "msg_123" {
		t.Errorf("expected ID 'msg_123', got %q", result.ID)
	}
	if result.Type != "message" {
		t.Errorf("expected type 'message', got %q", result.Type)
	}
	if result.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", result.Role)
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Type != "text" || result.Content[0].Text == nil || *result.Content[0].Text != "Hello there!" {
		t.Errorf("unexpected content: %+v", result.Content[0])
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", result.StopReason)
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Errorf("unexpected usage: %+v", result.Usage)
	}
}

func TestToMessagesResponse_WithToolCalls(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_123",
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "Paris"}),
					},
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
	}

	result := ToMessagesResponse("msg_123", resp)

	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Type != "tool_use" {
		t.Errorf("expected type 'tool_use', got %q", result.Content[0].Type)
	}
	if result.Content[0].ID != "call_123" {
		t.Errorf("expected ID 'call_123', got %q", result.Content[0].ID)
	}
	if result.Content[0].Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", result.Content[0].Name)
	}
	if result.StopReason != "tool_use" {
		t.Errorf("expected stop_reason 'tool_use', got %q", result.StopReason)
	}
}

func TestToMessagesResponse_WithThinking(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role:     "assistant",
			Content:  "The answer is 42.",
			Thinking: "Let me think about this...",
		},
		Done:       true,
		DoneReason: "stop",
	}

	result := ToMessagesResponse("msg_123", resp)

	if len(result.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(result.Content))
	}
	if result.Content[0].Type != "thinking" {
		t.Errorf("expected first block type 'thinking', got %q", result.Content[0].Type)
	}
	if result.Content[0].Thinking == nil || *result.Content[0].Thinking != "Let me think about this..." {
		t.Errorf("unexpected thinking content: %v", result.Content[0].Thinking)
	}
	if result.Content[1].Type != "text" {
		t.Errorf("expected second block type 'text', got %q", result.Content[1].Type)
	}
}

func TestMapStopReason(t *testing.T) {
	tests := []struct {
		reason       string
		hasToolCalls bool
		want         string
	}{
		{"stop", false, "end_turn"},
		{"length", false, "max_tokens"},
		{"stop", true, "tool_use"},
		{"other", false, "stop_sequence"},
		{"", false, ""},
	}

	for _, tt := range tests {
		got := mapStopReason(tt.reason, tt.hasToolCalls)
		if got != tt.want {
			t.Errorf("mapStopReason(%q, %v) = %q, want %q", tt.reason, tt.hasToolCalls, got, tt.want)
		}
	}
}

func TestNewError(t *testing.T) {
	tests := []struct {
		code int
		want string
	}{
		{400, "invalid_request_error"},
		{401, "authentication_error"},
		{403, "permission_error"},
		{404, "not_found_error"},
		{429, "rate_limit_error"},
		{500, "api_error"},
		{503, "overloaded_error"},
		{529, "overloaded_error"},
	}

	for _, tt := range tests {
		result := NewError(tt.code, "test message")
		if result.Type != "error" {
			t.Errorf("NewError(%d) type = %q, want 'error'", tt.code, result.Type)
		}
		if result.Error.Type != tt.want {
			t.Errorf("NewError(%d) error.type = %q, want %q", tt.code, result.Error.Type, tt.want)
		}
		if result.Error.Message != "test message" {
			t.Errorf("NewError(%d) message = %q, want 'test message'", tt.code, result.Error.Message)
		}
		if result.RequestID == "" {
			t.Errorf("NewError(%d) request_id should not be empty", tt.code)
		}
	}
}

func TestGenerateMessageID(t *testing.T) {
	id1 := GenerateMessageID()
	id2 := GenerateMessageID()

	if id1 == "" {
		t.Error("GenerateMessageID returned empty string")
	}
	if id1 == id2 {
		t.Error("GenerateMessageID returned duplicate IDs")
	}
	if len(id1) < 10 {
		t.Errorf("GenerateMessageID returned short ID: %q", id1)
	}
	if id1[:4] != "msg_" {
		t.Errorf("GenerateMessageID should start with 'msg_', got %q", id1[:4])
	}
}

func TestStreamConverter_Basic(t *testing.T) {
	conv := NewStreamConverter("msg_123", "test-model", 0)

	// First chunk
	resp1 := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role:    "assistant",
			Content: "Hello",
		},
		Metrics: api.Metrics{PromptEvalCount: 10},
	}

	events1 := conv.Process(resp1)
	if len(events1) < 3 {
		t.Fatalf("expected at least 3 events for first chunk, got %d", len(events1))
	}

	// Should have message_start, content_block_start, content_block_delta
	if events1[0].Event != "message_start" {
		t.Errorf("expected first event 'message_start', got %q", events1[0].Event)
	}
	if events1[1].Event != "content_block_start" {
		t.Errorf("expected second event 'content_block_start', got %q", events1[1].Event)
	}
	if events1[2].Event != "content_block_delta" {
		t.Errorf("expected third event 'content_block_delta', got %q", events1[2].Event)
	}

	// Final chunk
	resp2 := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role:    "assistant",
			Content: " world!",
		},
		Done:       true,
		DoneReason: "stop",
		Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
	}

	events2 := conv.Process(resp2)

	// Should have content_block_delta, content_block_stop, message_delta, message_stop
	hasStop := false
	for _, e := range events2 {
		if e.Event == "message_delta" {
			if data, ok := e.Data.(MessageDeltaEvent); ok {
				if data.Type != "message_delta" {
					t.Errorf("unexpected data type: %+v", data)
				}

				if data.Delta.StopReason != "end_turn" {
					t.Errorf("unexpected stop reason: %+v", data.Delta.StopReason)
				}

				if data.Usage.InputTokens != 10 || data.Usage.OutputTokens != 5 {
					t.Errorf("unexpected usage: %+v", data.Usage)
				}
			} else {
				t.Errorf("unexpected data: %+v", e.Data)
			}
		}

		if e.Event == "message_stop" {
			hasStop = true
		}
	}
	if !hasStop {
		t.Error("expected message_stop event in final chunk")
	}
}

func TestStreamConverter_WithToolCalls(t *testing.T) {
	conv := NewStreamConverter("msg_123", "test-model", 0)

	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_123",
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"location": "Paris"}),
					},
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
		Metrics:    api.Metrics{PromptEvalCount: 10, EvalCount: 5},
	}

	events := conv.Process(resp)

	hasToolStart := false
	hasToolDelta := false
	for _, e := range events {
		if e.Event == "content_block_start" {
			if start, ok := e.Data.(ContentBlockStartEvent); ok {
				if start.ContentBlock.Type == "tool_use" {
					hasToolStart = true
				}
			}
		}
		if e.Event == "content_block_delta" {
			if delta, ok := e.Data.(ContentBlockDeltaEvent); ok {
				if delta.Delta.Type == "input_json_delta" {
					hasToolDelta = true
				}
			}
		}
	}

	if !hasToolStart {
		t.Error("expected tool_use content_block_start event")
	}
	if !hasToolDelta {
		t.Error("expected input_json_delta event")
	}
}

func TestStreamConverter_ToolCallWithUnmarshalableArgs(t *testing.T) {
	// Test that unmarshalable arguments (like channels) are handled gracefully
	// and don't cause a panic or corrupt stream
	conv := NewStreamConverter("msg_123", "test-model", 0)

	// Create a channel which cannot be JSON marshaled
	unmarshalable := make(chan int)
	badArgs := api.NewToolCallFunctionArguments()
	badArgs.Set("channel", unmarshalable)

	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_bad",
					Function: api.ToolCallFunction{
						Name:      "bad_function",
						Arguments: badArgs,
					},
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
	}

	// Should not panic and should skip the unmarshalable tool call
	events := conv.Process(resp)

	// Verify no tool_use block was started (since marshal failed before block start)
	hasToolStart := false
	for _, e := range events {
		if e.Event == "content_block_start" {
			if start, ok := e.Data.(ContentBlockStartEvent); ok {
				if start.ContentBlock.Type == "tool_use" {
					hasToolStart = true
				}
			}
		}
	}

	if hasToolStart {
		t.Error("expected no tool_use block when arguments cannot be marshaled")
	}
}

func TestStreamConverter_MultipleToolCallsWithMixedValidity(t *testing.T) {
	// Test that valid tool calls still work when mixed with invalid ones
	conv := NewStreamConverter("msg_123", "test-model", 0)

	unmarshalable := make(chan int)
	badArgs := api.NewToolCallFunctionArguments()
	badArgs.Set("channel", unmarshalable)

	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_good",
					Function: api.ToolCallFunction{
						Name:      "good_function",
						Arguments: testArgs(map[string]any{"location": "Paris"}),
					},
				},
				{
					ID: "call_bad",
					Function: api.ToolCallFunction{
						Name:      "bad_function",
						Arguments: badArgs,
					},
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
	}

	events := conv.Process(resp)

	// Count tool_use blocks - should only have 1 (the valid one)
	toolStartCount := 0
	toolDeltaCount := 0
	for _, e := range events {
		if e.Event == "content_block_start" {
			if start, ok := e.Data.(ContentBlockStartEvent); ok {
				if start.ContentBlock.Type == "tool_use" {
					toolStartCount++
					if start.ContentBlock.Name != "good_function" {
						t.Errorf("expected tool name 'good_function', got %q", start.ContentBlock.Name)
					}
				}
			}
		}
		if e.Event == "content_block_delta" {
			if delta, ok := e.Data.(ContentBlockDeltaEvent); ok {
				if delta.Delta.Type == "input_json_delta" {
					toolDeltaCount++
				}
			}
		}
	}

	if toolStartCount != 1 {
		t.Errorf("expected 1 tool_use block, got %d", toolStartCount)
	}
	if toolDeltaCount != 1 {
		t.Errorf("expected 1 input_json_delta, got %d", toolDeltaCount)
	}
}

func TestContentBlockJSON_EmptyFieldsPresent(t *testing.T) {
	tests := []struct {
		name     string
		block    ContentBlock
		wantKeys []string
	}{
		{
			name: "text block includes empty text field",
			block: ContentBlock{
				Type: "text",
				Text: ptr(""),
			},
			wantKeys: []string{"type", "text"},
		},
		{
			name: "thinking block includes empty thinking field",
			block: ContentBlock{
				Type:     "thinking",
				Thinking: ptr(""),
			},
			wantKeys: []string{"type", "thinking"},
		},
		{
			name: "text block with content",
			block: ContentBlock{
				Type: "text",
				Text: ptr("hello"),
			},
			wantKeys: []string{"type", "text"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.block)
			if err != nil {
				t.Fatalf("failed to marshal: %v", err)
			}

			var result map[string]any
			if err := json.Unmarshal(data, &result); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			for _, key := range tt.wantKeys {
				if _, ok := result[key]; !ok {
					t.Errorf("expected key %q to be present in JSON output, got: %s", key, string(data))
				}
			}
		})
	}
}

func TestStreamConverter_ContentBlockStartIncludesEmptyFields(t *testing.T) {
	t.Run("text block start includes empty text", func(t *testing.T) {
		conv := NewStreamConverter("msg_123", "test-model", 0)

		resp := api.ChatResponse{
			Model:   "test-model",
			Message: api.Message{Role: "assistant", Content: "hello"},
		}

		events := conv.Process(resp)

		var foundTextStart bool
		for _, e := range events {
			if e.Event == "content_block_start" {
				if start, ok := e.Data.(ContentBlockStartEvent); ok {
					if start.ContentBlock.Type == "text" {
						foundTextStart = true
						// Marshal and verify the text field is present
						data, _ := json.Marshal(start)
						var result map[string]any
						json.Unmarshal(data, &result)
						cb := result["content_block"].(map[string]any)
						if _, ok := cb["text"]; !ok {
							t.Error("content_block_start for text should include 'text' field")
						}
					}
				}
			}
		}

		if !foundTextStart {
			t.Error("expected text content_block_start event")
		}
	})

	t.Run("thinking block start includes empty thinking", func(t *testing.T) {
		conv := NewStreamConverter("msg_123", "test-model", 0)

		resp := api.ChatResponse{
			Model:   "test-model",
			Message: api.Message{Role: "assistant", Thinking: "let me think..."},
		}

		events := conv.Process(resp)

		var foundThinkingStart bool
		for _, e := range events {
			if e.Event == "content_block_start" {
				if start, ok := e.Data.(ContentBlockStartEvent); ok {
					if start.ContentBlock.Type == "thinking" {
						foundThinkingStart = true
						data, _ := json.Marshal(start)
						var result map[string]any
						json.Unmarshal(data, &result)
						cb := result["content_block"].(map[string]any)
						if _, ok := cb["thinking"]; !ok {
							t.Error("content_block_start for thinking should include 'thinking' field")
						}
					}
				}
			}
		}

		if !foundThinkingStart {
			t.Error("expected thinking content_block_start event")
		}
	})
}

func TestEstimateTokens_SimpleMessage(t *testing.T) {
	req := CountTokensRequest{
		Model: "test-model",
		Messages: []MessageParam{
			{Role: "user", Content: "Hello, world!"},
		},
	}

	tokens := estimateTokens(req)

	// "user" (4) + "Hello, world!" (13) = 17 chars / 4 = 4 tokens
	if tokens < 1 {
		t.Errorf("expected at least 1 token, got %d", tokens)
	}
	// Sanity check: shouldn't be wildly off
	if tokens > 10 {
		t.Errorf("expected fewer than 10 tokens for short message, got %d", tokens)
	}
}

func TestEstimateTokens_WithSystemPrompt(t *testing.T) {
	req := CountTokensRequest{
		Model:  "test-model",
		System: "You are a helpful assistant.",
		Messages: []MessageParam{
			{Role: "user", Content: "Hello"},
		},
	}

	tokens := estimateTokens(req)

	// System prompt adds to count
	if tokens < 5 {
		t.Errorf("expected at least 5 tokens with system prompt, got %d", tokens)
	}
}

func TestEstimateTokens_WithTools(t *testing.T) {
	req := CountTokensRequest{
		Model: "test-model",
		Messages: []MessageParam{
			{Role: "user", Content: "What's the weather?"},
		},
		Tools: []Tool{
			{
				Name:        "get_weather",
				Description: "Get the current weather for a location",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
			},
		},
	}

	tokens := estimateTokens(req)

	// Tools add significant content
	if tokens < 10 {
		t.Errorf("expected at least 10 tokens with tools, got %d", tokens)
	}
}

func TestEstimateTokens_WithThinking(t *testing.T) {
	req := CountTokensRequest{
		Model: "test-model",
		Messages: []MessageParam{
			{Role: "user", Content: "Hello"},
			{
				Role: "assistant",
				Content: []any{
					map[string]any{
						"type":     "thinking",
						"thinking": "Let me think about this carefully...",
					},
					map[string]any{
						"type": "text",
						"text": "Here is my response.",
					},
				},
			},
		},
	}

	tokens := estimateTokens(req)

	// Thinking content should be counted
	if tokens < 10 {
		t.Errorf("expected at least 10 tokens with thinking content, got %d", tokens)
	}
}

func TestEstimateTokens_EmptyContent(t *testing.T) {
	req := CountTokensRequest{
		Model:    "test-model",
		Messages: []MessageParam{},
	}

	tokens := estimateTokens(req)

	if tokens != 0 {
		t.Errorf("expected 0 tokens for empty content, got %d", tokens)
	}
}

// Web Search Tests

func TestConvertTool_WebSearch(t *testing.T) {
	tool := Tool{
		Type:    "web_search_20250305",
		Name:    "web_search",
		MaxUses: 5,
	}

	result, isServerTool, err := convertTool(tool)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !isServerTool {
		t.Error("expected isServerTool to be true for web_search tool")
	}

	if result.Type != "function" {
		t.Errorf("expected type 'function', got %q", result.Type)
	}

	if result.Function.Name != "web_search" {
		t.Errorf("expected name 'web_search', got %q", result.Function.Name)
	}

	if result.Function.Description == "" {
		t.Error("expected non-empty description for web_search tool")
	}

	// Check that query parameter is defined
	if result.Function.Parameters.Properties == nil {
		t.Fatal("expected properties to be defined")
	}

	queryProp, ok := result.Function.Parameters.Properties.Get("query")
	if !ok {
		t.Error("expected 'query' property to be defined")
	}

	if len(queryProp.Type) == 0 || queryProp.Type[0] != "string" {
		t.Errorf("expected query type to be 'string', got %v", queryProp.Type)
	}
}

func TestConvertTool_RegularTool(t *testing.T) {
	tool := Tool{
		Type:        "custom",
		Name:        "get_weather",
		Description: "Get the weather",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
	}

	result, isServerTool, err := convertTool(tool)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if isServerTool {
		t.Error("expected isServerTool to be false for regular tool")
	}

	if result.Function.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", result.Function.Name)
	}
}

func TestConvertMessage_ServerToolUse(t *testing.T) {
	msg := MessageParam{
		Role: "assistant",
		Content: []any{
			map[string]any{
				"type":  "server_tool_use",
				"id":    "srvtoolu_123",
				"name":  "web_search",
				"input": map[string]any{"query": "test query"},
			},
		},
	}

	messages, err := convertMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if len(messages[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(messages[0].ToolCalls))
	}

	tc := messages[0].ToolCalls[0]
	if tc.ID != "srvtoolu_123" {
		t.Errorf("expected tool call ID 'srvtoolu_123', got %q", tc.ID)
	}

	if tc.Function.Name != "web_search" {
		t.Errorf("expected tool name 'web_search', got %q", tc.Function.Name)
	}
}

func TestConvertMessage_WebSearchToolResult(t *testing.T) {
	msg := MessageParam{
		Role: "user",
		Content: []any{
			map[string]any{
				"type":        "web_search_tool_result",
				"tool_use_id": "srvtoolu_123",
				"content": []any{
					map[string]any{
						"type":  "web_search_result",
						"title": "Test Result",
						"url":   "https://example.com",
					},
				},
			},
		},
	}

	messages, err := convertMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have a tool result message
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}

	if messages[0].Role != "tool" {
		t.Errorf("expected role 'tool', got %q", messages[0].Role)
	}

	if messages[0].ToolCallID != "srvtoolu_123" {
		t.Errorf("expected tool_call_id 'srvtoolu_123', got %q", messages[0].ToolCallID)
	}

	if messages[0].Content == "" {
		t.Error("expected non-empty content from web search results")
	}
}

func TestConvertMessage_WebSearchToolResultEmptyStillCreatesToolMessage(t *testing.T) {
	msg := MessageParam{
		Role: "user",
		Content: []any{
			map[string]any{
				"type":        "web_search_tool_result",
				"tool_use_id": "srvtoolu_empty",
				"content":     []any{},
			},
		},
	}

	messages, err := convertMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != "tool" {
		t.Fatalf("expected role tool, got %q", messages[0].Role)
	}
	if messages[0].ToolCallID != "srvtoolu_empty" {
		t.Fatalf("expected tool_call_id srvtoolu_empty, got %q", messages[0].ToolCallID)
	}
	if messages[0].Content != "" {
		t.Fatalf("expected empty content for empty web search results, got %q", messages[0].Content)
	}
}

func TestConvertMessage_WebSearchToolResultErrorStillCreatesToolMessage(t *testing.T) {
	msg := MessageParam{
		Role: "user",
		Content: []any{
			map[string]any{
				"type":        "web_search_tool_result",
				"tool_use_id": "srvtoolu_error",
				"content": map[string]any{
					"type":       "web_search_tool_result_error",
					"error_code": "max_uses_exceeded",
				},
			},
		},
	}

	messages, err := convertMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != "tool" {
		t.Fatalf("expected role tool, got %q", messages[0].Role)
	}
	if messages[0].ToolCallID != "srvtoolu_error" {
		t.Fatalf("expected tool_call_id srvtoolu_error, got %q", messages[0].ToolCallID)
	}
	if !strings.Contains(messages[0].Content, "max_uses_exceeded") {
		t.Fatalf("expected error code in converted tool content, got %q", messages[0].Content)
	}
}

func TestConvertOllamaToAnthropicResults(t *testing.T) {
	ollamaResp := &OllamaWebSearchResponse{
		Results: []OllamaWebSearchResult{
			{
				Title:   "Test Title",
				URL:     "https://example.com",
				Content: "Test content",
			},
			{
				Title:   "Another Result",
				URL:     "https://example.org",
				Content: "More content",
			},
		},
	}

	results := ConvertOllamaToAnthropicResults(ollamaResp)

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results[0].Type != "web_search_result" {
		t.Errorf("expected type 'web_search_result', got %q", results[0].Type)
	}

	if results[0].Title != "Test Title" {
		t.Errorf("expected title 'Test Title', got %q", results[0].Title)
	}

	if results[0].URL != "https://example.com" {
		t.Errorf("expected URL 'https://example.com', got %q", results[0].URL)
	}
}

func TestWebSearchTypes(t *testing.T) {
	// Test that WebSearchResult serializes correctly
	result := WebSearchResult{
		Type:             "web_search_result",
		URL:              "https://example.com",
		Title:            "Test",
		EncryptedContent: "abc123",
		PageAge:          "2025-01-01",
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal WebSearchResult: %v", err)
	}

	var unmarshaled WebSearchResult
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("failed to unmarshal WebSearchResult: %v", err)
	}

	if unmarshaled.Type != result.Type {
		t.Errorf("type mismatch: expected %q, got %q", result.Type, unmarshaled.Type)
	}

	// Test WebSearchToolResultError
	errResult := WebSearchToolResultError{
		Type:      "web_search_tool_result_error",
		ErrorCode: "max_uses_exceeded",
	}

	data, err = json.Marshal(errResult)
	if err != nil {
		t.Fatalf("failed to marshal WebSearchToolResultError: %v", err)
	}

	var unmarshaledErr WebSearchToolResultError
	if err := json.Unmarshal(data, &unmarshaledErr); err != nil {
		t.Fatalf("failed to unmarshal WebSearchToolResultError: %v", err)
	}

	if unmarshaledErr.ErrorCode != "max_uses_exceeded" {
		t.Errorf("error_code mismatch: expected 'max_uses_exceeded', got %q", unmarshaledErr.ErrorCode)
	}
}

func TestCitation(t *testing.T) {
	citation := Citation{
		Type:           "web_search_result_location",
		URL:            "https://example.com",
		Title:          "Example",
		EncryptedIndex: "enc123",
		CitedText:      "Some cited text...",
	}

	data, err := json.Marshal(citation)
	if err != nil {
		t.Fatalf("failed to marshal Citation: %v", err)
	}

	var unmarshaled Citation
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("failed to unmarshal Citation: %v", err)
	}

	if unmarshaled.Type != "web_search_result_location" {
		t.Errorf("type mismatch: expected 'web_search_result_location', got %q", unmarshaled.Type)
	}

	if unmarshaled.CitedText != "Some cited text..." {
		t.Errorf("cited_text mismatch: expected 'Some cited text...', got %q", unmarshaled.CitedText)
	}
}
