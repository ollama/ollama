package anthropic

import (
	"encoding/base64"
	"encoding/json"
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

// TestFromMessagesRequest_ThinkingOnlyBlock verifies that messages containing only
// a thinking block (no text, images, or tool calls) are preserved and not dropped.
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
	conv := NewStreamConverter("msg_123", "test-model")

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
		Metrics:    api.Metrics{EvalCount: 5},
	}

	events2 := conv.Process(resp2)

	// Should have content_block_delta, content_block_stop, message_delta, message_stop
	hasStop := false
	for _, e := range events2 {
		if e.Event == "message_stop" {
			hasStop = true
		}
	}
	if !hasStop {
		t.Error("expected message_stop event in final chunk")
	}
}

func TestStreamConverter_WithToolCalls(t *testing.T) {
	conv := NewStreamConverter("msg_123", "test-model")

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
	conv := NewStreamConverter("msg_123", "test-model")

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
	conv := NewStreamConverter("msg_123", "test-model")

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

// TestContentBlockJSON_EmptyFieldsPresent verifies that empty text and thinking fields
// are serialized in JSON output. The Anthropic SDK requires these fields to be present
// (even when empty) in content_block_start events to properly accumulate streaming deltas.
// Without these fields, the SDK throws: "TypeError: unsupported operand type(s) for +=: 'NoneType' and 'str'"
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

// TestStreamConverter_ContentBlockStartIncludesEmptyFields verifies that content_block_start
// events include the required empty fields for SDK compatibility.
func TestStreamConverter_ContentBlockStartIncludesEmptyFields(t *testing.T) {
	t.Run("text block start includes empty text", func(t *testing.T) {
		conv := NewStreamConverter("msg_123", "test-model")

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
		conv := NewStreamConverter("msg_123", "test-model")

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
