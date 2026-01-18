package openai

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestResponsesInputMessage_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    ResponsesInputMessage
		wantErr bool
	}{
		{
			name: "text content",
			json: `{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}`,
			want: ResponsesInputMessage{
				Type:    "message",
				Role:    "user",
				Content: []ResponsesContent{ResponsesTextContent{Type: "input_text", Text: "hello"}},
			},
		},
		{
			name: "image content",
			json: `{"type": "message", "role": "user", "content": [{"type": "input_image", "detail": "auto", "image_url": "https://example.com/img.png"}]}`,
			want: ResponsesInputMessage{
				Type: "message",
				Role: "user",
				Content: []ResponsesContent{ResponsesImageContent{
					Type:     "input_image",
					Detail:   "auto",
					ImageURL: "https://example.com/img.png",
				}},
			},
		},
		{
			name: "multiple content items",
			json: `{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}, {"type": "input_text", "text": "world"}]}`,
			want: ResponsesInputMessage{
				Type: "message",
				Role: "user",
				Content: []ResponsesContent{
					ResponsesTextContent{Type: "input_text", Text: "hello"},
					ResponsesTextContent{Type: "input_text", Text: "world"},
				},
			},
		},
		{
			name:    "unknown content type",
			json:    `{"type": "message", "role": "user", "content": [{"type": "unknown"}]}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got ResponsesInputMessage
			err := json.Unmarshal([]byte(tt.json), &got)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got.Type != tt.want.Type {
				t.Errorf("Type = %q, want %q", got.Type, tt.want.Type)
			}

			if got.Role != tt.want.Role {
				t.Errorf("Role = %q, want %q", got.Role, tt.want.Role)
			}

			if len(got.Content) != len(tt.want.Content) {
				t.Fatalf("len(Content) = %d, want %d", len(got.Content), len(tt.want.Content))
			}

			for i := range tt.want.Content {
				switch wantContent := tt.want.Content[i].(type) {
				case ResponsesTextContent:
					gotContent, ok := got.Content[i].(ResponsesTextContent)
					if !ok {
						t.Fatalf("Content[%d] type = %T, want ResponsesTextContent", i, got.Content[i])
					}
					if gotContent != wantContent {
						t.Errorf("Content[%d] = %+v, want %+v", i, gotContent, wantContent)
					}
				case ResponsesImageContent:
					gotContent, ok := got.Content[i].(ResponsesImageContent)
					if !ok {
						t.Fatalf("Content[%d] type = %T, want ResponsesImageContent", i, got.Content[i])
					}
					if gotContent != wantContent {
						t.Errorf("Content[%d] = %+v, want %+v", i, gotContent, wantContent)
					}
				}
			}
		})
	}
}

func TestResponsesInput_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name      string
		json      string
		wantText  string
		wantItems int
		wantErr   bool
	}{
		{
			name:     "plain string",
			json:     `"hello world"`,
			wantText: "hello world",
		},
		{
			name:      "array with one message",
			json:      `[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}]`,
			wantItems: 1,
		},
		{
			name:      "array with multiple messages",
			json:      `[{"type": "message", "role": "system", "content": [{"type": "input_text", "text": "you are helpful"}]}, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}]`,
			wantItems: 2,
		},
		{
			name:    "invalid input",
			json:    `123`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got ResponsesInput
			err := json.Unmarshal([]byte(tt.json), &got)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got.Text != tt.wantText {
				t.Errorf("Text = %q, want %q", got.Text, tt.wantText)
			}

			if len(got.Items) != tt.wantItems {
				t.Errorf("len(Items) = %d, want %d", len(got.Items), tt.wantItems)
			}
		})
	}
}

func TestUnmarshalResponsesInputItem(t *testing.T) {
	t.Run("message item", func(t *testing.T) {
		got, err := unmarshalResponsesInputItem([]byte(`{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}`))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		msg, ok := got.(ResponsesInputMessage)
		if !ok {
			t.Fatalf("got type %T, want ResponsesInputMessage", got)
		}

		if msg.Role != "user" {
			t.Errorf("Role = %q, want %q", msg.Role, "user")
		}
	})

	t.Run("function_call item", func(t *testing.T) {
		got, err := unmarshalResponsesInputItem([]byte(`{"type": "function_call", "call_id": "call_abc123", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}`))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		fc, ok := got.(ResponsesFunctionCall)
		if !ok {
			t.Fatalf("got type %T, want ResponsesFunctionCall", got)
		}

		if fc.Type != "function_call" {
			t.Errorf("Type = %q, want %q", fc.Type, "function_call")
		}
		if fc.CallID != "call_abc123" {
			t.Errorf("CallID = %q, want %q", fc.CallID, "call_abc123")
		}
		if fc.Name != "get_weather" {
			t.Errorf("Name = %q, want %q", fc.Name, "get_weather")
		}
	})

	t.Run("function_call_output item", func(t *testing.T) {
		got, err := unmarshalResponsesInputItem([]byte(`{"type": "function_call_output", "call_id": "call_abc123", "output": "the result"}`))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		output, ok := got.(ResponsesFunctionCallOutput)
		if !ok {
			t.Fatalf("got type %T, want ResponsesFunctionCallOutput", got)
		}

		if output.Type != "function_call_output" {
			t.Errorf("Type = %q, want %q", output.Type, "function_call_output")
		}
		if output.CallID != "call_abc123" {
			t.Errorf("CallID = %q, want %q", output.CallID, "call_abc123")
		}
		if output.Output != "the result" {
			t.Errorf("Output = %q, want %q", output.Output, "the result")
		}
	})

	t.Run("unknown item type", func(t *testing.T) {
		_, err := unmarshalResponsesInputItem([]byte(`{"type": "unknown_type"}`))
		if err == nil {
			t.Error("expected error, got nil")
		}
	})
}

func TestResponsesRequest_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		check   func(t *testing.T, req ResponsesRequest)
		wantErr bool
	}{
		{
			name: "simple string input",
			json: `{"model": "gpt-oss:20b", "input": "hello"}`,
			check: func(t *testing.T, req ResponsesRequest) {
				if req.Model != "gpt-oss:20b" {
					t.Errorf("Model = %q, want %q", req.Model, "gpt-oss:20b")
				}
				if req.Input.Text != "hello" {
					t.Errorf("Input.Text = %q, want %q", req.Input.Text, "hello")
				}
			},
		},
		{
			name: "array input with messages",
			json: `{"model": "gpt-oss:20b", "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}]}`,
			check: func(t *testing.T, req ResponsesRequest) {
				if len(req.Input.Items) != 1 {
					t.Fatalf("len(Input.Items) = %d, want 1", len(req.Input.Items))
				}
				msg, ok := req.Input.Items[0].(ResponsesInputMessage)
				if !ok {
					t.Fatalf("Input.Items[0] type = %T, want ResponsesInputMessage", req.Input.Items[0])
				}
				if msg.Role != "user" {
					t.Errorf("Role = %q, want %q", msg.Role, "user")
				}
			},
		},
		{
			name: "with temperature",
			json: `{"model": "gpt-oss:20b", "input": "hello", "temperature": 0.5}`,
			check: func(t *testing.T, req ResponsesRequest) {
				if req.Temperature == nil || *req.Temperature != 0.5 {
					t.Errorf("Temperature = %v, want 0.5", req.Temperature)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got ResponsesRequest
			err := json.Unmarshal([]byte(tt.json), &got)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.check != nil {
				tt.check(t, got)
			}
		})
	}
}

func TestFromResponsesRequest_Tools(t *testing.T) {
	reqJSON := `{
		"model": "gpt-oss:20b",
		"input": "hello",
		"tools": [
			{
				"type": "function",
				"name": "shell",
				"description": "Runs a shell command",
				"strict": false,
				"parameters": {
					"type": "object",
					"properties": {
						"command": {
							"type": "array",
							"items": {"type": "string"},
							"description": "The command to execute"
						}
					},
					"required": ["command"]
				}
			}
		]
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	// Check that tools were parsed
	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(req.Tools))
	}

	if req.Tools[0].Name != "shell" {
		t.Errorf("expected tool name 'shell', got %q", req.Tools[0].Name)
	}

	// Convert and check
	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	if len(chatReq.Tools) != 1 {
		t.Fatalf("expected 1 converted tool, got %d", len(chatReq.Tools))
	}

	tool := chatReq.Tools[0]
	if tool.Type != "function" {
		t.Errorf("expected tool type 'function', got %q", tool.Type)
	}
	if tool.Function.Name != "shell" {
		t.Errorf("expected function name 'shell', got %q", tool.Function.Name)
	}
	if tool.Function.Description != "Runs a shell command" {
		t.Errorf("expected function description 'Runs a shell command', got %q", tool.Function.Description)
	}
	if tool.Function.Parameters.Type != "object" {
		t.Errorf("expected parameters type 'object', got %q", tool.Function.Parameters.Type)
	}
	if len(tool.Function.Parameters.Required) != 1 || tool.Function.Parameters.Required[0] != "command" {
		t.Errorf("expected required ['command'], got %v", tool.Function.Parameters.Required)
	}
}

func TestFromResponsesRequest_FunctionCallOutput(t *testing.T) {
	// Test a complete tool call round-trip:
	// 1. User message asking about weather
	// 2. Assistant's function call (from previous response)
	// 3. Function call output (the tool result)
	reqJSON := `{
		"model": "gpt-oss:20b",
		"input": [
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is the weather?"}]},
			{"type": "function_call", "call_id": "call_abc123", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
			{"type": "function_call_output", "call_id": "call_abc123", "output": "sunny, 72F"}
		]
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	// Check that input items were parsed
	if len(req.Input.Items) != 3 {
		t.Fatalf("expected 3 input items, got %d", len(req.Input.Items))
	}

	// Verify the function_call item
	fc, ok := req.Input.Items[1].(ResponsesFunctionCall)
	if !ok {
		t.Fatalf("Input.Items[1] type = %T, want ResponsesFunctionCall", req.Input.Items[1])
	}
	if fc.Name != "get_weather" {
		t.Errorf("Name = %q, want %q", fc.Name, "get_weather")
	}

	// Verify the function_call_output item
	fcOutput, ok := req.Input.Items[2].(ResponsesFunctionCallOutput)
	if !ok {
		t.Fatalf("Input.Items[2] type = %T, want ResponsesFunctionCallOutput", req.Input.Items[2])
	}
	if fcOutput.CallID != "call_abc123" {
		t.Errorf("CallID = %q, want %q", fcOutput.CallID, "call_abc123")
	}

	// Convert and check
	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	if len(chatReq.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
	}

	// Check the user message
	userMsg := chatReq.Messages[0]
	if userMsg.Role != "user" {
		t.Errorf("expected role 'user', got %q", userMsg.Role)
	}

	// Check the assistant message with tool call
	assistantMsg := chatReq.Messages[1]
	if assistantMsg.Role != "assistant" {
		t.Errorf("expected role 'assistant', got %q", assistantMsg.Role)
	}
	if len(assistantMsg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
	}
	if assistantMsg.ToolCalls[0].ID != "call_abc123" {
		t.Errorf("expected tool call ID 'call_abc123', got %q", assistantMsg.ToolCalls[0].ID)
	}
	if assistantMsg.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", assistantMsg.ToolCalls[0].Function.Name)
	}

	// Check the tool response message
	toolMsg := chatReq.Messages[2]
	if toolMsg.Role != "tool" {
		t.Errorf("expected role 'tool', got %q", toolMsg.Role)
	}
	if toolMsg.Content != "sunny, 72F" {
		t.Errorf("expected content 'sunny, 72F', got %q", toolMsg.Content)
	}
	if toolMsg.ToolCallID != "call_abc123" {
		t.Errorf("expected ToolCallID 'call_abc123', got %q", toolMsg.ToolCallID)
	}
}

func TestFromResponsesRequest_FunctionCallMerge(t *testing.T) {
	t.Run("function call merges with preceding assistant message", func(t *testing.T) {
		// When assistant message has content followed by function_call,
		// they should be merged into a single message
		reqJSON := `{
			"model": "gpt-oss:20b",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is the weather?"}]},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I'll check the weather for you."}]},
				{"type": "function_call", "call_id": "call_abc123", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 2 messages: user and assistant (with content + tool call merged)
		if len(chatReq.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
		}

		// Check user message
		if chatReq.Messages[0].Role != "user" {
			t.Errorf("Messages[0].Role = %q, want %q", chatReq.Messages[0].Role, "user")
		}

		// Check assistant message has both content and tool call
		assistantMsg := chatReq.Messages[1]
		if assistantMsg.Role != "assistant" {
			t.Errorf("Messages[1].Role = %q, want %q", assistantMsg.Role, "assistant")
		}
		if assistantMsg.Content != "I'll check the weather for you." {
			t.Errorf("Messages[1].Content = %q, want %q", assistantMsg.Content, "I'll check the weather for you.")
		}
		if len(assistantMsg.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
		}
		if assistantMsg.ToolCalls[0].Function.Name != "get_weather" {
			t.Errorf("ToolCalls[0].Function.Name = %q, want %q", assistantMsg.ToolCalls[0].Function.Name, "get_weather")
		}
	})

	t.Run("function call without preceding assistant creates new message", func(t *testing.T) {
		// When there's no preceding assistant message, function_call creates its own message
		reqJSON := `{
			"model": "gpt-oss:20b",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is the weather?"}]},
				{"type": "function_call", "call_id": "call_abc123", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 2 messages: user and assistant (tool call only)
		if len(chatReq.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
		}

		// Check assistant message has tool call but no content
		assistantMsg := chatReq.Messages[1]
		if assistantMsg.Role != "assistant" {
			t.Errorf("Messages[1].Role = %q, want %q", assistantMsg.Role, "assistant")
		}
		if assistantMsg.Content != "" {
			t.Errorf("Messages[1].Content = %q, want empty", assistantMsg.Content)
		}
		if len(assistantMsg.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
		}
	})

	t.Run("multiple function calls merge into same assistant message", func(t *testing.T) {
		// Multiple consecutive function_calls should all merge into the same assistant message
		reqJSON := `{
			"model": "gpt-oss:20b",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "check weather and time"}]},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I'll check both."}]},
				{"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
				{"type": "function_call", "call_id": "call_2", "name": "get_time", "arguments": "{\"city\":\"Paris\"}"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 2 messages: user and assistant (content + both tool calls)
		if len(chatReq.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
		}

		// Assistant has content + both tool calls
		assistantMsg := chatReq.Messages[1]
		if assistantMsg.Content != "I'll check both." {
			t.Errorf("Messages[1].Content = %q, want %q", assistantMsg.Content, "I'll check both.")
		}
		if len(assistantMsg.ToolCalls) != 2 {
			t.Fatalf("expected 2 tool calls, got %d", len(assistantMsg.ToolCalls))
		}
		if assistantMsg.ToolCalls[0].Function.Name != "get_weather" {
			t.Errorf("ToolCalls[0].Function.Name = %q, want %q", assistantMsg.ToolCalls[0].Function.Name, "get_weather")
		}
		if assistantMsg.ToolCalls[1].Function.Name != "get_time" {
			t.Errorf("ToolCalls[1].Function.Name = %q, want %q", assistantMsg.ToolCalls[1].Function.Name, "get_time")
		}
	})

	t.Run("new assistant message starts fresh tool call group", func(t *testing.T) {
		// assistant → tool_call → tool_call → assistant → tool_call
		// Should result in 2 assistant messages with their respective tool calls
		reqJSON := `{
			"model": "gpt-oss:20b",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do multiple things"}]},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "First batch."}]},
				{"type": "function_call", "call_id": "call_1", "name": "func_a", "arguments": "{}"},
				{"type": "function_call", "call_id": "call_2", "name": "func_b", "arguments": "{}"},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Second batch."}]},
				{"type": "function_call", "call_id": "call_3", "name": "func_c", "arguments": "{}"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 3 messages:
		// 1. user
		// 2. assistant "First batch." + tool calls [func_a, func_b]
		// 3. assistant "Second batch." + tool calls [func_c]
		if len(chatReq.Messages) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
		}

		asst1 := chatReq.Messages[1]
		if asst1.Content != "First batch." {
			t.Errorf("Messages[1].Content = %q, want %q", asst1.Content, "First batch.")
		}
		if len(asst1.ToolCalls) != 2 {
			t.Fatalf("expected 2 tool calls in Messages[1], got %d", len(asst1.ToolCalls))
		}
		if asst1.ToolCalls[0].Function.Name != "func_a" {
			t.Errorf("Messages[1].ToolCalls[0] = %q, want %q", asst1.ToolCalls[0].Function.Name, "func_a")
		}
		if asst1.ToolCalls[1].Function.Name != "func_b" {
			t.Errorf("Messages[1].ToolCalls[1] = %q, want %q", asst1.ToolCalls[1].Function.Name, "func_b")
		}

		asst2 := chatReq.Messages[2]
		if asst2.Content != "Second batch." {
			t.Errorf("Messages[2].Content = %q, want %q", asst2.Content, "Second batch.")
		}
		if len(asst2.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call in Messages[2], got %d", len(asst2.ToolCalls))
		}
		if asst2.ToolCalls[0].Function.Name != "func_c" {
			t.Errorf("Messages[2].ToolCalls[0] = %q, want %q", asst2.ToolCalls[0].Function.Name, "func_c")
		}
	})

	t.Run("function call merges with assistant that has thinking", func(t *testing.T) {
		// reasoning → assistant (gets thinking) → function_call → should merge
		reqJSON := `{
			"model": "gpt-oss:20b",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "think and act"}]},
				{"type": "reasoning", "id": "rs_1", "encrypted_content": "Let me think...", "summary": []},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I thought about it."}]},
				{"type": "function_call", "call_id": "call_1", "name": "do_thing", "arguments": "{}"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 2 messages: user and assistant (thinking + content + tool call)
		if len(chatReq.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
		}

		asst := chatReq.Messages[1]
		if asst.Thinking != "Let me think..." {
			t.Errorf("Messages[1].Thinking = %q, want %q", asst.Thinking, "Let me think...")
		}
		if asst.Content != "I thought about it." {
			t.Errorf("Messages[1].Content = %q, want %q", asst.Content, "I thought about it.")
		}
		if len(asst.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(asst.ToolCalls))
		}
		if asst.ToolCalls[0].Function.Name != "do_thing" {
			t.Errorf("ToolCalls[0].Function.Name = %q, want %q", asst.ToolCalls[0].Function.Name, "do_thing")
		}
	})

	t.Run("mixed thinking and content with multiple tool calls", func(t *testing.T) {
		// Test:
		// 1. reasoning → assistant (empty content, gets thinking) → tc (merges)
		// 2. assistant with content → tc → tc (both merge)
		// Result: 2 assistant messages
		reqJSON := `{
			"model": "gpt-oss:20b",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "complex task"}]},
				{"type": "reasoning", "id": "rs_1", "encrypted_content": "Thinking first...", "summary": []},
				{"type": "message", "role": "assistant", "content": ""},
				{"type": "function_call", "call_id": "call_1", "name": "think_action", "arguments": "{}"},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Now doing more."}]},
				{"type": "function_call", "call_id": "call_2", "name": "action_a", "arguments": "{}"},
				{"type": "function_call", "call_id": "call_3", "name": "action_b", "arguments": "{}"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 3 messages:
		// 1. user
		// 2. assistant with thinking + tool call [think_action]
		// 3. assistant with content "Now doing more." + tool calls [action_a, action_b]
		if len(chatReq.Messages) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
		}

		// First assistant: thinking + tool call
		asst1 := chatReq.Messages[1]
		if asst1.Thinking != "Thinking first..." {
			t.Errorf("Messages[1].Thinking = %q, want %q", asst1.Thinking, "Thinking first...")
		}
		if asst1.Content != "" {
			t.Errorf("Messages[1].Content = %q, want empty", asst1.Content)
		}
		if len(asst1.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call in Messages[1], got %d", len(asst1.ToolCalls))
		}
		if asst1.ToolCalls[0].Function.Name != "think_action" {
			t.Errorf("Messages[1].ToolCalls[0] = %q, want %q", asst1.ToolCalls[0].Function.Name, "think_action")
		}

		// Second assistant: content + 2 tool calls
		asst2 := chatReq.Messages[2]
		if asst2.Content != "Now doing more." {
			t.Errorf("Messages[2].Content = %q, want %q", asst2.Content, "Now doing more.")
		}
		if len(asst2.ToolCalls) != 2 {
			t.Fatalf("expected 2 tool calls in Messages[2], got %d", len(asst2.ToolCalls))
		}
		if asst2.ToolCalls[0].Function.Name != "action_a" {
			t.Errorf("Messages[2].ToolCalls[0] = %q, want %q", asst2.ToolCalls[0].Function.Name, "action_a")
		}
		if asst2.ToolCalls[1].Function.Name != "action_b" {
			t.Errorf("Messages[2].ToolCalls[1] = %q, want %q", asst2.ToolCalls[1].Function.Name, "action_b")
		}
	})
}

func TestDecodeImageURL(t *testing.T) {
	// Valid PNG base64 (1x1 red pixel)
	validPNG := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

	t.Run("valid png", func(t *testing.T) {
		img, err := decodeImageURL(validPNG)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(img) == 0 {
			t.Error("expected non-empty image data")
		}
	})

	t.Run("valid jpeg", func(t *testing.T) {
		// Just test the prefix validation with minimal base64
		_, err := decodeImageURL("data:image/jpeg;base64,/9j/4AAQSkZJRg==")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("blank mime type", func(t *testing.T) {
		_, err := decodeImageURL("data:;base64,dGVzdA==")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("invalid mime type", func(t *testing.T) {
		_, err := decodeImageURL("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
		if err == nil {
			t.Error("expected error for unsupported mime type")
		}
	})

	t.Run("invalid base64", func(t *testing.T) {
		_, err := decodeImageURL("data:image/png;base64,not-valid-base64!")
		if err == nil {
			t.Error("expected error for invalid base64")
		}
	})

	t.Run("not a data url", func(t *testing.T) {
		_, err := decodeImageURL("https://example.com/image.png")
		if err == nil {
			t.Error("expected error for non-data URL")
		}
	})
}

func TestFromResponsesRequest_Images(t *testing.T) {
	// 1x1 red PNG pixel
	pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

	reqJSON := `{
		"model": "llava",
		"input": [
			{"type": "message", "role": "user", "content": [
				{"type": "input_text", "text": "What is in this image?"},
				{"type": "input_image", "detail": "auto", "image_url": "data:image/png;base64,` + pngBase64 + `"}
			]}
		]
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	if len(chatReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(chatReq.Messages))
	}

	msg := chatReq.Messages[0]
	if msg.Role != "user" {
		t.Errorf("expected role 'user', got %q", msg.Role)
	}
	if msg.Content != "What is in this image?" {
		t.Errorf("expected content 'What is in this image?', got %q", msg.Content)
	}
	if len(msg.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(msg.Images))
	}
	if len(msg.Images[0]) == 0 {
		t.Error("expected non-empty image data")
	}
}

func TestResponsesStreamConverter_TextOnly(t *testing.T) {
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	// First chunk with content
	events := converter.Process(api.ChatResponse{
		Message: api.Message{
			Content: "Hello",
		},
	})

	// Should have: response.created, response.in_progress, output_item.added, content_part.added, output_text.delta
	if len(events) != 5 {
		t.Fatalf("expected 5 events, got %d", len(events))
	}

	if events[0].Event != "response.created" {
		t.Errorf("events[0].Event = %q, want %q", events[0].Event, "response.created")
	}
	if events[1].Event != "response.in_progress" {
		t.Errorf("events[1].Event = %q, want %q", events[1].Event, "response.in_progress")
	}
	if events[2].Event != "response.output_item.added" {
		t.Errorf("events[2].Event = %q, want %q", events[2].Event, "response.output_item.added")
	}
	if events[3].Event != "response.content_part.added" {
		t.Errorf("events[3].Event = %q, want %q", events[3].Event, "response.content_part.added")
	}
	if events[4].Event != "response.output_text.delta" {
		t.Errorf("events[4].Event = %q, want %q", events[4].Event, "response.output_text.delta")
	}

	// Second chunk with more content
	events = converter.Process(api.ChatResponse{
		Message: api.Message{
			Content: " World",
		},
	})

	// Should only have output_text.delta (no more created/in_progress/added)
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	if events[0].Event != "response.output_text.delta" {
		t.Errorf("events[0].Event = %q, want %q", events[0].Event, "response.output_text.delta")
	}

	// Final chunk
	events = converter.Process(api.ChatResponse{
		Message: api.Message{},
		Done:    true,
	})

	// Should have: output_text.done, content_part.done, output_item.done, response.completed
	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d", len(events))
	}
	if events[0].Event != "response.output_text.done" {
		t.Errorf("events[0].Event = %q, want %q", events[0].Event, "response.output_text.done")
	}
	// Check that accumulated text is present
	data := events[0].Data.(map[string]any)
	if data["text"] != "Hello World" {
		t.Errorf("accumulated text = %q, want %q", data["text"], "Hello World")
	}
}

func TestResponsesStreamConverter_ToolCalls(t *testing.T) {
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	events := converter.Process(api.ChatResponse{
		Message: api.Message{
			ToolCalls: []api.ToolCall{
				{
					ID: "call_abc",
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "Paris"}),
					},
				},
			},
		},
	})

	// Should have: created, in_progress, output_item.added, arguments.delta, arguments.done, output_item.done
	if len(events) != 6 {
		t.Fatalf("expected 6 events, got %d", len(events))
	}

	if events[2].Event != "response.output_item.added" {
		t.Errorf("events[2].Event = %q, want %q", events[2].Event, "response.output_item.added")
	}
	if events[3].Event != "response.function_call_arguments.delta" {
		t.Errorf("events[3].Event = %q, want %q", events[3].Event, "response.function_call_arguments.delta")
	}
	if events[4].Event != "response.function_call_arguments.done" {
		t.Errorf("events[4].Event = %q, want %q", events[4].Event, "response.function_call_arguments.done")
	}
	if events[5].Event != "response.output_item.done" {
		t.Errorf("events[5].Event = %q, want %q", events[5].Event, "response.output_item.done")
	}
}

func TestResponsesStreamConverter_Reasoning(t *testing.T) {
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	// First chunk with thinking
	events := converter.Process(api.ChatResponse{
		Message: api.Message{
			Thinking: "Let me think...",
		},
	})

	// Should have: created, in_progress, output_item.added (reasoning), reasoning_summary_text.delta
	if len(events) != 4 {
		t.Fatalf("expected 4 events, got %d", len(events))
	}

	if events[2].Event != "response.output_item.added" {
		t.Errorf("events[2].Event = %q, want %q", events[2].Event, "response.output_item.added")
	}
	// Check it's a reasoning item
	data := events[2].Data.(map[string]any)
	item := data["item"].(map[string]any)
	if item["type"] != "reasoning" {
		t.Errorf("item type = %q, want %q", item["type"], "reasoning")
	}

	if events[3].Event != "response.reasoning_summary_text.delta" {
		t.Errorf("events[3].Event = %q, want %q", events[3].Event, "response.reasoning_summary_text.delta")
	}

	// Second chunk with text content (reasoning should close first)
	events = converter.Process(api.ChatResponse{
		Message: api.Message{
			Content: "The answer is 42",
		},
	})

	// Should have: reasoning_summary_text.done, output_item.done (reasoning), output_item.added (message), content_part.added, output_text.delta
	if len(events) != 5 {
		t.Fatalf("expected 5 events, got %d", len(events))
	}

	if events[0].Event != "response.reasoning_summary_text.done" {
		t.Errorf("events[0].Event = %q, want %q", events[0].Event, "response.reasoning_summary_text.done")
	}
	if events[1].Event != "response.output_item.done" {
		t.Errorf("events[1].Event = %q, want %q", events[1].Event, "response.output_item.done")
	}
	// Check the reasoning done item has encrypted_content
	doneData := events[1].Data.(map[string]any)
	doneItem := doneData["item"].(map[string]any)
	if doneItem["encrypted_content"] != "Let me think..." {
		t.Errorf("encrypted_content = %q, want %q", doneItem["encrypted_content"], "Let me think...")
	}
}

func TestFromResponsesRequest_ReasoningMerge(t *testing.T) {
	t.Run("reasoning merged with following message", func(t *testing.T) {
		reqJSON := `{
			"model": "qwen3",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "solve 2+2"}]},
				{"type": "reasoning", "id": "rs_123", "encrypted_content": "Let me think about this math problem...", "summary": [{"type": "summary_text", "text": "Thinking about math"}]},
				{"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "The answer is 4"}]}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 2 messages: user and assistant (with thinking merged)
		if len(chatReq.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
		}

		// Check user message
		if chatReq.Messages[0].Role != "user" {
			t.Errorf("Messages[0].Role = %q, want %q", chatReq.Messages[0].Role, "user")
		}

		// Check assistant message has both content and thinking
		assistantMsg := chatReq.Messages[1]
		if assistantMsg.Role != "assistant" {
			t.Errorf("Messages[1].Role = %q, want %q", assistantMsg.Role, "assistant")
		}
		if assistantMsg.Content != "The answer is 4" {
			t.Errorf("Messages[1].Content = %q, want %q", assistantMsg.Content, "The answer is 4")
		}
		if assistantMsg.Thinking != "Let me think about this math problem..." {
			t.Errorf("Messages[1].Thinking = %q, want %q", assistantMsg.Thinking, "Let me think about this math problem...")
		}
	})

	t.Run("reasoning merged with following function call", func(t *testing.T) {
		reqJSON := `{
			"model": "qwen3",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is the weather?"}]},
				{"type": "reasoning", "id": "rs_123", "encrypted_content": "I need to call a tool for this...", "summary": []},
				{"type": "function_call", "call_id": "call_abc", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 2 messages: user and assistant (with thinking + tool call)
		if len(chatReq.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
		}

		// Check assistant message has both tool call and thinking
		assistantMsg := chatReq.Messages[1]
		if assistantMsg.Role != "assistant" {
			t.Errorf("Messages[1].Role = %q, want %q", assistantMsg.Role, "assistant")
		}
		if assistantMsg.Thinking != "I need to call a tool for this..." {
			t.Errorf("Messages[1].Thinking = %q, want %q", assistantMsg.Thinking, "I need to call a tool for this...")
		}
		if len(assistantMsg.ToolCalls) != 1 {
			t.Fatalf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
		}
		if assistantMsg.ToolCalls[0].Function.Name != "get_weather" {
			t.Errorf("ToolCalls[0].Function.Name = %q, want %q", assistantMsg.ToolCalls[0].Function.Name, "get_weather")
		}
	})

	t.Run("multi-turn conversation with reasoning", func(t *testing.T) {
		// Simulates: user asks -> model thinks + responds -> user follows up
		reqJSON := `{
			"model": "qwen3",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is 2+2?"}]},
				{"type": "reasoning", "id": "rs_001", "encrypted_content": "This is a simple arithmetic problem. 2+2=4.", "summary": [{"type": "summary_text", "text": "Calculating 2+2"}]},
				{"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "The answer is 4."}]},
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Now multiply that by 3"}]}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 3 messages:
		// 1. user: "What is 2+2?"
		// 2. assistant: thinking + "The answer is 4."
		// 3. user: "Now multiply that by 3"
		if len(chatReq.Messages) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
		}

		// Check first user message
		if chatReq.Messages[0].Role != "user" || chatReq.Messages[0].Content != "What is 2+2?" {
			t.Errorf("Messages[0] = {Role: %q, Content: %q}, want {Role: \"user\", Content: \"What is 2+2?\"}",
				chatReq.Messages[0].Role, chatReq.Messages[0].Content)
		}

		// Check assistant message has merged thinking + content
		if chatReq.Messages[1].Role != "assistant" {
			t.Errorf("Messages[1].Role = %q, want \"assistant\"", chatReq.Messages[1].Role)
		}
		if chatReq.Messages[1].Content != "The answer is 4." {
			t.Errorf("Messages[1].Content = %q, want \"The answer is 4.\"", chatReq.Messages[1].Content)
		}
		if chatReq.Messages[1].Thinking != "This is a simple arithmetic problem. 2+2=4." {
			t.Errorf("Messages[1].Thinking = %q, want \"This is a simple arithmetic problem. 2+2=4.\"",
				chatReq.Messages[1].Thinking)
		}

		// Check second user message
		if chatReq.Messages[2].Role != "user" || chatReq.Messages[2].Content != "Now multiply that by 3" {
			t.Errorf("Messages[2] = {Role: %q, Content: %q}, want {Role: \"user\", Content: \"Now multiply that by 3\"}",
				chatReq.Messages[2].Role, chatReq.Messages[2].Content)
		}
	})

	t.Run("multi-turn with tool calls and reasoning", func(t *testing.T) {
		// Simulates: user asks -> model thinks + calls tool -> tool responds -> model thinks + responds -> user follows up
		reqJSON := `{
			"model": "qwen3",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is the weather in Paris?"}]},
				{"type": "reasoning", "id": "rs_001", "encrypted_content": "I need to call the weather API for Paris.", "summary": []},
				{"type": "function_call", "call_id": "call_abc", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
				{"type": "function_call_output", "call_id": "call_abc", "output": "Sunny, 72°F"},
				{"type": "reasoning", "id": "rs_002", "encrypted_content": "The weather API returned sunny and 72°F. I should format this nicely.", "summary": []},
				{"type": "message", "role": "assistant", "content": [{"type": "input_text", "text": "It's sunny and 72°F in Paris!"}]},
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What about London?"}]}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 5 messages:
		// 1. user: "What is the weather in Paris?"
		// 2. assistant: thinking + tool call
		// 3. tool: "Sunny, 72°F"
		// 4. assistant: thinking + "It's sunny and 72°F in Paris!"
		// 5. user: "What about London?"
		if len(chatReq.Messages) != 5 {
			t.Fatalf("expected 5 messages, got %d", len(chatReq.Messages))
		}

		// Message 1: user
		if chatReq.Messages[0].Role != "user" {
			t.Errorf("Messages[0].Role = %q, want \"user\"", chatReq.Messages[0].Role)
		}

		// Message 2: assistant with thinking + tool call
		if chatReq.Messages[1].Role != "assistant" {
			t.Errorf("Messages[1].Role = %q, want \"assistant\"", chatReq.Messages[1].Role)
		}
		if chatReq.Messages[1].Thinking != "I need to call the weather API for Paris." {
			t.Errorf("Messages[1].Thinking = %q, want \"I need to call the weather API for Paris.\"", chatReq.Messages[1].Thinking)
		}
		if len(chatReq.Messages[1].ToolCalls) != 1 || chatReq.Messages[1].ToolCalls[0].Function.Name != "get_weather" {
			t.Errorf("Messages[1].ToolCalls not as expected")
		}

		// Message 3: tool response
		if chatReq.Messages[2].Role != "tool" || chatReq.Messages[2].Content != "Sunny, 72°F" {
			t.Errorf("Messages[2] = {Role: %q, Content: %q}, want {Role: \"tool\", Content: \"Sunny, 72°F\"}",
				chatReq.Messages[2].Role, chatReq.Messages[2].Content)
		}

		// Message 4: assistant with thinking + content
		if chatReq.Messages[3].Role != "assistant" {
			t.Errorf("Messages[3].Role = %q, want \"assistant\"", chatReq.Messages[3].Role)
		}
		if chatReq.Messages[3].Thinking != "The weather API returned sunny and 72°F. I should format this nicely." {
			t.Errorf("Messages[3].Thinking = %q, want correct thinking", chatReq.Messages[3].Thinking)
		}
		if chatReq.Messages[3].Content != "It's sunny and 72°F in Paris!" {
			t.Errorf("Messages[3].Content = %q, want \"It's sunny and 72°F in Paris!\"", chatReq.Messages[3].Content)
		}

		// Message 5: user follow-up
		if chatReq.Messages[4].Role != "user" || chatReq.Messages[4].Content != "What about London?" {
			t.Errorf("Messages[4] = {Role: %q, Content: %q}, want {Role: \"user\", Content: \"What about London?\"}",
				chatReq.Messages[4].Role, chatReq.Messages[4].Content)
		}
	})

	t.Run("trailing reasoning creates separate message", func(t *testing.T) {
		reqJSON := `{
			"model": "qwen3",
			"input": [
				{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "think about this"}]},
				{"type": "reasoning", "id": "rs_123", "encrypted_content": "Still thinking...", "summary": []}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		// Should have 2 messages: user and assistant (thinking only)
		if len(chatReq.Messages) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
		}

		// Check assistant message has only thinking
		assistantMsg := chatReq.Messages[1]
		if assistantMsg.Role != "assistant" {
			t.Errorf("Messages[1].Role = %q, want %q", assistantMsg.Role, "assistant")
		}
		if assistantMsg.Thinking != "Still thinking..." {
			t.Errorf("Messages[1].Thinking = %q, want %q", assistantMsg.Thinking, "Still thinking...")
		}
		if assistantMsg.Content != "" {
			t.Errorf("Messages[1].Content = %q, want empty", assistantMsg.Content)
		}
	})
}

func TestToResponse_WithReasoning(t *testing.T) {
	response := ToResponse("gpt-oss:20b", "resp_123", "msg_456", api.ChatResponse{
		CreatedAt: time.Now(),
		Message: api.Message{
			Thinking: "Analyzing the question...",
			Content:  "The answer is 42",
		},
		Done: true,
	}, ResponsesRequest{})

	// Should have 2 output items: reasoning + message
	if len(response.Output) != 2 {
		t.Fatalf("expected 2 output items, got %d", len(response.Output))
	}

	// First item should be reasoning
	if response.Output[0].Type != "reasoning" {
		t.Errorf("Output[0].Type = %q, want %q", response.Output[0].Type, "reasoning")
	}
	if len(response.Output[0].Summary) != 1 {
		t.Fatalf("expected 1 summary item, got %d", len(response.Output[0].Summary))
	}
	if response.Output[0].Summary[0].Text != "Analyzing the question..." {
		t.Errorf("Summary[0].Text = %q, want %q", response.Output[0].Summary[0].Text, "Analyzing the question...")
	}
	if response.Output[0].EncryptedContent != "Analyzing the question..." {
		t.Errorf("EncryptedContent = %q, want %q", response.Output[0].EncryptedContent, "Analyzing the question...")
	}

	// Second item should be message
	if response.Output[1].Type != "message" {
		t.Errorf("Output[1].Type = %q, want %q", response.Output[1].Type, "message")
	}
	if response.Output[1].Content[0].Text != "The answer is 42" {
		t.Errorf("Content[0].Text = %q, want %q", response.Output[1].Content[0].Text, "The answer is 42")
	}
}

func TestFromResponsesRequest_Instructions(t *testing.T) {
	reqJSON := `{
		"model": "gpt-oss:20b",
		"instructions": "You are a helpful pirate. Always respond in pirate speak.",
		"input": "Hello"
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	// Should have 2 messages: system (instructions) + user
	if len(chatReq.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(chatReq.Messages))
	}

	// First message should be system with instructions
	if chatReq.Messages[0].Role != "system" {
		t.Errorf("Messages[0].Role = %q, want %q", chatReq.Messages[0].Role, "system")
	}
	if chatReq.Messages[0].Content != "You are a helpful pirate. Always respond in pirate speak." {
		t.Errorf("Messages[0].Content = %q, want instructions", chatReq.Messages[0].Content)
	}

	// Second message should be user
	if chatReq.Messages[1].Role != "user" {
		t.Errorf("Messages[1].Role = %q, want %q", chatReq.Messages[1].Role, "user")
	}
	if chatReq.Messages[1].Content != "Hello" {
		t.Errorf("Messages[1].Content = %q, want %q", chatReq.Messages[1].Content, "Hello")
	}
}

func TestFromResponsesRequest_MaxOutputTokens(t *testing.T) {
	reqJSON := `{
		"model": "gpt-oss:20b",
		"input": "Write a story",
		"max_output_tokens": 100
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	// Check that num_predict is set in options
	numPredict, ok := chatReq.Options["num_predict"]
	if !ok {
		t.Fatal("expected num_predict in options")
	}
	if numPredict != 100 {
		t.Errorf("num_predict = %v, want 100", numPredict)
	}
}

func TestFromResponsesRequest_TextFormatJsonSchema(t *testing.T) {
	reqJSON := `{
		"model": "gpt-oss:20b",
		"input": "Give me info about John who is 30",
		"text": {
			"format": {
				"type": "json_schema",
				"name": "person",
				"strict": true,
				"schema": {
					"type": "object",
					"properties": {
						"name": {"type": "string"},
						"age": {"type": "integer"}
					},
					"required": ["name", "age"]
				}
			}
		}
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	// Verify the text format was parsed
	if req.Text == nil || req.Text.Format == nil {
		t.Fatal("expected Text.Format to be set")
	}
	if req.Text.Format.Type != "json_schema" {
		t.Errorf("Text.Format.Type = %q, want %q", req.Text.Format.Type, "json_schema")
	}

	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	// Check that Format is set
	if chatReq.Format == nil {
		t.Fatal("expected Format to be set")
	}

	// Verify the schema is passed through
	var schema map[string]any
	if err := json.Unmarshal(chatReq.Format, &schema); err != nil {
		t.Fatalf("failed to unmarshal format: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("schema type = %v, want %q", schema["type"], "object")
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}
	if _, ok := props["name"]; !ok {
		t.Error("expected 'name' in schema properties")
	}
	if _, ok := props["age"]; !ok {
		t.Error("expected 'age' in schema properties")
	}
}

func TestFromResponsesRequest_TextFormatText(t *testing.T) {
	// When format type is "text", Format should be nil (no constraint)
	reqJSON := `{
		"model": "gpt-oss:20b",
		"input": "Hello",
		"text": {
			"format": {
				"type": "text"
			}
		}
	}`

	var req ResponsesRequest
	if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}

	chatReq, err := FromResponsesRequest(req)
	if err != nil {
		t.Fatalf("failed to convert request: %v", err)
	}

	// Format should be nil for "text" type
	if chatReq.Format != nil {
		t.Errorf("expected Format to be nil for text type, got %s", string(chatReq.Format))
	}
}

func TestResponsesInputMessage_ShorthandFormats(t *testing.T) {
	t.Run("string content shorthand", func(t *testing.T) {
		// Content can be a plain string instead of an array of content items
		jsonStr := `{"type": "message", "role": "user", "content": "Hello world"}`

		var msg ResponsesInputMessage
		if err := json.Unmarshal([]byte(jsonStr), &msg); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if msg.Role != "user" {
			t.Errorf("Role = %q, want %q", msg.Role, "user")
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}

		textContent, ok := msg.Content[0].(ResponsesTextContent)
		if !ok {
			t.Fatalf("Content[0] type = %T, want ResponsesTextContent", msg.Content[0])
		}
		if textContent.Text != "Hello world" {
			t.Errorf("Content[0].Text = %q, want %q", textContent.Text, "Hello world")
		}
		if textContent.Type != "input_text" {
			t.Errorf("Content[0].Type = %q, want %q", textContent.Type, "input_text")
		}
	})

	t.Run("output_text content type", func(t *testing.T) {
		// Previous assistant responses come back with output_text content type
		jsonStr := `{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I am an assistant"}]}`

		var msg ResponsesInputMessage
		if err := json.Unmarshal([]byte(jsonStr), &msg); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if msg.Role != "assistant" {
			t.Errorf("Role = %q, want %q", msg.Role, "assistant")
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}

		outputContent, ok := msg.Content[0].(ResponsesOutputTextContent)
		if !ok {
			t.Fatalf("Content[0] type = %T, want ResponsesOutputTextContent", msg.Content[0])
		}
		if outputContent.Text != "I am an assistant" {
			t.Errorf("Content[0].Text = %q, want %q", outputContent.Text, "I am an assistant")
		}
	})
}

func TestUnmarshalResponsesInputItem_ShorthandMessage(t *testing.T) {
	t.Run("message without type field", func(t *testing.T) {
		// When type is omitted but role is present, treat as message
		jsonStr := `{"role": "user", "content": "Hello"}`

		item, err := unmarshalResponsesInputItem([]byte(jsonStr))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		msg, ok := item.(ResponsesInputMessage)
		if !ok {
			t.Fatalf("got type %T, want ResponsesInputMessage", item)
		}
		if msg.Role != "user" {
			t.Errorf("Role = %q, want %q", msg.Role, "user")
		}
		if len(msg.Content) != 1 {
			t.Fatalf("len(Content) = %d, want 1", len(msg.Content))
		}
	})

	t.Run("message with both type and role", func(t *testing.T) {
		// Explicit type should still work
		jsonStr := `{"type": "message", "role": "system", "content": "You are helpful"}`

		item, err := unmarshalResponsesInputItem([]byte(jsonStr))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		msg, ok := item.(ResponsesInputMessage)
		if !ok {
			t.Fatalf("got type %T, want ResponsesInputMessage", item)
		}
		if msg.Role != "system" {
			t.Errorf("Role = %q, want %q", msg.Role, "system")
		}
	})
}

func TestFromResponsesRequest_ShorthandFormats(t *testing.T) {
	t.Run("shorthand message without type", func(t *testing.T) {
		// Real-world format from OpenAI SDK
		reqJSON := `{
			"model": "gpt-4.1",
			"input": [
				{"role": "user", "content": "What is the weather in Tokyo?"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		if len(req.Input.Items) != 1 {
			t.Fatalf("expected 1 input item, got %d", len(req.Input.Items))
		}

		msg, ok := req.Input.Items[0].(ResponsesInputMessage)
		if !ok {
			t.Fatalf("Input.Items[0] type = %T, want ResponsesInputMessage", req.Input.Items[0])
		}
		if msg.Role != "user" {
			t.Errorf("Role = %q, want %q", msg.Role, "user")
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		if len(chatReq.Messages) != 1 {
			t.Fatalf("expected 1 message, got %d", len(chatReq.Messages))
		}
		if chatReq.Messages[0].Content != "What is the weather in Tokyo?" {
			t.Errorf("Content = %q, want %q", chatReq.Messages[0].Content, "What is the weather in Tokyo?")
		}
	})

	t.Run("conversation with output_text from previous response", func(t *testing.T) {
		// Simulates a multi-turn conversation where previous assistant response is sent back
		reqJSON := `{
			"model": "gpt-4.1",
			"input": [
				{"role": "user", "content": "Hello"},
				{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi there!"}]},
				{"role": "user", "content": "How are you?"}
			]
		}`

		var req ResponsesRequest
		if err := json.Unmarshal([]byte(reqJSON), &req); err != nil {
			t.Fatalf("failed to unmarshal request: %v", err)
		}

		chatReq, err := FromResponsesRequest(req)
		if err != nil {
			t.Fatalf("failed to convert request: %v", err)
		}

		if len(chatReq.Messages) != 3 {
			t.Fatalf("expected 3 messages, got %d", len(chatReq.Messages))
		}

		// Check first user message
		if chatReq.Messages[0].Role != "user" || chatReq.Messages[0].Content != "Hello" {
			t.Errorf("Messages[0] = {Role: %q, Content: %q}, want {Role: \"user\", Content: \"Hello\"}",
				chatReq.Messages[0].Role, chatReq.Messages[0].Content)
		}

		// Check assistant message (output_text should be converted to content)
		if chatReq.Messages[1].Role != "assistant" || chatReq.Messages[1].Content != "Hi there!" {
			t.Errorf("Messages[1] = {Role: %q, Content: %q}, want {Role: \"assistant\", Content: \"Hi there!\"}",
				chatReq.Messages[1].Role, chatReq.Messages[1].Content)
		}

		// Check second user message
		if chatReq.Messages[2].Role != "user" || chatReq.Messages[2].Content != "How are you?" {
			t.Errorf("Messages[2] = {Role: %q, Content: %q}, want {Role: \"user\", Content: \"How are you?\"}",
				chatReq.Messages[2].Role, chatReq.Messages[2].Content)
		}
	})
}

func TestResponsesStreamConverter_OutputIncludesContent(t *testing.T) {
	// Verify that response.output_item.done includes content field for messages
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	// First chunk
	converter.Process(api.ChatResponse{
		Message: api.Message{Content: "Hello World"},
	})

	// Final chunk
	events := converter.Process(api.ChatResponse{
		Message: api.Message{},
		Done:    true,
	})

	// Find the output_item.done event
	var outputItemDone map[string]any
	for _, event := range events {
		if event.Event == "response.output_item.done" {
			outputItemDone = event.Data.(map[string]any)
			break
		}
	}

	if outputItemDone == nil {
		t.Fatal("expected response.output_item.done event")
	}

	item := outputItemDone["item"].(map[string]any)
	if item["type"] != "message" {
		t.Errorf("item.type = %q, want %q", item["type"], "message")
	}

	content, ok := item["content"].([]map[string]any)
	if !ok {
		t.Fatalf("item.content type = %T, want []map[string]any", item["content"])
	}
	if len(content) != 1 {
		t.Fatalf("len(content) = %d, want 1", len(content))
	}
	if content[0]["type"] != "output_text" {
		t.Errorf("content[0].type = %q, want %q", content[0]["type"], "output_text")
	}
	if content[0]["text"] != "Hello World" {
		t.Errorf("content[0].text = %q, want %q", content[0]["text"], "Hello World")
	}
}

func TestResponsesStreamConverter_ResponseCompletedIncludesOutput(t *testing.T) {
	// Verify that response.completed includes the output array
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	// Process some content
	converter.Process(api.ChatResponse{
		Message: api.Message{Content: "Test response"},
	})

	// Final chunk
	events := converter.Process(api.ChatResponse{
		Message: api.Message{},
		Done:    true,
	})

	// Find the response.completed event
	var responseCompleted map[string]any
	for _, event := range events {
		if event.Event == "response.completed" {
			responseCompleted = event.Data.(map[string]any)
			break
		}
	}

	if responseCompleted == nil {
		t.Fatal("expected response.completed event")
	}

	response := responseCompleted["response"].(map[string]any)
	output, ok := response["output"].([]any)
	if !ok {
		t.Fatalf("response.output type = %T, want []any", response["output"])
	}

	if len(output) != 1 {
		t.Fatalf("len(output) = %d, want 1", len(output))
	}

	item := output[0].(map[string]any)
	if item["type"] != "message" {
		t.Errorf("output[0].type = %q, want %q", item["type"], "message")
	}
}

func TestResponsesStreamConverter_ResponseCreatedIncludesOutput(t *testing.T) {
	// Verify that response.created includes an empty output array
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	events := converter.Process(api.ChatResponse{
		Message: api.Message{Content: "Hi"},
	})

	// First event should be response.created
	if events[0].Event != "response.created" {
		t.Fatalf("events[0].Event = %q, want %q", events[0].Event, "response.created")
	}

	data := events[0].Data.(map[string]any)
	response := data["response"].(map[string]any)

	output, ok := response["output"].([]any)
	if !ok {
		t.Fatalf("response.output type = %T, want []any", response["output"])
	}

	// Should be empty array initially
	if len(output) != 0 {
		t.Errorf("len(output) = %d, want 0", len(output))
	}
}

func TestResponsesStreamConverter_SequenceNumbers(t *testing.T) {
	// Verify that events include incrementing sequence numbers
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	events := converter.Process(api.ChatResponse{
		Message: api.Message{Content: "Hello"},
	})

	for i, event := range events {
		data := event.Data.(map[string]any)
		seqNum, ok := data["sequence_number"].(int)
		if !ok {
			t.Fatalf("events[%d] missing sequence_number", i)
		}
		if seqNum != i {
			t.Errorf("events[%d].sequence_number = %d, want %d", i, seqNum, i)
		}
	}

	// Process more content, sequence should continue
	moreEvents := converter.Process(api.ChatResponse{
		Message: api.Message{Content: " World"},
	})

	expectedSeq := len(events)
	for i, event := range moreEvents {
		data := event.Data.(map[string]any)
		seqNum := data["sequence_number"].(int)
		if seqNum != expectedSeq+i {
			t.Errorf("moreEvents[%d].sequence_number = %d, want %d", i, seqNum, expectedSeq+i)
		}
	}
}

func TestResponsesStreamConverter_FunctionCallStatus(t *testing.T) {
	// Verify that function call items include status field
	converter := NewResponsesStreamConverter("resp_123", "msg_456", "gpt-oss:20b", ResponsesRequest{})

	events := converter.Process(api.ChatResponse{
		Message: api.Message{
			ToolCalls: []api.ToolCall{
				{
					ID: "call_abc",
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: testArgs(map[string]any{"city": "Paris"}),
					},
				},
			},
		},
	})

	// Find output_item.added event
	var addedItem map[string]any
	var doneItem map[string]any
	for _, event := range events {
		data := event.Data.(map[string]any)
		if data["type"] == "response.output_item.added" {
			item := data["item"].(map[string]any)
			if item["type"] == "function_call" {
				addedItem = item
			}
		}
		if data["type"] == "response.output_item.done" {
			item := data["item"].(map[string]any)
			if item["type"] == "function_call" {
				doneItem = item
			}
		}
	}

	if addedItem == nil {
		t.Fatal("expected function_call output_item.added event")
	}
	if addedItem["status"] != "in_progress" {
		t.Errorf("output_item.added status = %q, want %q", addedItem["status"], "in_progress")
	}

	if doneItem == nil {
		t.Fatal("expected function_call output_item.done event")
	}
	if doneItem["status"] != "completed" {
		t.Errorf("output_item.done status = %q, want %q", doneItem["status"], "completed")
	}
}
