package openai

import (
	"encoding/base64"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

const (
	prefix = `data:image/jpeg;base64,`
	image  = `iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=`
)

func TestFromChatRequest_Basic(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := FromChatRequest(req)
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
}

func TestFromChatRequest_WithImage(t *testing.T) {
	imgData, _ := base64.StdEncoding.DecodeString(image)

	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{
				Role: "user",
				Content: []any{
					map[string]any{"type": "text", "text": "Hello"},
					map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": prefix + image},
					},
				},
			},
		},
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result.Messages))
	}

	if result.Messages[0].Content != "Hello" {
		t.Errorf("expected first message content 'Hello', got %q", result.Messages[0].Content)
	}

	if len(result.Messages[1].Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(result.Messages[1].Images))
	}

	if string(result.Messages[1].Images[0]) != string(imgData) {
		t.Error("image data mismatch")
	}
}

func TestFromCompleteRequest_Basic(t *testing.T) {
	temp := float32(0.8)
	req := CompletionRequest{
		Model:       "test-model",
		Prompt:      "Hello",
		Temperature: &temp,
	}

	result, err := FromCompleteRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Model != "test-model" {
		t.Errorf("expected model 'test-model', got %q", result.Model)
	}

	if result.Prompt != "Hello" {
		t.Errorf("expected prompt 'Hello', got %q", result.Prompt)
	}

	if tempVal, ok := result.Options["temperature"].(float32); !ok || tempVal != 0.8 {
		t.Errorf("expected temperature 0.8, got %v", result.Options["temperature"])
	}
}

func TestToUsage(t *testing.T) {
	resp := api.ChatResponse{
		Metrics: api.Metrics{
			PromptEvalCount: 10,
			EvalCount:       20,
		},
	}

	usage := ToUsage(resp)

	if usage.PromptTokens != 10 {
		t.Errorf("expected PromptTokens 10, got %d", usage.PromptTokens)
	}

	if usage.CompletionTokens != 20 {
		t.Errorf("expected CompletionTokens 20, got %d", usage.CompletionTokens)
	}

	if usage.TotalTokens != 30 {
		t.Errorf("expected TotalTokens 30, got %d", usage.TotalTokens)
	}
}

func TestNewError(t *testing.T) {
	tests := []struct {
		code int
		want string
	}{
		{400, "invalid_request_error"},
		{404, "not_found_error"},
		{500, "api_error"},
	}

	for _, tt := range tests {
		result := NewError(tt.code, "test message")
		if result.Error.Type != tt.want {
			t.Errorf("NewError(%d) type = %q, want %q", tt.code, result.Error.Type, tt.want)
		}
		if result.Error.Message != "test message" {
			t.Errorf("NewError(%d) message = %q, want %q", tt.code, result.Error.Message, "test message")
		}
	}
}

func TestToToolCallsPreservesIDs(t *testing.T) {
	original := []api.ToolCall{
		{
			ID: "call_abc123",
			Function: api.ToolCallFunction{
				Index: 2,
				Name:  "get_weather",
				Arguments: api.ToolCallFunctionArguments{
					"location": "Seattle",
				},
			},
		},
		{
			ID: "call_def456",
			Function: api.ToolCallFunction{
				Index: 7,
				Name:  "get_time",
				Arguments: api.ToolCallFunctionArguments{
					"timezone": "UTC",
				},
			},
		},
	}

	toolCalls := make([]api.ToolCall, len(original))
	copy(toolCalls, original)
	got := ToToolCalls(toolCalls)

	if len(got) != len(original) {
		t.Fatalf("expected %d tool calls, got %d", len(original), len(got))
	}

	expected := []ToolCall{
		{
			ID:    "call_abc123",
			Type:  "function",
			Index: 2,
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{
				Name:      "get_weather",
				Arguments: `{"location":"Seattle"}`,
			},
		},
		{
			ID:    "call_def456",
			Type:  "function",
			Index: 7,
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{
				Name:      "get_time",
				Arguments: `{"timezone":"UTC"}`,
			},
		},
	}

	if diff := cmp.Diff(expected, got); diff != "" {
		t.Errorf("tool calls mismatch (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff(original, toolCalls); diff != "" {
		t.Errorf("input tool calls mutated (-want +got):\n%s", diff)
	}
}

func TestFromChatRequest_WithLogprobs(t *testing.T) {
	trueVal := true

	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Logprobs:    &trueVal,
		TopLogprobs: 5,
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !result.Logprobs {
		t.Error("expected Logprobs to be true")
	}

	if result.TopLogprobs != 5 {
		t.Errorf("expected TopLogprobs to be 5, got %d", result.TopLogprobs)
	}
}

func TestFromChatRequest_LogprobsDefault(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "test-model",
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	result, err := FromChatRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Logprobs {
		t.Error("expected Logprobs to be false by default")
	}

	if result.TopLogprobs != 0 {
		t.Errorf("expected TopLogprobs to be 0 by default, got %d", result.TopLogprobs)
	}
}

func TestFromCompleteRequest_WithLogprobs(t *testing.T) {
	logprobsVal := 5

	req := CompletionRequest{
		Model:    "test-model",
		Prompt:   "Hello",
		Logprobs: &logprobsVal,
	}

	result, err := FromCompleteRequest(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !result.Logprobs {
		t.Error("expected Logprobs to be true")
	}

	if result.TopLogprobs != 5 {
		t.Errorf("expected TopLogprobs to be 5, got %d", result.TopLogprobs)
	}
}

func TestToChatCompletion_WithLogprobs(t *testing.T) {
	createdAt := time.Unix(1234567890, 0)
	resp := api.ChatResponse{
		Model:     "test-model",
		CreatedAt: createdAt,
		Message:   api.Message{Role: "assistant", Content: "Hello there"},
		Logprobs: []api.Logprob{
			{
				TokenLogprob: api.TokenLogprob{
					Token:   "Hello",
					Logprob: -0.5,
				},
				TopLogprobs: []api.TokenLogprob{
					{Token: "Hello", Logprob: -0.5},
					{Token: "Hi", Logprob: -1.2},
				},
			},
			{
				TokenLogprob: api.TokenLogprob{
					Token:   " there",
					Logprob: -0.3,
				},
				TopLogprobs: []api.TokenLogprob{
					{Token: " there", Logprob: -0.3},
					{Token: " world", Logprob: -1.5},
				},
			},
		},
		Done: true,
		Metrics: api.Metrics{
			PromptEvalCount: 5,
			EvalCount:       2,
		},
	}

	id := "test-id"

	result := ToChatCompletion(id, resp)

	if result.Id != id {
		t.Errorf("expected Id %q, got %q", id, result.Id)
	}

	if result.Created != 1234567890 {
		t.Errorf("expected Created %d, got %d", int64(1234567890), result.Created)
	}

	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}

	choice := result.Choices[0]
	if choice.Message.Content != "Hello there" {
		t.Errorf("expected content %q, got %q", "Hello there", choice.Message.Content)
	}

	if choice.Logprobs == nil {
		t.Fatal("expected Logprobs to be present")
	}

	if len(choice.Logprobs.Content) != 2 {
		t.Fatalf("expected 2 logprobs, got %d", len(choice.Logprobs.Content))
	}

	// Verify first logprob
	if choice.Logprobs.Content[0].Token != "Hello" {
		t.Errorf("expected first token %q, got %q", "Hello", choice.Logprobs.Content[0].Token)
	}
	if choice.Logprobs.Content[0].Logprob != -0.5 {
		t.Errorf("expected first logprob -0.5, got %f", choice.Logprobs.Content[0].Logprob)
	}
	if len(choice.Logprobs.Content[0].TopLogprobs) != 2 {
		t.Errorf("expected 2 top_logprobs, got %d", len(choice.Logprobs.Content[0].TopLogprobs))
	}

	// Verify second logprob
	if choice.Logprobs.Content[1].Token != " there" {
		t.Errorf("expected second token %q, got %q", " there", choice.Logprobs.Content[1].Token)
	}
}

func TestToChatCompletion_WithoutLogprobs(t *testing.T) {
	createdAt := time.Unix(1234567890, 0)
	resp := api.ChatResponse{
		Model:     "test-model",
		CreatedAt: createdAt,
		Message:   api.Message{Role: "assistant", Content: "Hello"},
		Done:      true,
		Metrics: api.Metrics{
			PromptEvalCount: 5,
			EvalCount:       1,
		},
	}

	id := "test-id"

	result := ToChatCompletion(id, resp)

	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}

	// When no logprobs, Logprobs should be nil
	if result.Choices[0].Logprobs != nil {
		t.Error("expected Logprobs to be nil when not requested")
	}
}

func TestFromChatRequest_TopLogprobsRange(t *testing.T) {
	tests := []struct {
		name        string
		topLogprobs int
		expectValid bool
	}{
		{name: "valid: 0", topLogprobs: 0, expectValid: true},
		{name: "valid: 1", topLogprobs: 1, expectValid: true},
		{name: "valid: 10", topLogprobs: 10, expectValid: true},
		{name: "valid: 20", topLogprobs: 20, expectValid: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trueVal := true
			req := ChatCompletionRequest{
				Model: "test-model",
				Messages: []Message{
					{Role: "user", Content: "Hello"},
				},
				Logprobs:    &trueVal,
				TopLogprobs: tt.topLogprobs,
			}

			result, err := FromChatRequest(req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if result.TopLogprobs != tt.topLogprobs {
				t.Errorf("expected TopLogprobs %d, got %d", tt.topLogprobs, result.TopLogprobs)
			}
		})
	}
}

func TestToolChoice_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		wantMode string
		wantObj  bool
	}{
		{
			name:     "string auto",
			json:     `"auto"`,
			wantMode: "auto",
			wantObj:  false,
		},
		{
			name:     "string none",
			json:     `"none"`,
			wantMode: "none",
			wantObj:  false,
		},
		{
			name:     "string required",
			json:     `"required"`,
			wantMode: "required",
			wantObj:  false,
		},
		{
			name:     "object function with name",
			json:     `{"type": "function", "name": "get_weather"}`,
			wantMode: "",
			wantObj:  true,
		},
		{
			name:     "object function with function.name",
			json:     `{"type": "function", "function": {"name": "get_weather"}}`,
			wantMode: "",
			wantObj:  true,
		},
		{
			name:     "object allowed_tools",
			json:     `{"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}`,
			wantMode: "",
			wantObj:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tc ToolChoice
			if err := tc.UnmarshalJSON([]byte(tt.json)); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tc.Mode != tt.wantMode {
				t.Errorf("Mode = %q, want %q", tc.Mode, tt.wantMode)
			}

			if (tc.Object != nil) != tt.wantObj {
				t.Errorf("Object = %v, wantObj = %v", tc.Object, tt.wantObj)
			}
		})
	}
}

func TestToolChoice_Methods(t *testing.T) {
	tests := []struct {
		name             string
		toolChoice       *ToolChoice
		isNone           bool
		isRequired       bool
		isAuto           bool
		isForcedFunction bool
		forcedFuncName   string
		isAllowedTools   bool
		allowedToolNames []string
		allowedToolsMode string
	}{
		{
			name:       "nil",
			toolChoice: nil,
			isNone:     false,
			isRequired: false,
			isAuto:     true,
		},
		{
			name:       "auto string",
			toolChoice: &ToolChoice{Mode: "auto"},
			isNone:     false,
			isRequired: false,
			isAuto:     true,
		},
		{
			name:       "none string",
			toolChoice: &ToolChoice{Mode: "none"},
			isNone:     true,
			isRequired: false,
			isAuto:     false,
		},
		{
			name:       "required string",
			toolChoice: &ToolChoice{Mode: "required"},
			isNone:     false,
			isRequired: true,
			isAuto:     false,
		},
		{
			name: "forced function with name",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{Type: "function", Name: "get_weather"},
			},
			isNone:           false,
			isRequired:       false,
			isAuto:           false,
			isForcedFunction: true,
			forcedFuncName:   "get_weather",
		},
		{
			name: "forced function with function.name",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{
					Type:     "function",
					Function: &ToolChoiceFunctionRef{Name: "search"},
				},
			},
			isNone:           false,
			isRequired:       false,
			isAuto:           false,
			isForcedFunction: true,
			forcedFuncName:   "search",
		},
		{
			name: "allowed_tools auto",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{
					Type: "allowed_tools",
					Mode: "auto",
					Tools: []ToolChoiceAllowedTool{
						{Type: "function", Name: "get_weather"},
						{Type: "function", Name: "search"},
					},
				},
			},
			isNone:           false,
			isRequired:       false,
			isAuto:           true,
			isForcedFunction: false,
			isAllowedTools:   true,
			allowedToolNames: []string{"get_weather", "search"},
			allowedToolsMode: "auto",
		},
		{
			name: "allowed_tools required",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{
					Type: "allowed_tools",
					Mode: "required",
					Tools: []ToolChoiceAllowedTool{
						{Type: "function", Name: "get_weather"},
					},
				},
			},
			isNone:           false,
			isRequired:       false,
			isAuto:           false,
			isForcedFunction: false,
			isAllowedTools:   true,
			allowedToolNames: []string{"get_weather"},
			allowedToolsMode: "required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.toolChoice.IsNone(); got != tt.isNone {
				t.Errorf("IsNone() = %v, want %v", got, tt.isNone)
			}
			if got := tt.toolChoice.IsRequired(); got != tt.isRequired {
				t.Errorf("IsRequired() = %v, want %v", got, tt.isRequired)
			}
			if got := tt.toolChoice.IsAuto(); got != tt.isAuto {
				t.Errorf("IsAuto() = %v, want %v", got, tt.isAuto)
			}
			if got := tt.toolChoice.IsForcedFunction(); got != tt.isForcedFunction {
				t.Errorf("IsForcedFunction() = %v, want %v", got, tt.isForcedFunction)
			}
			if got := tt.toolChoice.GetForcedFunctionName(); got != tt.forcedFuncName {
				t.Errorf("GetForcedFunctionName() = %q, want %q", got, tt.forcedFuncName)
			}
			if got := tt.toolChoice.IsAllowedTools(); got != tt.isAllowedTools {
				t.Errorf("IsAllowedTools() = %v, want %v", got, tt.isAllowedTools)
			}
			if tt.isAllowedTools {
				if got := tt.toolChoice.GetAllowedToolNames(); !cmp.Equal(got, tt.allowedToolNames) {
					t.Errorf("GetAllowedToolNames() = %v, want %v", got, tt.allowedToolNames)
				}
				if got := tt.toolChoice.GetAllowedToolsMode(); got != tt.allowedToolsMode {
					t.Errorf("GetAllowedToolsMode() = %q, want %q", got, tt.allowedToolsMode)
				}
			}
		})
	}
}

func TestApplyToolChoice(t *testing.T) {
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"location": {Type: []string{"string"}},
					},
					Required: []string{"location"},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "search",
				Description: "Search the web",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"query": {Type: []string{"string"}},
					},
					Required: []string{"query"},
				},
			},
		},
	}

	tests := []struct {
		name          string
		toolChoice    *ToolChoice
		wantToolCount int
		wantFormat    bool
		wantForced    bool
		wantError     bool
		wantToolNames []string
	}{
		{
			name:          "nil (auto)",
			toolChoice:    nil,
			wantToolCount: 2,
			wantFormat:    false,
			wantForced:    false,
		},
		{
			name:          "auto string",
			toolChoice:    &ToolChoice{Mode: "auto"},
			wantToolCount: 2,
			wantFormat:    false,
			wantForced:    false,
		},
		{
			name:          "none string",
			toolChoice:    &ToolChoice{Mode: "none"},
			wantToolCount: 0,
			wantFormat:    false,
			wantForced:    false,
		},
		{
			name:          "required string",
			toolChoice:    &ToolChoice{Mode: "required"},
			wantToolCount: 2,
			wantFormat:    true,
			wantForced:    true,
		},
		{
			name: "forced function",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{Type: "function", Name: "get_weather"},
			},
			wantToolCount: 1,
			wantFormat:    true,
			wantForced:    true,
			wantToolNames: []string{"get_weather"},
		},
		{
			name: "forced unknown function",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{Type: "function", Name: "unknown_func"},
			},
			wantError: true,
		},
		{
			name: "allowed_tools auto",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{
					Type: "allowed_tools",
					Mode: "auto",
					Tools: []ToolChoiceAllowedTool{
						{Type: "function", Name: "get_weather"},
					},
				},
			},
			wantToolCount: 1,
			wantFormat:    false,
			wantForced:    false,
			wantToolNames: []string{"get_weather"},
		},
		{
			name: "allowed_tools required",
			toolChoice: &ToolChoice{
				Object: &ToolChoiceObject{
					Type: "allowed_tools",
					Mode: "required",
					Tools: []ToolChoiceAllowedTool{
						{Type: "function", Name: "search"},
					},
				},
			},
			wantToolCount: 1,
			wantFormat:    true,
			wantForced:    true,
			wantToolNames: []string{"search"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filteredTools, format, forced, err := ApplyToolChoice(tools, tt.toolChoice)

			if tt.wantError {
				if err == nil {
					t.Errorf("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(filteredTools) != tt.wantToolCount {
				t.Errorf("got %d tools, want %d", len(filteredTools), tt.wantToolCount)
			}

			if (format != nil) != tt.wantFormat {
				t.Errorf("format = %v, wantFormat = %v", format != nil, tt.wantFormat)
			}

			if forced != tt.wantForced {
				t.Errorf("forced = %v, want %v", forced, tt.wantForced)
			}

			if tt.wantToolNames != nil {
				gotNames := make([]string, len(filteredTools))
				for i, tool := range filteredTools {
					gotNames[i] = tool.Function.Name
				}
				if !cmp.Equal(gotNames, tt.wantToolNames) {
					t.Errorf("tool names = %v, want %v", gotNames, tt.wantToolNames)
				}
			}
		})
	}
}

func TestFromChatRequest_WithToolChoice(t *testing.T) {
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"location": {Type: []string{"string"}},
					},
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "search",
				Description: "Search",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: map[string]api.ToolProperty{
						"query": {Type: []string{"string"}},
					},
				},
			},
		},
	}

	t.Run("tool_choice none removes tools", func(t *testing.T) {
		req := ChatCompletionRequest{
			Model:      "test-model",
			Messages:   []Message{{Role: "user", Content: "Hello"}},
			Tools:      tools,
			ToolChoice: &ToolChoice{Mode: "none"},
		}

		result, err := FromChatRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Tools) != 0 {
			t.Errorf("expected 0 tools, got %d", len(result.Tools))
		}
	})

	t.Run("tool_choice required adds format", func(t *testing.T) {
		req := ChatCompletionRequest{
			Model:      "test-model",
			Messages:   []Message{{Role: "user", Content: "Hello"}},
			Tools:      tools,
			ToolChoice: &ToolChoice{Mode: "required"},
		}

		result, err := FromChatRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Tools) != 2 {
			t.Errorf("expected 2 tools, got %d", len(result.Tools))
		}

		if result.Format == nil {
			t.Error("expected format to be set for required tool_choice")
		}
	})

	t.Run("tool_choice forced function filters and adds format", func(t *testing.T) {
		req := ChatCompletionRequest{
			Model:    "test-model",
			Messages: []Message{{Role: "user", Content: "Hello"}},
			Tools:    tools,
			ToolChoice: &ToolChoice{
				Object: &ToolChoiceObject{Type: "function", Name: "get_weather"},
			},
		}

		result, err := FromChatRequest(req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if len(result.Tools) != 1 {
			t.Errorf("expected 1 tool, got %d", len(result.Tools))
		}

		if result.Tools[0].Function.Name != "get_weather" {
			t.Errorf("expected tool 'get_weather', got %q", result.Tools[0].Function.Name)
		}

		if result.Format == nil {
			t.Error("expected format to be set for forced function")
		}
	})

	t.Run("tool_choice unknown function returns error", func(t *testing.T) {
		req := ChatCompletionRequest{
			Model:    "test-model",
			Messages: []Message{{Role: "user", Content: "Hello"}},
			Tools:    tools,
			ToolChoice: &ToolChoice{
				Object: &ToolChoiceObject{Type: "function", Name: "unknown"},
			},
		}

		_, err := FromChatRequest(req)
		if err == nil {
			t.Error("expected error for unknown function")
		}
	})
}
