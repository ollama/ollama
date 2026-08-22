package openai

import (
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests)
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

// argsComparer provides cmp options for comparing ToolCallFunctionArguments by value
var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	return cmp.Equal(a.ToMap(), b.ToMap())
})

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

func TestFromChatRequest_ReasoningEffort(t *testing.T) {
	effort := func(s string) *string { return &s }

	cases := []struct {
		name    string
		effort  *string
		want    any // expected ThinkValue.Value; nil means req.Think should be nil
		wantErr bool
	}{
		{name: "unset", effort: nil, want: nil},
		{name: "high", effort: effort("high"), want: "high"},
		{name: "medium", effort: effort("medium"), want: "medium"},
		{name: "low", effort: effort("low"), want: "low"},
		{name: "max", effort: effort("max"), want: "max"},
		{name: "minimal clamps to low", effort: effort("minimal"), want: "low"},
		{name: "xhigh clamps to max", effort: effort("xhigh"), want: "max"},
		{name: "ultra clamps to max", effort: effort("ultra"), want: "max"},
		{name: "none disables", effort: effort("none"), want: false},
		{name: "invalid", effort: effort("extreme"), wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := ChatCompletionRequest{
				Model:           "test-model",
				Messages:        []Message{{Role: "user", Content: "hi"}},
				ReasoningEffort: tc.effort,
			}
			result, err := FromChatRequest(req)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error for effort=%v, got none", *tc.effort)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.want == nil {
				if result.Think != nil {
					t.Fatalf("expected nil Think, got %+v", result.Think)
				}
				return
			}
			if result.Think == nil {
				t.Fatalf("expected Think=%v, got nil", tc.want)
			}
			if result.Think.Value != tc.want {
				t.Fatalf("got Think.Value=%v, want %v", result.Think.Value, tc.want)
			}
		})
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

func TestFromCompleteRequest_TopP(t *testing.T) {
	t.Run("explicit zero is preserved", func(t *testing.T) {
		topP := float32(0)
		result, err := FromCompleteRequest(CompletionRequest{Model: "test-model", TopP: &topP})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if v, ok := result.Options["top_p"].(float32); !ok || v != 0 {
			t.Errorf("expected top_p 0, got %v", result.Options["top_p"])
		}
	})

	t.Run("unset defaults to 1", func(t *testing.T) {
		result, err := FromCompleteRequest(CompletionRequest{Model: "test-model"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if v, ok := result.Options["top_p"].(float64); !ok || v != 1.0 {
			t.Errorf("expected top_p 1.0, got %v", result.Options["top_p"])
		}
	})
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
				Arguments: testArgs(map[string]any{
					"location": "Seattle",
				}),
			},
		},
		{
			ID: "call_def456",
			Function: api.ToolCallFunction{
				Index: 7,
				Name:  "get_time",
				Arguments: testArgs(map[string]any{
					"timezone": "UTC",
				}),
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

	if diff := cmp.Diff(original, toolCalls, argsComparer); diff != "" {
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

func TestToListCompletionUsesModelIdentity(t *testing.T) {
	modified := time.Unix(1234567890, 0).UTC()

	result := ToListCompletion(api.ListResponse{
		Models: []api.ListModelResponse{
			{
				Name:       "legacy-name:latest",
				Model:      "namespace/exposed-model:latest",
				ModifiedAt: modified,
			},
			{
				Name:       "fallback-name:latest",
				ModifiedAt: modified.Add(time.Second),
			},
		},
	})

	if result.Object != "list" {
		t.Fatalf("object = %q, want list", result.Object)
	}
	if len(result.Data) != 2 {
		t.Fatalf("models = %d, want 2", len(result.Data))
	}

	if result.Data[0].Id != "namespace/exposed-model:latest" {
		t.Fatalf("id = %q, want model field", result.Data[0].Id)
	}
	if result.Data[0].OwnedBy != "namespace" {
		t.Fatalf("owned_by = %q, want namespace", result.Data[0].OwnedBy)
	}
	if result.Data[0].Created != modified.Unix() {
		t.Fatalf("created = %d, want %d", result.Data[0].Created, modified.Unix())
	}

	if result.Data[1].Id != "fallback-name:latest" {
		t.Fatalf("fallback id = %q, want name field", result.Data[1].Id)
	}
	if result.Data[1].OwnedBy != "library" {
		t.Fatalf("fallback owned_by = %q, want library", result.Data[1].OwnedBy)
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

func TestToStreamChunks_SplitsThinkingAndContent(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "step-by-step",
			Content:  "final answer",
		},
		Done:       true,
		DoneReason: "stop",
	}

	chunks := ToStreamChunks("test-id", resp, true)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "step-by-step" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if reasoning.Delta.Content != nil {
		t.Fatalf("expected reasoning chunk content to be nil, got %v", reasoning.Delta.Content)
	}
	if len(reasoning.Delta.ToolCalls) != 0 {
		t.Fatalf("expected reasoning chunk tool calls to be empty, got %d", len(reasoning.Delta.ToolCalls))
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason to be nil, got %q", *reasoning.FinishReason)
	}
	if reasoning.Delta.Role != "assistant" {
		t.Fatalf("expected reasoning chunk role %q, got %q", "assistant", reasoning.Delta.Role)
	}

	content := chunks[1].Choices[0]
	if content.Delta.Reasoning != "" {
		t.Fatalf("expected content chunk reasoning to be empty, got %q", content.Delta.Reasoning)
	}
	if content.Delta.Content != "final answer" {
		t.Fatalf("expected content chunk content %q, got %v", "final answer", content.Delta.Content)
	}
	if content.FinishReason != nil {
		t.Fatalf("expected content chunk finish reason to be nil, got %v", content.FinishReason)
	}
	if content.Delta.Role != "" {
		t.Fatalf("expected content chunk role to be empty, got %q", content.Delta.Role)
	}
}

func TestToStreamChunks_SplitsThinkingAndToolCalls(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "need a tool",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_123",
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "Seattle",
						}),
					},
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
	}

	chunks := ToStreamChunks("test-id", resp, true)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "need a tool" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if len(reasoning.Delta.ToolCalls) != 0 {
		t.Fatalf("expected reasoning chunk tool calls to be empty, got %d", len(reasoning.Delta.ToolCalls))
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason to be nil, got %q", *reasoning.FinishReason)
	}

	toolCallChunk := chunks[1].Choices[0]
	if toolCallChunk.Delta.Reasoning != "" {
		t.Fatalf("expected tool-call chunk reasoning to be empty, got %q", toolCallChunk.Delta.Reasoning)
	}
	if len(toolCallChunk.Delta.ToolCalls) != 1 {
		t.Fatalf("expected one tool call in second chunk, got %d", len(toolCallChunk.Delta.ToolCalls))
	}
	if toolCallChunk.Delta.ToolCalls[0].ID != "call_123" {
		t.Fatalf("expected tool call id %q, got %q", "call_123", toolCallChunk.Delta.ToolCalls[0].ID)
	}
	if toolCallChunk.FinishReason != nil {
		t.Fatalf("expected tool-call chunk finish reason to be nil, got %v", toolCallChunk.FinishReason)
	}
}

func TestToStreamChunks_SingleChunkForNonMixedResponses(t *testing.T) {
	toolCalls := []api.ToolCall{
		{
			ID: "call_456",
			Function: api.ToolCallFunction{
				Index: 0,
				Name:  "get_time",
				Arguments: testArgs(map[string]any{
					"timezone": "UTC",
				}),
			},
		},
	}

	tests := []struct {
		name    string
		message api.Message
	}{
		{
			name:    "thinking-only",
			message: api.Message{Thinking: "pondering"},
		},
		{
			name:    "content-only",
			message: api.Message{Content: "hello"},
		},
		{
			name:    "toolcalls-only",
			message: api.Message{ToolCalls: toolCalls},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := api.ChatResponse{
				Model:   "test-model",
				Message: tt.message,
			}

			chunks := ToStreamChunks("test-id", resp, true)
			if len(chunks) != 1 {
				t.Fatalf("expected 1 chunk, got %d", len(chunks))
			}
		})
	}
}

func TestToStreamChunks_SplitsThinkingAndToolCallsWhenNotDone(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "need a tool",
			ToolCalls: []api.ToolCall{
				{
					ID: "call_789",
					Function: api.ToolCallFunction{
						Index: 0,
						Name:  "get_weather",
						Arguments: testArgs(map[string]any{
							"location": "San Francisco",
						}),
					},
				},
			},
		},
		Done: false,
	}

	chunks := ToStreamChunks("test-id", resp, true)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "need a tool" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason nil, got %v", reasoning.FinishReason)
	}

	toolCallChunk := chunks[1].Choices[0]
	if len(toolCallChunk.Delta.ToolCalls) != 1 {
		t.Fatalf("expected one tool call in second chunk, got %d", len(toolCallChunk.Delta.ToolCalls))
	}
	if toolCallChunk.Delta.ToolCalls[0].ID != "call_789" {
		t.Fatalf("expected tool call id %q, got %q", "call_789", toolCallChunk.Delta.ToolCalls[0].ID)
	}
	if toolCallChunk.FinishReason != nil {
		t.Fatalf("expected tool-call chunk finish reason nil when not done, got %v", toolCallChunk.FinishReason)
	}
}

func TestToStreamChunks_SplitsThinkingAndContentWhenNotDone(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "thinking",
			Content:  "partial content",
		},
		Done: false,
	}

	chunks := ToStreamChunks("test-id", resp, true)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	reasoning := chunks[0].Choices[0]
	if reasoning.Delta.Reasoning != "thinking" {
		t.Fatalf("expected reasoning chunk to contain thinking, got %q", reasoning.Delta.Reasoning)
	}
	if reasoning.FinishReason != nil {
		t.Fatalf("expected reasoning chunk finish reason nil, got %v", reasoning.FinishReason)
	}

	content := chunks[1].Choices[0]
	if content.Delta.Content != "partial content" {
		t.Fatalf("expected content chunk content %q, got %v", "partial content", content.Delta.Content)
	}
	if content.FinishReason != nil {
		t.Fatalf("expected content chunk finish reason nil when not done, got %v", content.FinishReason)
	}
}

func TestToStreamChunks_SplitSendsLogprobsOnlyOnFirstChunk(t *testing.T) {
	resp := api.ChatResponse{
		Model: "test-model",
		Message: api.Message{
			Thinking: "thinking",
			Content:  "content",
		},
		Logprobs: []api.Logprob{
			{
				TokenLogprob: api.TokenLogprob{
					Token:   "tok",
					Logprob: -0.25,
				},
			},
		},
		Done:       true,
		DoneReason: "stop",
	}

	chunks := ToStreamChunks("test-id", resp, true)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}

	first := chunks[0].Choices[0]
	if first.Logprobs == nil {
		t.Fatal("expected first chunk to include logprobs")
	}
	if len(first.Logprobs.Content) != 1 || first.Logprobs.Content[0].Token != "tok" {
		t.Fatalf("unexpected first chunk logprobs: %+v", first.Logprobs.Content)
	}

	second := chunks[1].Choices[0]
	if second.Logprobs != nil {
		t.Fatalf("expected second chunk logprobs to be nil, got %+v", second.Logprobs)
	}
}

func TestFinishChunk(t *testing.T) {
	tests := []struct {
		name           string
		doneReason     string
		toolCallSent   bool
		expectedReason string
	}{
		{
			name:           "stop",
			doneReason:     "stop",
			toolCallSent:   false,
			expectedReason: "stop",
		},
		{
			name:           "length",
			doneReason:     "length",
			toolCallSent:   false,
			expectedReason: "length",
		},
		{
			name:           "tool_calls",
			doneReason:     "stop",
			toolCallSent:   true,
			expectedReason: "tool_calls",
		},
		{
			name:           "length_with_tool_calls",
			doneReason:     "length",
			toolCallSent:   true,
			expectedReason: "length",
		},
		{
			name:           "empty_reason_defaults_to_stop",
			doneReason:     "",
			toolCallSent:   false,
			expectedReason: "stop",
		},
		{
			name:           "unknown_reason_passes_through",
			doneReason:     "unload",
			toolCallSent:   false,
			expectedReason: "unload",
		},
		{
			name:           "unknown_reason_not_relabeled_tool_calls",
			doneReason:     "unload",
			toolCallSent:   true,
			expectedReason: "unload",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunk := FinishChunk("test-id", api.ChatResponse{Model: "test-model", DoneReason: tt.doneReason}, tt.toolCallSent)
			if len(chunk.Choices) != 1 {
				t.Fatalf("expected 1 choice, got %d", len(chunk.Choices))
			}
			choice := chunk.Choices[0]
			if choice.Delta.Content != nil || choice.Delta.Reasoning != "" || len(choice.Delta.ToolCalls) != 0 || choice.Delta.Role != "" {
				t.Fatalf("expected empty delta, got %+v", choice.Delta)
			}
			if choice.FinishReason == nil || *choice.FinishReason != tt.expectedReason {
				t.Fatalf("expected finish reason %q, got %v", tt.expectedReason, choice.FinishReason)
			}
		})
	}
}

func TestFinishChunk_JSONDeltaEmpty(t *testing.T) {
	chunk := FinishChunk("test-id", api.ChatResponse{Model: "test-model", DoneReason: "stop"}, false)
	d, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	// Parse back as generic JSON to inspect the delta field
	var raw map[string]any
	if err := json.Unmarshal(d, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	choices := raw["choices"].([]any)
	choice := choices[0].(map[string]any)
	delta := choice["delta"].(map[string]any)

	// The delta must be completely empty {} — no role, content, or other fields
	if len(delta) != 0 {
		t.Fatalf("expected empty delta {}, got %v", delta)
	}

	if choice["finish_reason"] != "stop" {
		t.Fatalf("expected finish_reason %q, got %v", "stop", choice["finish_reason"])
	}
}

func TestToStreamChunks_RoleOnlyWhenRequested(t *testing.T) {
	resp := api.ChatResponse{
		Model:   "test-model",
		Message: api.Message{Content: "hello"},
	}

	// With includeRole=true, delta should have role
	withRole := ToStreamChunks("test-id", resp, true)
	if withRole[0].Choices[0].Delta.Role != "assistant" {
		t.Fatalf("expected role %q, got %q", "assistant", withRole[0].Choices[0].Delta.Role)
	}

	// With includeRole=false, delta should omit role
	withoutRole := ToStreamChunks("test-id", resp, false)
	if withoutRole[0].Choices[0].Delta.Role != "" {
		t.Fatalf("expected empty role, got %q", withoutRole[0].Choices[0].Delta.Role)
	}
}

func TestToStreamChunks_ContentChunkJSON(t *testing.T) {
	resp := api.ChatResponse{
		Model:   "test-model",
		Message: api.Message{Content: "Hi"},
	}

	chunks := ToStreamChunks("test-id", resp, false)
	d, err := json.Marshal(chunks[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(d, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices := raw["choices"].([]any)
	delta := choices[0].(map[string]any)["delta"].(map[string]any)

	// Content should be present
	if delta["content"] != "Hi" {
		t.Fatalf("expected content %q, got %v", "Hi", delta["content"])
	}
	// Role should be absent (includeRole=false)
	if _, hasRole := delta["role"]; hasRole {
		t.Fatalf("expected role to be absent, got %v", delta["role"])
	}
	// Reasoning should be absent
	if _, hasReasoning := delta["reasoning"]; hasReasoning {
		t.Fatalf("expected reasoning to be absent, got %v", delta["reasoning"])
	}
}

func TestToStreamChunks_EmptyContentChunkJSON(t *testing.T) {
	resp := api.ChatResponse{
		Model:   "test-model",
		Message: api.Message{Content: ""},
	}

	chunks := ToStreamChunks("test-id", resp, true)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}

	d, err := json.Marshal(chunks[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(d, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	delta := raw["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)

	// Empty-string content must serialize explicitly as "content":"" (OpenAI's
	// first chunk is {"role":"assistant","content":""}); only nil content is omitted.
	content, hasContent := delta["content"]
	if !hasContent {
		t.Fatalf("expected content key to be present for empty-string content, got %v", delta)
	}
	if content != "" {
		t.Fatalf("expected content %q, got %v", "", content)
	}
	if delta["role"] != "assistant" {
		t.Fatalf("expected role %q, got %v", "assistant", delta["role"])
	}
	if _, hasReasoning := delta["reasoning"]; hasReasoning {
		t.Fatalf("expected reasoning to be absent, got %v", delta["reasoning"])
	}
}

func TestToChatCompletion_FinishReasonPrecedence(t *testing.T) {
	newToolCallResponse := func(doneReason string) api.ChatResponse {
		return api.ChatResponse{
			Model: "test-model",
			Message: api.Message{
				Role: "assistant",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_123",
						Function: api.ToolCallFunction{
							Index: 0,
							Name:  "get_weather",
							Arguments: testArgs(map[string]any{
								"location": "Seattle",
							}),
						},
					},
				},
			},
			Done:       true,
			DoneReason: doneReason,
		}
	}

	// A truncated tool-call response keeps "length" rather than "tool_calls".
	if got := *ToChatCompletion("test-id", newToolCallResponse("length")).Choices[0].FinishReason; got != "length" {
		t.Fatalf("expected finish reason %q for truncated tool-call response, got %q", "length", got)
	}
	// A completed tool-call response reports "tool_calls".
	if got := *ToChatCompletion("test-id", newToolCallResponse("stop")).Choices[0].FinishReason; got != "tool_calls" {
		t.Fatalf("expected finish reason %q for completed tool-call response, got %q", "tool_calls", got)
	}
	// An unrelated finish reason passes through unchanged.
	if got := *ToChatCompletion("test-id", newToolCallResponse("unload")).Choices[0].FinishReason; got != "unload" {
		t.Fatalf("expected unknown finish reason %q to pass through, got %q", "unload", got)
	}
}

func TestFinishChunk_UsesResponseCreatedAt(t *testing.T) {
	resp := api.ChatResponse{
		Model:      "test-model",
		DoneReason: "stop",
		CreatedAt:  time.Unix(1700000000, 0),
	}
	if got := FinishChunk("test-id", resp, false).Created; got != 1700000000 {
		t.Fatalf("expected created %d, got %d", 1700000000, got)
	}

	// A zero CreatedAt falls back to the current time.
	zero := api.ChatResponse{Model: "test-model", DoneReason: "stop"}
	if got := FinishChunk("test-id", zero, false).Created; got <= 1700000000 {
		t.Fatalf("expected fallback created to be the current time, got %d", got)
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
