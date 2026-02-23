package renderers

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
)

func TestLFM2Renderer_ChatTemplateParity(t *testing.T) {
	tests := []struct {
		name       string
		renderer   *LFM2Renderer
		messages   []api.Message
		tools      []api.Tool
		thinkValue *api.ThinkValue
		expected   string
	}{
		{
			name:     "user_only",
			renderer: &LFM2Renderer{IsThinking: false},
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "system_and_user",
			renderer: &LFM2Renderer{IsThinking: false},
			messages: []api.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hi"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "tools_without_system",
			renderer: &LFM2Renderer{IsThinking: false},
			messages: []api.Message{
				{Role: "user", Content: "Use tools"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "get_weather",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: "<|startoftext|><|im_start|>system\nList of tools: <|tool_list_start|>[{\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": null}}]<|tool_list_end|><|im_end|>\n" +
				"<|im_start|>user\nUse tools<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "first_system_combined_with_tools",
			renderer: &LFM2Renderer{IsThinking: false},
			messages: []api.Message{
				{Role: "system", Content: "Follow instructions."},
				{Role: "user", Content: "Do work"},
			},
			tools: []api.Tool{
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "tool_a",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
						},
					},
				},
				{
					Type: "function",
					Function: api.ToolFunction{
						Name: "tool_b",
						Parameters: api.ToolFunctionParameters{
							Type: "object",
						},
					},
				},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected: "<|startoftext|><|im_start|>system\nFollow instructions.\nList of tools: <|tool_list_start|>[{\"name\": \"tool_a\", \"parameters\": {\"type\": \"object\", \"properties\": null}}, {\"name\": \"tool_b\", \"parameters\": {\"type\": \"object\", \"properties\": null}}]<|tool_list_end|><|im_end|>\n" +
				"<|im_start|>user\nDo work<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "assistant_tool_calls_and_tool_responses_are_rendered",
			renderer: &LFM2Renderer{IsThinking: false},
			messages: []api.Message{
				{Role: "user", Content: "Call a tool"},
				{
					Role:    "assistant",
					Content: "",
					ToolCalls: []api.ToolCall{
						{
							Function: api.ToolCallFunction{
								Name: "get_weather",
								Arguments: testArgs(map[string]any{
									"location": "Paris",
								}),
							},
						},
					},
				},
				{Role: "tool", Content: "22C"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|startoftext|><|im_start|>user\nCall a tool<|im_end|>\n<|im_start|>assistant\n<|tool_call_start|>[get_weather(location=\"Paris\")]<|tool_call_end|><|im_end|>\n<|im_start|>tool\n<|tool_response_start|>22C<|tool_response_end|><|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "thinking_strips_non_last_assistant_when_disabled",
			renderer: &LFM2Renderer{IsThinking: true},
			messages: []api.Message{
				{Role: "user", Content: "Q1"},
				{Role: "assistant", Content: "<think>reason1</think>A1"},
				{Role: "user", Content: "Q2"},
				{Role: "assistant", Content: "<think>reason2</think>A2"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|startoftext|><|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>reason2</think>A2",
		},
		{
			name:     "thinking_preserves_past_assistant_when_enabled",
			renderer: &LFM2Renderer{IsThinking: true},
			messages: []api.Message{
				{Role: "user", Content: "Q1"},
				{Role: "assistant", Content: "<think>reason1</think>A1"},
				{Role: "user", Content: "Q2"},
				{Role: "assistant", Content: "<think>reason2</think>A2"},
			},
			thinkValue: &api.ThinkValue{Value: true},
			expected:   "<|startoftext|><|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>reason1</think>A1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>reason2</think>A2",
		},
		{
			name:     "arbitrary_roles_are_rendered_verbatim",
			renderer: &LFM2Renderer{IsThinking: false},
			messages: []api.Message{
				{Role: "developer", Content: "Do X"},
				{Role: "user", Content: "Hi"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|startoftext|><|im_start|>developer\nDo X<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:       "empty_messages_still_add_generation_prompt",
			renderer:   &LFM2Renderer{IsThinking: false},
			messages:   nil,
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|startoftext|><|im_start|>assistant\n",
		},
		{
			name:     "assistant_prefill_no_generation_prompt",
			renderer: &LFM2Renderer{IsThinking: false},
			messages: []api.Message{
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello"},
			},
			thinkValue: &api.ThinkValue{Value: false},
			expected:   "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rendered, err := tt.renderer.Render(tt.messages, tt.tools, tt.thinkValue)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			if diff := cmp.Diff(tt.expected, rendered); diff != "" {
				t.Fatalf("Render() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLFM2Renderer_Images(t *testing.T) {
	tests := []struct {
		name     string
		renderer *LFM2Renderer
		message  api.Message
		expected string
	}{
		{
			name:     "single_image_default_placeholder",
			renderer: &LFM2Renderer{},
			message: api.Message{
				Role:    "user",
				Content: "Describe this image.",
				Images:  []api.ImageData{api.ImageData("img1")},
			},
			expected: "<|startoftext|><|im_start|>user\n<image>Describe this image.<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "multiple_images_default_placeholder",
			renderer: &LFM2Renderer{},
			message: api.Message{
				Role:    "user",
				Content: "Describe these images.",
				Images:  []api.ImageData{api.ImageData("img1"), api.ImageData("img2")},
			},
			expected: "<|startoftext|><|im_start|>user\n<image><image>Describe these images.<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "single_image_img_tag_placeholder",
			renderer: &LFM2Renderer{useImgTags: true},
			message: api.Message{
				Role:    "user",
				Content: "Describe this image.",
				Images:  []api.ImageData{api.ImageData("img1")},
			},
			expected: "<|startoftext|><|im_start|>user\n[img]Describe this image.<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "existing_indexed_img_placeholder_not_duplicated",
			renderer: &LFM2Renderer{useImgTags: true},
			message: api.Message{
				Role:    "user",
				Content: "[img-0]Describe this image.",
				Images:  []api.ImageData{api.ImageData("img1")},
			},
			expected: "<|startoftext|><|im_start|>user\n[img-0]Describe this image.<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "existing_template_image_placeholder_not_duplicated",
			renderer: &LFM2Renderer{},
			message: api.Message{
				Role:    "user",
				Content: "<image>Describe this image.",
				Images:  []api.ImageData{api.ImageData("img1")},
			},
			expected: "<|startoftext|><|im_start|>user\n<image>Describe this image.<|im_end|>\n<|im_start|>assistant\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.renderer.Render([]api.Message{tt.message}, nil, nil)
			if err != nil {
				t.Fatalf("Render() error = %v", err)
			}
			if diff := cmp.Diff(tt.expected, got); diff != "" {
				t.Fatalf("Render() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestLFM2Renderer_JSONFormatting(t *testing.T) {
	tool := api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        "echo",
			Description: "<html>",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
			},
		},
	}

	got := lfm2JSON(tool)
	want := "{\"type\": \"function\", \"function\": {\"name\": \"echo\", \"description\": \"<html>\", \"parameters\": {\"type\": \"object\", \"properties\": null}}}"
	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("lfm2JSON mismatch (-want +got):\n%s", diff)
	}
}
