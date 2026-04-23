package server

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func testConfigWithRenderer(renderer string) model.ConfigV2 {
	return model.ConfigV2{Renderer: renderer}
}

func testConfigWithRendererAndType(renderer, modelType string) model.ConfigV2 {
	return model.ConfigV2{Renderer: renderer, ModelType: modelType}
}

func TestChatPrompt(t *testing.T) {
	type expect struct {
		prompt string
		images [][]byte
		error  error
	}

	tmpl, err := template.Parse(`
{{- if .System }}{{ .System }} {{ end }}
{{- if .Prompt }}{{ .Prompt }} {{ end }}
{{- if .Response }}{{ .Response }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	visionModel := Model{Template: tmpl, ProjectorPaths: []string{"vision"}}

	cases := []struct {
		name     string
		model    Model
		limit    int
		truncate bool
		msgs     []api.Message
		expect
	}{
		{
			name:     "messages",
			model:    visionModel,
			limit:    64,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "truncate messages",
			model:    visionModel,
			limit:    1,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "truncate messages with image",
			model:    visionModel,
			limit:    64,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("something")}},
			},
			expect: expect{
				prompt: "[img-0]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
				},
			},
		},
		{
			name:     "truncate messages with images",
			model:    visionModel,
			limit:    64,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "[img-0]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "messages with images",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "[img-0]You're a test, Harry! I-I'm a what? [img-1]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "message with image tag",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry! [img]", Images: []api.ImageData{[]byte("something")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager.", Images: []api.ImageData{[]byte("somethingelse")}},
			},
			expect: expect{
				prompt: "You're a test, Harry! [img-0] I-I'm a what? [img-1]A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "messages with interleaved images",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "user", Images: []api.ImageData{[]byte("something")}},
				{Role: "user", Images: []api.ImageData{[]byte("somethingelse")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry!\n\n[img-0]\n\n[img-1] I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("something"),
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "truncate message with interleaved images",
			model:    visionModel,
			limit:    1024,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "user", Images: []api.ImageData{[]byte("something")}},
				{Role: "user", Images: []api.ImageData{[]byte("somethingelse")}},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "[img-0] I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
				images: [][]byte{
					[]byte("somethingelse"),
				},
			},
		},
		{
			name:     "message with system prompt",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "system", Content: "You are the Test Who Lived."},
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You are the Test Who Lived. You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "out of order system",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "system", Content: "You are the Test Who Lived."},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? You are the Test Who Lived. A test. And a thumping good one at that, I'd wager. ",
			},
		},
		{
			name:     "multiple images same prompt",
			model:    visionModel,
			limit:    2048,
			truncate: true,
			msgs: []api.Message{
				{Role: "user", Content: "Compare these two pictures of hotdogs", Images: []api.ImageData{[]byte("one hotdog"), []byte("two hotdogs")}},
			},
			expect: expect{
				prompt: "[img-0][img-1]Compare these two pictures of hotdogs ",
				images: [][]byte{[]byte("one hotdog"), []byte("two hotdogs")},
			},
		},
		{
			name:     "no truncate with limit exceeded",
			model:    visionModel,
			limit:    10,
			truncate: false,
			msgs: []api.Message{
				{Role: "user", Content: "You're a test, Harry!"},
				{Role: "assistant", Content: "I-I'm a what?"},
				{Role: "user", Content: "A test. And a thumping good one at that, I'd wager."},
			},
			expect: expect{
				prompt: "You're a test, Harry! I-I'm a what? A test. And a thumping good one at that, I'd wager. ",
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			model := tt.model
			opts := api.Options{Runner: api.Runner{NumCtx: tt.limit}}
			think := false
			prompt, images, err := chatPrompt(t.Context(), &model, mockRunner{}.Tokenize, &opts, tt.msgs, nil, &api.ThinkValue{Value: think}, tt.truncate)
			if tt.error == nil && err != nil {
				t.Fatal(err)
			} else if tt.error != nil && err != tt.error {
				t.Fatalf("expected err '%q', got '%q'", tt.error, err)
			}

			if diff := cmp.Diff(prompt, tt.prompt); diff != "" {
				t.Errorf("mismatch (-got +want):\n%s", diff)
			}

			if len(images) != len(tt.images) {
				t.Fatalf("expected %d images, got %d", len(tt.images), len(images))
			}

			for i := range images {
				if images[i].ID != i {
					t.Errorf("expected ID %d, got %d", i, images[i].ID)
				}

				if len(model.Config.ModelFamilies) == 0 {
					if !bytes.Equal(images[i].Data, tt.images[i]) {
						t.Errorf("expected %q, got %q", tt.images[i], images[i].Data)
					}
				}
			}
		})
	}
}

func TestChatPromptTokenizeCalls(t *testing.T) {
	tmpl, err := template.Parse(`
{{- if .System }}{{ .System }} {{ end }}
{{- if .Prompt }}{{ .Prompt }} {{ end }}
{{- if .Response }}{{ .Response }} {{ end }}`)
	if err != nil {
		t.Fatal(err)
	}
	model := Model{Template: tmpl}

	cases := []struct {
		name         string
		limit        int
		msgs         []api.Message
		maxTokenizes int
	}{
		{
			name:  "all messages fit",
			limit: 2048,
			msgs: []api.Message{
				{Role: "user", Content: "message 1"},
				{Role: "assistant", Content: "response 1"},
				{Role: "user", Content: "message 2"},
				{Role: "assistant", Content: "response 2"},
				{Role: "user", Content: "message 3"},
			},
			maxTokenizes: 1,
		},
		{
			name:  "truncate to last message",
			limit: 5,
			msgs: []api.Message{
				{Role: "user", Content: "message 1"},
				{Role: "assistant", Content: "response 1"},
				{Role: "user", Content: "message 2"},
				{Role: "assistant", Content: "response 2"},
				{Role: "user", Content: "message 3"},
			},
			maxTokenizes: 5,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			tokenizeCount := 0
			countingTokenize := func(ctx context.Context, s string) ([]int, error) {
				tokenizeCount++
				tokens, err := mockRunner{}.Tokenize(ctx, s)
				return tokens, err
			}

			opts := api.Options{Runner: api.Runner{NumCtx: tt.limit}}
			think := false
			_, _, err := chatPrompt(t.Context(), &model, countingTokenize, &opts, tt.msgs, nil, &api.ThinkValue{Value: think}, true)
			if err != nil {
				t.Fatal(err)
			}

			if tokenizeCount > tt.maxTokenizes {
				t.Errorf("tokenize called %d times, expected at most %d", tokenizeCount, tt.maxTokenizes)
			}
		})
	}
}

func TestChatPromptRendererDoesNotRewriteMessageContent(t *testing.T) {
	msgs := []api.Message{
		{
			Role:    "user",
			Content: "what do these photos have in common?",
			Images:  []api.ImageData{[]byte("img-1"), []byte("img-2"), []byte("img-3")},
		},
	}
	originalContent := msgs[0].Content

	m := Model{
		Config:         model.ConfigV2{Renderer: "qwen3-vl-instruct"},
		ProjectorPaths: []string{"vision"},
	}
	opts := api.Options{Runner: api.Runner{NumCtx: 8192}}
	think := false

	prompt, images, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: think}, true)
	if err != nil {
		t.Fatal(err)
	}

	if msgs[0].Content != originalContent {
		t.Fatalf("renderer path should not mutate message content: got %q, want %q", msgs[0].Content, originalContent)
	}

	if got, want := len(images), 3; got != want {
		t.Fatalf("len(images) = %d, want %d", got, want)
	}

	if prompt == "" {
		t.Fatal("prompt is empty")
	}
}

func TestToolBlockStartIndex(t *testing.T) {
	msgs := []api.Message{
		{Role: "user", Content: "before"},
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				{Function: api.ToolCallFunction{Name: "echo"}},
			},
		},
		{Role: "tool", Content: "first"},
		{Role: "tool", Content: "second"},
		{Role: "user", Content: "after"},
	}

	if got := toolBlockStartIndex(msgs, 2); got != 1 {
		t.Fatalf("toolBlockStartIndex(..., 2) = %d, want 1", got)
	}
	if got := toolBlockStartIndex(msgs, 3); got != 1 {
		t.Fatalf("toolBlockStartIndex(..., 3) = %d, want 1", got)
	}
	if got := toolBlockStartIndex(msgs, 4); got != 4 {
		t.Fatalf("toolBlockStartIndex(..., 4) = %d, want 4", got)
	}
}

func TestChatPromptPreservesOrDropsToolBlockAsAUnit(t *testing.T) {
	newModel := func() Model {
		return Model{
			Config: model.ConfigV2{Renderer: "qwen3.5"},
		}
	}

	tokenize := func(_ context.Context, s string) ([]int, error) {
		score := 0
		if strings.Contains(s, "older_context") {
			score += 100
		}
		if strings.Contains(s, "assistant_tool_turn") {
			score += 100
		}
		if strings.Contains(s, "tool_result_payload") {
			score += 100
		}
		if strings.Contains(s, "final_user_turn") {
			score++
		}

		return make([]int, score), nil
	}

	msgs := []api.Message{
		{Role: "user", Content: "older_context"},
		{
			Role:    "assistant",
			Content: "assistant_tool_turn",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "echo",
						Arguments: testArgs(map[string]any{"payload": "payload"}),
					},
				},
			},
		},
		{Role: "tool", Content: "tool_result_payload"},
		{Role: "user", Content: "final_user_turn"},
	}
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "echo",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsMap(map[string]api.ToolProperty{
						"payload": {Type: api.PropertyType{"string"}},
					}),
				},
			},
		},
	}

	t.Run("preserves block when it fits after dropping older history", func(t *testing.T) {
		model := newModel()
		opts := api.Options{Runner: api.Runner{NumCtx: 250}}

		prompt, _, err := chatPrompt(t.Context(), &model, tokenize, &opts, msgs, tools, &api.ThinkValue{Value: false}, true)
		if err != nil {
			t.Fatal(err)
		}

		if !strings.Contains(prompt, "<tool_call>") || !strings.Contains(prompt, "<tool_response>") {
			t.Fatalf("expected assistant tool-call turn and tool response block to survive together, got:\n%s", prompt)
		}
		if !strings.Contains(prompt, "assistant_tool_turn") || !strings.Contains(prompt, "tool_result_payload") {
			t.Fatalf("expected preserved tool block contents, got:\n%s", prompt)
		}
		if strings.Contains(prompt, "older_context") {
			t.Fatalf("expected older context to be truncated, got:\n%s", prompt)
		}
	})

	t.Run("drops block when assistant turn cannot fit with its tool responses", func(t *testing.T) {
		model := newModel()
		opts := api.Options{Runner: api.Runner{NumCtx: 50}}

		prompt, _, err := chatPrompt(t.Context(), &model, tokenize, &opts, msgs, tools, &api.ThinkValue{Value: false}, true)
		if err != nil {
			t.Fatal(err)
		}

		if strings.Contains(prompt, "assistant_tool_turn") || strings.Contains(prompt, "tool_result_payload") {
			t.Fatalf("tool responses must not survive truncation without their preceding assistant tool-call turn, got:\n%s", prompt)
		}
		if !strings.Contains(prompt, "final_user_turn") {
			t.Fatalf("expected final user turn to remain, got:\n%s", prompt)
		}
	})
}

func TestChatPromptPreservesContiguousMultiToolResponsesAsAUnit(t *testing.T) {
	model := Model{
		Config: model.ConfigV2{Renderer: "qwen3.5"},
	}
	tokenize := func(_ context.Context, s string) ([]int, error) {
		score := 0
		if strings.Contains(s, "older_context") {
			score += 100
		}
		if strings.Contains(s, "assistant_tool_turn") {
			score += 100
		}
		if strings.Contains(s, "tool_result_one") {
			score += 100
		}
		if strings.Contains(s, "tool_result_two") {
			score += 100
		}
		if strings.Contains(s, "final_user_turn") {
			score++
		}
		return make([]int, score), nil
	}

	msgs := []api.Message{
		{Role: "user", Content: "older_context"},
		{
			Role:    "assistant",
			Content: "assistant_tool_turn",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "echo",
						Arguments: testArgs(map[string]any{"payload": "payload"}),
					},
				},
			},
		},
		{Role: "tool", Content: "tool_result_one"},
		{Role: "tool", Content: "tool_result_two"},
		{Role: "user", Content: "final_user_turn"},
	}
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "echo",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsMap(map[string]api.ToolProperty{
						"payload": {Type: api.PropertyType{"string"}},
					}),
				},
			},
		},
	}

	opts := api.Options{Runner: api.Runner{NumCtx: 250}}
	prompt, _, err := chatPrompt(t.Context(), &model, tokenize, &opts, msgs, tools, &api.ThinkValue{Value: false}, true)
	if err != nil {
		t.Fatal(err)
	}

	if strings.Contains(prompt, "tool_result_one") || strings.Contains(prompt, "tool_result_two") {
		t.Fatalf("expected contiguous tool-response block to be dropped as a unit when it does not fit, got:\n%s", prompt)
	}
	if strings.Contains(prompt, "assistant_tool_turn") {
		t.Fatalf("expected parent assistant tool-call turn to drop with its tool responses, got:\n%s", prompt)
	}
	if !strings.Contains(prompt, "final_user_turn") {
		t.Fatalf("expected final user turn to remain, got:\n%s", prompt)
	}
}

func TestChatPromptDoesNotAttachNonContiguousToolResponsesToEarlierToolBlock(t *testing.T) {
	model := Model{
		Config: model.ConfigV2{Renderer: "qwen3.5"},
	}
	tokenize := func(_ context.Context, s string) ([]int, error) {
		score := 0
		if strings.Contains(s, "assistant_tool_turn") {
			score += 100
		}
		if strings.Contains(s, "tool_result_one") {
			score += 100
		}
		if strings.Contains(s, "followup_assistant") {
			score += 100
		}
		if strings.Contains(s, "tool_result_two") {
			score += 100
		}
		if strings.Contains(s, "final_user_turn") {
			score++
		}
		return make([]int, score), nil
	}

	msgs := []api.Message{
		{
			Role:    "assistant",
			Content: "assistant_tool_turn",
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "echo",
						Arguments: testArgs(map[string]any{"payload": "payload"}),
					},
				},
			},
		},
		{Role: "tool", Content: "tool_result_one"},
		{Role: "assistant", Content: "followup_assistant"},
		{Role: "tool", Content: "tool_result_two"},
		{Role: "user", Content: "final_user_turn"},
	}
	tools := []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name: "echo",
				Parameters: api.ToolFunctionParameters{
					Type: "object",
					Properties: testPropsMap(map[string]api.ToolProperty{
						"payload": {Type: api.PropertyType{"string"}},
					}),
				},
			},
		},
	}

	opts := api.Options{Runner: api.Runner{NumCtx: 150}}
	prompt, _, err := chatPrompt(t.Context(), &model, tokenize, &opts, msgs, tools, &api.ThinkValue{Value: false}, true)
	if err != nil {
		t.Fatal(err)
	}

	if strings.Contains(prompt, "assistant_tool_turn") || strings.Contains(prompt, "tool_result_one") {
		t.Fatalf("expected first contiguous tool block to drop together, got:\n%s", prompt)
	}
	if !strings.Contains(prompt, "tool_result_two") {
		t.Fatalf("expected non-contiguous stray tool response to remain independent, got:\n%s", prompt)
	}
	if !strings.Contains(prompt, "final_user_turn") {
		t.Fatalf("expected final user turn to remain, got:\n%s", prompt)
	}
}

func TestChatPromptGLMOcrRendererAddsImageTags(t *testing.T) {
	msgs := []api.Message{
		{
			Role:    "user",
			Content: "extract text",
			Images:  []api.ImageData{[]byte("img-1"), []byte("img-2")},
		},
	}

	m := Model{
		Config:         model.ConfigV2{Renderer: "glm-ocr"},
		ProjectorPaths: []string{"vision"},
	}
	opts := api.Options{Runner: api.Runner{NumCtx: 8192}}
	think := false

	prompt, images, err := chatPrompt(t.Context(), &m, mockRunner{}.Tokenize, &opts, msgs, nil, &api.ThinkValue{Value: think}, true)
	if err != nil {
		t.Fatal(err)
	}

	if got, want := len(images), 2; got != want {
		t.Fatalf("len(images) = %d, want %d", got, want)
	}

	if !strings.Contains(prompt, "<|user|>\n[img-0][img-1]extract text") {
		t.Fatalf("prompt missing glm-ocr image tags, got: %q", prompt)
	}
}

func TestRenderPromptResolvesDynamicGemma4Renderer(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "Hello"}}

	tests := []struct {
		name  string
		model Model
		want  string
	}{
		{
			name: "small from name",
			model: Model{
				Name:      "gemma4:e4b",
				ShortName: "gemma4:e4b",
				Config:    testConfigWithRenderer(gemma4RendererLegacy),
			},
			want: "<bos><|turn>user\nHello<turn|>\n<|turn>model\n",
		},
		{
			name: "large from model type",
			model: Model{
				Config: testConfigWithRendererAndType(gemma4RendererLegacy, "25.2B"),
			},
			want: "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := renderPrompt(&tt.model, msgs, nil, nil)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Fatalf("rendered prompt mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
