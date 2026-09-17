//go:build integration

package integration

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

var rawThinkingProtocolTags = []string{
	"<|channel>",
	"<channel|>",
	"<think>",
	"</think>",
	"<assistant>",
	"</assistant>",
	"<tool_call>",
	"</tool_call>",
}

func rejectRawThinkingProtocolTags(t *testing.T, field, value string) {
	t.Helper()
	for _, tag := range rawThinkingProtocolTags {
		if strings.Contains(value, tag) {
			t.Errorf("%s contains raw protocol tag %q: %s", field, tag, value)
		}
	}
}

// runThinkingEnabled verifies that thinking-capable models honor an explicit
// thinking request, complete a reasoning trace, and return the final answer
// without leaking raw channel tags.
func runThinkingEnabled(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	models := testModels([]string{smol})
	for _, modelName := range models {
		t.Run(modelName, func(t *testing.T) {
			requireCapability(ctx, t, client, modelName, "thinking")
			pullOrSkip(ctx, t, client, modelName)

			think := api.ThinkValue{Value: true}
			stream := false
			req := api.ChatRequest{
				Model:  modelName,
				Stream: &stream,
				Think:  &think,
				Messages: []api.Message{
					{Role: "user", Content: "What is 12 multiplied by 15? Give the final answer."},
				},
				Options: map[string]any{
					"temperature": 0,
					"seed":        42,
					// Deep-thinking models can use several thousand tokens on
					// simple problems before producing their final answer. Keep
					// this high enough to verify a natural stop, not truncation.
					"num_predict": 8192,
				},
			}

			var response api.ChatResponse
			err := client.Chat(ctx, &req, func(cr api.ChatResponse) error {
				response = cr
				return nil
			})
			if err != nil {
				if strings.Contains(err.Error(), "model requires more system memory") {
					t.Skip("model too large for test system")
				}
				t.Fatalf("chat failed: %v", err)
			}

			content := response.Message.Content
			thinking := response.Message.Thinking

			// Thinking should be non-empty when thinking is enabled
			if thinking == "" {
				t.Error("expected non-empty thinking output when thinking is enabled")
			}
			if content == "" {
				t.Error("expected non-empty final content after thinking")
			} else if !strings.Contains(content, "180") {
				t.Errorf("expected final answer 180, got content=%q", content)
			}
			if response.DoneReason != "stop" {
				t.Errorf("expected completed response, got done reason %q", response.DoneReason)
			}

			rejectRawThinkingProtocolTags(t, "content", content)
			rejectRawThinkingProtocolTags(t, "thinking", thinking)

			t.Logf("thinking (%d chars): %.100s...", len(thinking), thinking)
			t.Logf("content (%d chars): %s", len(content), content)
		})
	}
}

// runThinkingSuppressed verifies that when thinking is explicitly disabled,
// the model does not leak thinking/channel content into the response.
func runThinkingSuppressed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	models := testModels([]string{smol})
	for _, modelName := range models {
		t.Run(modelName, func(t *testing.T) {
			requireCapability(ctx, t, client, modelName, "thinking")
			pullOrSkip(ctx, t, client, modelName)

			stream := false
			think := api.ThinkValue{Value: false}
			req := api.ChatRequest{
				Model:  modelName,
				Stream: &stream,
				Think:  &think,
				Messages: []api.Message{
					{Role: "user", Content: "What is the capital of Japan? Answer in one word."},
				},
				Options: map[string]any{
					"temperature": 0,
					"seed":        42,
					"num_predict": 64,
				},
			}

			var response api.ChatResponse
			err := client.Chat(ctx, &req, func(cr api.ChatResponse) error {
				response = cr
				return nil
			})
			if err != nil {
				if strings.Contains(err.Error(), "model requires more system memory") {
					t.Skip("model too large for test system")
				}
				t.Fatalf("chat failed: %v", err)
			}

			content := response.Message.Content
			thinking := response.Message.Thinking

			// With thinking disabled, the answer must be returned as content.
			if !strings.Contains(content, "Tokyo") {
				t.Errorf("expected 'Tokyo' in content, got content=%q thinking=%q", content, thinking)
			}

			rejectRawThinkingProtocolTags(t, "content", content)
			rejectRawThinkingProtocolTags(t, "thinking", thinking)

			if thinking != "" {
				t.Errorf("expected empty thinking when thinking is disabled, got %q", thinking)
			}

			t.Logf("content: %s", content)
		})
	}
}
