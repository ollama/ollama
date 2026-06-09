//go:build integration

package integration

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestThinkingEnabled verifies that when thinking is requested, the model
// produces both thinking and content output without leaking raw channel tags.
func TestThinkingEnabled(t *testing.T) {
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
					{Role: "user", Content: "What is 12 * 15? Think step by step."},
				},
				Options: map[string]any{
					"temperature": 0,
					"seed":        42,
					"num_predict": 512,
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

			// The answer (180) should appear in thinking, content, or both.
			// Some models put everything in thinking and leave content empty
			// if they hit the token limit while still thinking.
			combined := thinking + " " + content
			if !strings.Contains(combined, "180") {
				t.Errorf("expected '180' in thinking or content, got thinking=%q content=%q", thinking, content)
			}

			// Neither thinking nor content should contain raw channel tags
			if strings.Contains(content, "<|channel>") || strings.Contains(content, "<channel|>") {
				t.Errorf("content contains raw channel tags: %s", content)
			}
			if strings.Contains(thinking, "<|channel>") || strings.Contains(thinking, "<channel|>") {
				t.Errorf("thinking contains raw channel tags: %s", thinking)
			}

			t.Logf("thinking (%d chars): %.100s...", len(thinking), thinking)
			t.Logf("content (%d chars): %s", len(content), content)
		})
	}
}

// TestThinkingSuppressed verifies that when thinking is NOT requested,
// the model does not leak thinking/channel content into the response.
func TestThinkingSuppressed(t *testing.T) {
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
			req := api.ChatRequest{
				Model:  modelName,
				Stream: &stream,
				// Think is nil — thinking not requested
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

			// The answer should appear in content or thinking
			combined := content + " " + thinking
			if !strings.Contains(combined, "Tokyo") {
				t.Errorf("expected 'Tokyo' in content or thinking, got content=%q thinking=%q", content, thinking)
			}

			// Content must NOT contain channel/thinking tags
			if strings.Contains(content, "<|channel>") || strings.Contains(content, "<channel|>") {
				t.Errorf("content contains leaked channel tags when thinking not requested: %s", content)
			}
			if strings.Contains(content, "thought") && strings.Contains(content, "<channel|>") {
				t.Errorf("content contains leaked thinking block: %s", content)
			}

			// Thinking field should ideally be empty when not requested.
			// Some small models may still produce thinking output; log but don't fail.
			if thinking != "" {
				t.Logf("WARNING: model produced thinking output when not requested (%d chars): %.100s...", len(thinking), thinking)
			}

			t.Logf("content: %s", content)
		})
	}
}
