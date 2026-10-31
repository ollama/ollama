//go:build integration

package integration

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/openai"
)

func TestOpenAIMaxCompletionTokens(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()

	client, endpoint, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	for _, modelName := range testModels([]string{"qwen3:0.6b"}) {
		t.Run(modelName, func(t *testing.T) {
			requireCapability(ctx, t, client, modelName, "thinking")
			const limit = 5
			for _, streaming := range []bool{false, true} {
				name := "non-streaming"
				if streaming {
					name = "streaming"
				}
				t.Run(name, func(t *testing.T) {
					payload, err := json.Marshal(map[string]any{
						"model":                 modelName,
						"messages":              []map[string]string{{"role": "user", "content": "Count from one to one hundred in words."}},
						"reasoning_effort":      "high",
						"max_completion_tokens": limit,
						"stream":                streaming,
						"stream_options":        map[string]bool{"include_usage": true},
					})
					if err != nil {
						t.Fatal(err)
					}
					req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+endpoint+"/v1/chat/completions", bytes.NewReader(payload))
					if err != nil {
						t.Fatal(err)
					}
					req.Header.Set("Content-Type", "application/json")
					resp, err := http.DefaultClient.Do(req)
					if err != nil {
						t.Fatal(err)
					}
					defer resp.Body.Close()
					if resp.StatusCode != http.StatusOK {
						t.Fatalf("status = %d, want 200", resp.StatusCode)
					}

					var finishReason string
					var completionTokens int
					var reasoning string
					if streaming {
						scanner := bufio.NewScanner(resp.Body)
						for scanner.Scan() {
							data, ok := strings.CutPrefix(scanner.Text(), "data: ")
							if !ok || data == "[DONE]" {
								continue
							}
							var chunk openai.ChatCompletionChunk
							if err := json.Unmarshal([]byte(data), &chunk); err != nil {
								t.Fatalf("decode stream chunk: %v", err)
							}
							if len(chunk.Choices) > 0 {
								reasoning += chunk.Choices[0].Delta.Reasoning
								if chunk.Choices[0].FinishReason != nil {
									finishReason = *chunk.Choices[0].FinishReason
								}
							}
							if chunk.Usage != nil {
								completionTokens = chunk.Usage.CompletionTokens
							}
						}
						if err := scanner.Err(); err != nil {
							t.Fatal(err)
						}
					} else {
						var completion openai.ChatCompletion
						if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
							t.Fatal(err)
						}
						if len(completion.Choices) != 1 || completion.Choices[0].FinishReason == nil {
							t.Fatalf("missing finish reason: %+v", completion.Choices)
						}
						finishReason = *completion.Choices[0].FinishReason
						completionTokens = completion.Usage.CompletionTokens
						reasoning = completion.Choices[0].Message.Reasoning
					}
					if finishReason != "length" || completionTokens != limit {
						t.Fatalf("finish_reason = %q, completion_tokens = %d; want length and %d", finishReason, completionTokens, limit)
					}
					if reasoning == "" {
						t.Fatal("expected reasoning tokens before the completion limit")
					}
				})
			}
		})
	}
}
