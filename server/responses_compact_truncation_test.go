package server

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/openai"
)

func TestResponsesLocalCompactionOverflowRecovery(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, path := range []string{"/v1/responses/compact", "/v1/responses"} {
		t.Run(path, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			t.Setenv("OLLAMA_CONTEXT_LENGTH", "300")
			t.Setenv("OLLAMA_GO_TEMPLATE", "")

			const contextLength = 300
			var prompts []string
			mock := mockRunner{contextLength: contextLength}
			mock.TemplateFn = func(_ context.Context, req llm.ChatRequest) (string, error) {
				var prompt strings.Builder
				for _, message := range req.Messages {
					prompt.WriteString(message.Content)
					prompt.WriteByte(' ')
				}
				return prompt.String(), nil
			}
			mock.ChatFn = func(ctx context.Context, req llm.ChatRequest, yield func(llm.ChatResponse)) error {
				prompt, err := mock.TemplateFn(ctx, req)
				if err != nil {
					return err
				}
				prompts = append(prompts, prompt)
				if tokens := len(strings.Fields(prompt)); tokens >= contextLength {
					return api.StatusError{
						StatusCode:   http.StatusBadRequest,
						ErrorMessage: fmt.Sprintf("request (%d tokens) exceeds the available context size (%d tokens), try increasing it", tokens, contextLength),
					}
				}
				arguments := api.NewToolCallFunctionArguments()
				arguments.Set("summary", "Continue the task.")
				arguments.Set("retain_item_ids", []string{})
				yield(llm.ChatResponse{
					Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "summary", Function: api.ToolCallFunction{
						Name: openai.CreateSummaryToolName, Arguments: arguments,
					}}}},
					Done: true, DoneReason: llm.DoneReasonStop,
				})
				return nil
			}
			s := newServerWithMockRunner(t, &mock)
			createMinimalGGUFModel(t, s, "local-summary", gguftest.KV{"tokenizer.chat_template": "{{ tools }} {{ messages }}"}, "", nil)
			router := gin.New()
			router.POST("/v1/responses/compact", s.ResponsesCompactHandler)
			router.POST("/v1/responses", s.responsesCompactionMiddleware(), middleware.ResponsesMiddleware(), s.ChatHandler)

			input := fmt.Sprintf(`[{"role":"user","content":"original goal"},{"role":"assistant","content":%q},{"role":"user","content":"latest request"}]`, strings.Repeat("old result ", 400))
			if path == "/v1/responses" {
				input = strings.TrimSuffix(input, "]") + `,{"type":"compaction_trigger"}]`
			}
			body := fmt.Sprintf(`{"model":"local-summary","stream":%t,"input":%s}`, path == "/v1/responses", input)
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
			request.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
			if len(prompts) != 2 || !strings.Contains(prompts[0], "old result") || strings.Contains(prompts[1], "old result") {
				t.Fatalf("expected one explicit overflow recovery, got prompts=%q", prompts)
			}
			for _, marker := range []string{"original goal", "latest request", "1 older transcript items were omitted"} {
				if !strings.Contains(prompts[1], marker) {
					t.Errorf("compaction retry lost %q", marker)
				}
			}
			if !strings.Contains(response.Body.String(), "1 older transcript items were omitted") {
				t.Fatalf("compaction output lost the omission warning: %s", response.Body.String())
			}
		})
	}
}
