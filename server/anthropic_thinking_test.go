package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
)

func TestAnthropicWebSearchThinking(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tt := range []struct {
		name, fields string
		thinking     bool
		wantThink    any
	}{
		{"nonthinking/enabled", `,"thinking":{"type":"enabled","budget_tokens":1024}`, false, nil},
		{"nonthinking/effort", `,"output_config":{"effort":"high"}`, false, nil},
		{"nonthinking/disabled", `,"thinking":{"type":"disabled"}`, false, false},
		{"nonthinking/omitted", "", false, nil},
		{"thinking/enabled", `,"thinking":{"type":"enabled","budget_tokens":1024}`, true, true},
		{"thinking/effort", `,"output_config":{"effort":"xhigh"}`, true, "high"},
		{"thinking/disabled", `,"thinking":{"type":"disabled"}`, true, false},
		{"thinking/omitted", "", true, true},
	} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tt.name, streaming), func(t *testing.T) {
				setTestHome(t, t.TempDir())
				t.Setenv("OLLAMA_NO_CLOUD", "")
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
				var completions, searches atomic.Int32
				mock := mockRunner{CompletionFn: func(_ context.Context, req llm.CompletionRequest, yield func(llm.CompletionResponse)) error {
					content := "answer"
					if completions.Add(1) == 1 {
						content = `{"name":"web_search","arguments":{"query":"test"}}`
					} else if !strings.Contains(req.Prompt, "test search result") {
						return fmt.Errorf("follow-up is missing the search result")
					}
					yield(llm.CompletionResponse{Content: content, Done: true, DoneReason: llm.DoneReasonStop})
					return nil
				}}
				s := newServerWithMockRunner(t, &mock)
				var info map[string]any
				if tt.thinking {
					info = map[string]any{"capabilities": []any{"thinking"}}
				}
				createMinimalGGUFModel(t, s, "test-model", nil, `{{ if .Tools }}{{ .Tools }}{{ end }}
{{ range .Messages }}{{ .Role }}: {{ .Content }}
{{ range .ToolCalls }}{"name":"{{ .Function.Name }}","arguments":{{ .Function.Arguments }}}{{ end }}
{{ end }}`, info)
				followUps := make(chan api.ChatRequest, 1)
				router := gin.New()
				router.POST("/v1/messages", middleware.AnthropicMessagesMiddleware(lookupThinking), s.ChatHandler)
				router.POST("/api/chat", func(c *gin.Context) {
					var body bytes.Buffer
					var req api.ChatRequest
					if err := json.NewDecoder(io.TeeReader(c.Request.Body, &body)).Decode(&req); err != nil {
						t.Errorf("decode follow-up: %v", err)
						c.AbortWithStatus(http.StatusBadRequest)
						return
					}
					c.Request.Body = io.NopCloser(&body)
					followUps <- req
					s.ChatHandler(c)
				})
				router.POST("/web_search", func(c *gin.Context) {
					searches.Add(1)
					c.JSON(http.StatusOK, api.WebSearchResponse{Results: []api.WebSearchResult{{
						Title: "Test", URL: "https://example.com", Content: "test search result",
					}}})
				})
				backend := httptest.NewServer(router)
				defer backend.Close()
				t.Setenv("OLLAMA_HOST", backend.URL)
				originalEndpoint := anthropic.WebSearchEndpoint
				anthropic.WebSearchEndpoint = backend.URL + "/web_search"
				t.Cleanup(func() { anthropic.WebSearchEndpoint = originalEndpoint })

				body := fmt.Sprintf(`{"model":"test-model","messages":[{"role":"user","content":"search test"}],"max_tokens":2048,"tools":[{"type":"web_search_20250305","name":"web_search"}],"stream":%t%s}`, streaming, tt.fields)
				w := NewRecorder()
				router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
				if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "answer") || strings.Contains(w.Body.String(), "error") {
					t.Fatalf("expected successful search follow-up, got %d %s", w.Code, w.Body.String())
				}
				if streaming && !strings.Contains(w.Body.String(), "message_stop") {
					t.Fatalf("missing completion event: %s", w.Body.String())
				}
				if searches.Load() != 1 || completions.Load() != 2 {
					t.Fatalf("searches=%d completions=%d, want 1 search and 2 completions", searches.Load(), completions.Load())
				}
				followUp := <-followUps
				var gotThink any
				if followUp.Think != nil {
					gotThink = followUp.Think.Value
				}
				if gotThink != tt.wantThink {
					t.Fatalf("follow-up think=%#v, want %#v", gotThink, tt.wantThink)
				}
			})
		}
	}
}
