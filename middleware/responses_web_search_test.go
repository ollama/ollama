package middleware

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func TestWebSearchResponsesWriterNonStreaming(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	request := openai.ResponsesRequest{
		Model: "test-model",
		Tools: []openai.ResponsesTool{{Type: "web_search"}},
	}
	inner := &ResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer},
		model:      request.Model,
		responseID: "resp_test",
		itemID:     "msg_test",
		request:    request,
	}
	followUps := 0
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer},
		inner:      inner,
		req:        request,
		chat:       &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(_ context.Context, query string) (*api.WebSearchResponse, error) {
			if query != "ollama news" {
				t.Fatalf("search query = %q", query)
			}
			return &api.WebSearchResponse{Results: []api.WebSearchResult{{Title: "Ollama", URL: "https://ollama.com/news", Content: "news"}}}, nil
		},
		followUpChat: func(_ context.Context, messages []api.Message, _ api.Tools) (api.ChatResponse, error) {
			followUps++
			if len(messages) != 2 || messages[1].Role != "tool" {
				t.Fatalf("follow-up messages = %#v", messages)
			}
			if strings.Contains(messages[1].Content, "Cite") || !strings.Contains(messages[1].Content, "URL: https://ollama.com/news") {
				t.Fatalf("unexpected search result content: %q", messages[1].Content)
			}
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "Read [Ollama](https://ollama.com/news)."}, Metrics: api.Metrics{PromptEvalCount: 7, EvalCount: 3}}, nil
		},
	}

	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "ollama news"})}}}}, Metrics: api.Metrics{PromptEvalCount: 5, EvalCount: 2}}
	data, err := json.Marshal(initial)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	if followUps != 1 {
		t.Fatalf("follow-up calls = %d", followUps)
	}

	var response openai.ResponsesResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v: %s", err, recorder.Body.String())
	}
	if len(response.Output) != 2 || response.Output[0].Type != "web_search_call" || response.Output[1].Type != "message" {
		t.Fatalf("output = %#v", response.Output)
	}
	if response.Output[0].Action == nil || response.Output[0].Action.Query != "ollama news" {
		t.Fatalf("search action = %#v", response.Output[0].Action)
	}
	if response.Usage == nil || response.Usage.InputTokens != 12 || response.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	if len(response.Output[1].Content[0].Annotations) != 0 {
		t.Fatalf("annotations = %#v, want none", response.Output[1].Content[0].Annotations)
	}
}

func TestWebSearchResponsesWriterStreamingNoSearchStreamsImmediately(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request}

	chunk, _ := json.Marshal(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "hello"}})
	if _, err := writer.Write(chunk); err != nil {
		t.Fatal(err)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "response.output_text.delta") || !strings.Contains(body, "hello") {
		t.Fatalf("content was not streamed immediately: %s", body)
	} else if strings.Contains(body, "response.completed") {
		t.Fatalf("response completed before terminal chunk: %s", body)
	}

	done, _ := json.Marshal(api.ChatResponse{Done: true})
	if _, err := writer.Write(done); err != nil {
		t.Fatal(err)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "response.completed") || !strings.Contains(body, "hello") {
		t.Fatalf("missing completed event: %s", recorder.Body.String())
	}
}

func TestWebSearchResponsesWriterNonStreamingAuthorizationError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	request := openai.ResponsesRequest{Model: "test-model", Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, model: request.Model, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request}

	writer.WriteHeader(http.StatusUnauthorized)
	data := []byte(`{"error":"sign in required","signin_url":"https://ollama.com/signin"}`)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusUnauthorized)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "https://ollama.com/signin") {
		t.Fatalf("missing sign-in URL: %s", body)
	}
}

func TestWebSearchResponsesWriterStreamingAuthorizationError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model:cloud", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request}

	writer.WriteHeader(http.StatusUnauthorized)
	data := []byte(`{"error":"sign in required","signin_url":"https://ollama.com/signin"}`)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, "response.failed") || !strings.Contains(body, "https://ollama.com/signin") {
		t.Fatalf("missing streaming authorization error: %s", body)
	}
}

func TestWebSearchResponsesWriterStreamingRateLimitError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request}

	if err := writer.writeWebSearchError(api.StatusError{StatusCode: http.StatusTooManyRequests, ErrorMessage: "slow down"}, api.Metrics{}); err != nil {
		t.Fatal(err)
	}
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, "response.failed") || !strings.Contains(body, "rate_limit_exceeded") {
		t.Fatalf("unexpected rate-limit response: %s", body)
	}
	if !strings.Contains(recorder.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("content type = %q", recorder.Header().Get("Content-Type"))
	}
}

func TestResponsesMiddlewareWebSearchStatusOnlyResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	router := gin.New()
	router.POST("/v1/responses", ResponsesMiddleware(), func(c *gin.Context) {
		c.AbortWithStatus(http.StatusServiceUnavailable)
	})

	request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"test-model",
		"input":"hello",
		"tools":[{"type":"web_search"}]
	}`))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusServiceUnavailable)
	}
}

func TestWebSearchResponsesWriterPreservesFollowUpErrors(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		body       string
		wantInBody string
	}{
		{name: "authorization", status: http.StatusUnauthorized, body: `{"error":"sign in required","signin_url":"https://ollama.com/signin/followup"}`, wantInBody: "https://ollama.com/signin/followup"},
		{name: "rate limit", status: http.StatusTooManyRequests, body: `{"error":"slow down"}`, wantInBody: "slow down"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/api/chat" {
					t.Errorf("follow-up path = %q", request.URL.Path)
				}
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)

			gin.SetMode(gin.TestMode)
			recorder := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(recorder)
			request := openai.ResponsesRequest{Model: "test-model:cloud", Tools: []openai.ResponsesTool{{Type: "web_search"}}}
			inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, model: request.Model, responseID: "resp_test", itemID: "msg_test", request: request}
			writer := &WebSearchResponsesWriter{
				BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
				chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
				search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
			}

			initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}}
			data, _ := json.Marshal(initial)
			if _, err := writer.Write(data); err != nil {
				t.Fatal(err)
			}
			if recorder.Code != test.status || !strings.Contains(recorder.Body.String(), test.wantInBody) {
				t.Fatalf("response status=%d body=%s", recorder.Code, recorder.Body.String())
			}
		})
	}
}

func TestWebSearchResponsesWriterStreamingHidesInternalFunction(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat: &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) {
			body := recorder.Body.String()
			if !strings.Contains(body, "response.web_search_call.searching") || strings.Contains(body, "response.web_search_call.completed") {
				t.Fatalf("search lifecycle before execution = %s", body)
			}
			return &api.WebSearchResponse{}, nil
		},
		followUpChat: func(context.Context, []api.Message, api.Tools) (api.ChatResponse, error) {
			body := recorder.Body.String()
			if !strings.Contains(body, "response.web_search_call.completed") || strings.Contains(body, "response.completed") {
				t.Fatalf("search lifecycle before follow-up = %s", body)
			}
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "done"}}, nil
		},
	}
	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}}
	data, _ := json.Marshal(initial)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, "response.web_search_call.completed") {
		t.Fatalf("missing completed web search event: %s", body)
	}
	if strings.Contains(body, "response.function_call_arguments") || strings.Contains(body, `"type":"function_call"`) {
		t.Fatalf("internal function call leaked: %s", body)
	}
}

func TestWebSearchResponsesWriterStreamingToolCallBeforeDoneChunk(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
		followUpChat: func(context.Context, []api.Message, api.Tools) (api.ChatResponse, error) {
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "done"}}, nil
		},
	}
	toolChunk := api.ChatResponse{Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}}
	data, _ := json.Marshal(toolChunk)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	if recorder.Body.Len() != 0 {
		t.Fatalf("tool chunk leaked before done: %s", recorder.Body.String())
	}
	done, _ := json.Marshal(api.ChatResponse{Done: true, Metrics: api.Metrics{PromptEvalCount: 9, EvalCount: 4}})
	if _, err := writer.Write(done); err != nil {
		t.Fatal(err)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, "response.web_search_call.completed") || strings.Contains(body, "response.function_call_arguments") {
		t.Fatalf("unexpected response stream: %s", body)
	}
}

func TestWebSearchResponsesWriterStreamingPreservesContentBeforeToolCall(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
		followUpChat: func(context.Context, []api.Message, api.Tools) (api.ChatResponse, error) {
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "done"}}, nil
		},
	}

	content, _ := json.Marshal(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "I will search."}})
	if _, err := writer.Write(content); err != nil {
		t.Fatal(err)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "response.output_text.delta") || !strings.Contains(body, "I will search.") {
		t.Fatalf("pre-search content was not streamed immediately: %s", body)
	} else if strings.Contains(body, "response.web_search_call.in_progress") {
		t.Fatalf("search started before its tool call: %s", body)
	}

	toolChunk, _ := json.Marshal(api.ChatResponse{Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}})
	if _, err := writer.Write(toolChunk); err != nil {
		t.Fatal(err)
	}
	done, _ := json.Marshal(api.ChatResponse{Done: true})
	if _, err := writer.Write(done); err != nil {
		t.Fatal(err)
	}

	body := recorder.Body.String()
	// Pre-search content must be emitted as a completed message item before
	// the web_search_call events, and the private function must not leak.
	if !strings.Contains(body, "I will search.") {
		t.Fatalf("pre-search content was discarded: %s", body)
	}
	if !strings.Contains(body, "response.web_search_call.completed") {
		t.Fatalf("missing web search completed event: %s", body)
	}
	if strings.Contains(body, "response.function_call_arguments") {
		t.Fatalf("private web_search function leaked: %s", body)
	}
	if strings.Count(body, "event: response.output_text.delta") != 2 {
		t.Fatalf("unexpected output delta count; pre-search content may have been replayed: %s", body)
	}
}

func TestWebSearchResponsesWriterStreamingPreservesThinkingBeforeToolCall(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
		followUpChat: func(_ context.Context, messages []api.Message, _ api.Tools) (api.ChatResponse, error) {
			assistant := messages[len(messages)-2]
			if assistant.Thinking != "I should search first." {
				t.Fatalf("follow-up thinking = %q", assistant.Thinking)
			}
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "done"}}, nil
		},
	}

	thinking, _ := json.Marshal(api.ChatResponse{Message: api.Message{Role: "assistant", Thinking: "I should search first."}})
	if _, err := writer.Write(thinking); err != nil {
		t.Fatal(err)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "response.reasoning_summary_text.delta") || !strings.Contains(body, "I should search first.") {
		t.Fatalf("pre-search reasoning was not streamed immediately: %s", body)
	} else if strings.Contains(body, "response.web_search_call.in_progress") {
		t.Fatalf("search started before its tool call: %s", body)
	}
	toolChunk, _ := json.Marshal(api.ChatResponse{Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}})
	if _, err := writer.Write(toolChunk); err != nil {
		t.Fatal(err)
	}
	done, _ := json.Marshal(api.ChatResponse{Done: true})
	if _, err := writer.Write(done); err != nil {
		t.Fatal(err)
	}

	body := recorder.Body.String()
	reasoningDelta := strings.Index(body, "response.reasoning_summary_text.delta")
	reasoningDone := strings.Index(body, "response.reasoning_summary_text.done")
	searchStarted := strings.Index(body, "response.web_search_call.in_progress")
	if reasoningDelta < 0 || reasoningDone < reasoningDelta || searchStarted < reasoningDone {
		t.Fatalf("reasoning/search lifecycle is out of order: %s", body)
	}
}

func TestWebSearchResponsesWriterStreamingContentAndToolCallInSameChunk(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
		followUpChat: func(context.Context, []api.Message, api.Tools) (api.ChatResponse, error) {
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "done"}}, nil
		},
	}

	chunk, _ := json.Marshal(api.ChatResponse{Message: api.Message{
		Role:    "assistant",
		Content: "Let me check.",
		ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{
			Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"}),
		}}},
	}})
	if _, err := writer.Write(chunk); err != nil {
		t.Fatal(err)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "Let me check.") || strings.Contains(body, "response.function_call_arguments") {
		t.Fatalf("same-chunk content was not streamed safely: %s", body)
	}

	done, _ := json.Marshal(api.ChatResponse{Done: true})
	if _, err := writer.Write(done); err != nil {
		t.Fatal(err)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, "response.web_search_call.completed") || strings.Count(body, "event: response.output_text.delta") != 2 {
		t.Fatalf("unexpected response stream: %s", body)
	}
}

func TestWebSearchResponsesWriterStreamsFollowUpAsProduced(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
		followUpStream: func(_ context.Context, _ []api.Message, _ api.Tools, yield func(api.ChatResponse) error) error {
			if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "streamed "}}); err != nil {
				return err
			}
			if body := recorder.Body.String(); !strings.Contains(body, `"delta":"streamed "`) || strings.Contains(body, "response.completed") {
				t.Fatalf("first follow-up chunk was not flushed immediately: %s", body)
			}
			if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "answer"}}); err != nil {
				return err
			}
			return yield(api.ChatResponse{Done: true, Metrics: api.Metrics{PromptEvalCount: 7, EvalCount: 3}})
		},
	}

	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}, Metrics: api.Metrics{PromptEvalCount: 5, EvalCount: 2}}
	data, _ := json.Marshal(initial)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}

	body := recorder.Body.String()
	searchDone := strings.Index(body, "response.web_search_call.completed")
	firstDelta := strings.Index(body, `"delta":"streamed "`)
	secondDelta := strings.Index(body, `"delta":"answer"`)
	completed := strings.Index(body, "response.completed")
	if searchDone < 0 || firstDelta < searchDone || secondDelta < firstDelta || completed < secondDelta {
		t.Fatalf("follow-up stream lifecycle is out of order: %s", body)
	}
	output := completedResponseOutput(t, body)
	if len(output) != 2 || output[0]["type"] != "web_search_call" || output[1]["type"] != "message" {
		t.Fatalf("terminal output = %#v", output)
	}
	content := output[1]["content"].([]any)[0].(map[string]any)
	if content["text"] != "streamed answer" {
		t.Fatalf("final text = %#v", content["text"])
	}
}

func TestWebSearchResponsesWriterStreamsFollowUpBeforeSecondSearch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	searches := 0
	followUps := 0
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat: &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(_ context.Context, query string) (*api.WebSearchResponse, error) {
			searches++
			if query != []string{"first", "second"}[searches-1] {
				t.Fatalf("search %d query = %q", searches, query)
			}
			return &api.WebSearchResponse{}, nil
		},
		followUpStream: func(_ context.Context, messages []api.Message, _ api.Tools, yield func(api.ChatResponse) error) error {
			followUps++
			if followUps == 1 {
				if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "Need another search."}}); err != nil {
					return err
				}
				if !strings.Contains(recorder.Body.String(), `"delta":"Need another search."`) {
					t.Fatalf("intermediate content was not streamed: %s", recorder.Body.String())
				}
				if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "call_2", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "second"})}}}}}); err != nil {
					return err
				}
				return yield(api.ChatResponse{Done: true, Metrics: api.Metrics{PromptEvalCount: 3, EvalCount: 4}})
			}

			assistant := messages[len(messages)-2]
			if assistant.Content != "Need another search." || len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].Function.Name != "web_search" {
				t.Fatalf("second-search assistant context = %#v", assistant)
			}
			if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "Final "}}); err != nil {
				return err
			}
			if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "answer."}}); err != nil {
				return err
			}
			return yield(api.ChatResponse{Done: true, Metrics: api.Metrics{PromptEvalCount: 5, EvalCount: 6}})
		},
	}

	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "first"})}}}}, Metrics: api.Metrics{PromptEvalCount: 1, EvalCount: 2}}
	data, _ := json.Marshal(initial)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	if searches != 2 || followUps != 2 {
		t.Fatalf("searches=%d follow-ups=%d, want 2 each", searches, followUps)
	}

	body := recorder.Body.String()
	firstSearchDone := strings.Index(body, "response.web_search_call.completed")
	intermediate := strings.Index(body, `"delta":"Need another search."`)
	secondSearch := strings.Index(body, `"query":"second"`)
	finalDelta := strings.Index(body, `"delta":"Final "`)
	completed := strings.Index(body, "response.completed")
	if firstSearchDone < 0 || intermediate < firstSearchDone || secondSearch < intermediate || finalDelta < secondSearch || completed < finalDelta {
		t.Fatalf("repeated search lifecycle is out of order: %s", body)
	}
	if strings.Count(body, "event: response.web_search_call.completed") != 2 || strings.Contains(body, "response.function_call_arguments") {
		t.Fatalf("unexpected search events: %s", body)
	}
	output := completedResponseOutput(t, body)
	wantTypes := []string{"web_search_call", "message", "web_search_call", "message"}
	if len(output) != len(wantTypes) {
		t.Fatalf("terminal output = %#v", output)
	}
	for i, want := range wantTypes {
		if output[i]["type"] != want {
			t.Fatalf("output[%d] type = %v, want %s: %#v", i, output[i]["type"], want, output)
		}
	}
}

func TestWebSearchResponsesWriterStreamingMixedFollowUpDoesNotLatchText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}, {Type: "function", Name: "get_weather", Description: ptr("weather"), Parameters: map[string]any{"type": "object"}}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
		followUpStream: func(_ context.Context, _ []api.Message, _ api.Tools, yield func(api.ChatResponse) error) error {
			chunks := []api.ChatResponse{
				{Message: api.Message{Role: "assistant", Content: "before"}},
				{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "call_weather", Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "SF"})}}}}},
				{Message: api.Message{Role: "assistant", Content: " after"}},
				{Done: true},
			}
			for _, chunk := range chunks {
				if err := yield(chunk); err != nil {
					return err
				}
			}
			return nil
		},
	}

	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}}
	data, _ := json.Marshal(initial)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	body := recorder.Body.String()
	if !strings.Contains(body, `"delta":"before"`) || !strings.Contains(body, `"delta":" after"`) {
		t.Fatalf("follow-up text was dropped: %s", body)
	}
	if strings.Count(body, "event: response.function_call_arguments.delta") != 1 {
		t.Fatalf("function call should be emitted once: %s", body)
	}
	output := completedResponseOutput(t, body)
	wantTypes := []string{"web_search_call", "message", "function_call", "message"}
	if len(output) != len(wantTypes) {
		t.Fatalf("terminal output = %#v", output)
	}
	for i, want := range wantTypes {
		if output[i]["type"] != want {
			t.Fatalf("output[%d] type = %v, want %s: %#v", i, output[i]["type"], want, output)
		}
	}
}

func TestWebSearchResponsesWriterStreamingSplitInitialToolsDoNotLatchFinalText(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}, {Type: "function", Name: "get_weather", Description: ptr("weather"), Parameters: map[string]any{"type": "object"}}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat:   &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) { return &api.WebSearchResponse{}, nil },
		followUpStream: func(_ context.Context, _ []api.Message, _ api.Tools, yield func(api.ChatResponse) error) error {
			if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "final answer"}}); err != nil {
				return err
			}
			return yield(api.ChatResponse{Done: true})
		},
	}

	weather, _ := json.Marshal(api.ChatResponse{Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_weather", Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "SF"})}}}}})
	search, _ := json.Marshal(api.ChatResponse{Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_search", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "weather"})}}}}})
	done, _ := json.Marshal(api.ChatResponse{Done: true})
	for _, chunk := range [][]byte{weather, search, done} {
		if _, err := writer.Write(chunk); err != nil {
			t.Fatal(err)
		}
	}
	body := recorder.Body.String()
	if !strings.Contains(body, `"delta":"final answer"`) || strings.Count(body, "event: response.function_call_arguments.delta") != 1 {
		t.Fatalf("split initial tools corrupted stream: %s", body)
	}
	output := completedResponseOutput(t, body)
	wantTypes := []string{"function_call", "web_search_call", "message"}
	if len(output) != len(wantTypes) {
		t.Fatalf("terminal output = %#v", output)
	}
	for i, want := range wantTypes {
		if output[i]["type"] != want {
			t.Fatalf("output[%d] type = %v, want %s: %#v", i, output[i]["type"], want, output)
		}
	}
}

func TestWebSearchResponsesWriterStreamingLoopExhaustionFails(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	searches := 0
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat: &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) {
			searches++
			return &api.WebSearchResponse{}, nil
		},
		followUpStream: func(_ context.Context, _ []api.Message, _ api.Tools, yield func(api.ChatResponse) error) error {
			call := api.ToolCall{ID: "call_next", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "again"})}}
			if err := yield(api.ChatResponse{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{call}}}); err != nil {
				return err
			}
			return yield(api.ChatResponse{Done: true})
		},
	}

	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "first"})}}}}}
	data, _ := json.Marshal(initial)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	body := recorder.Body.String()
	if searches != maxWebSearchLoops || !strings.Contains(body, "response.failed") || !strings.Contains(body, "exceeded the maximum") || strings.Contains(body, "response.completed") {
		t.Fatalf("loop exhaustion was not a terminal failure: searches=%d body=%s", searches, body)
	}
}

func TestWebSearchResponsesWriterNonStreamingPreservesContentBeforeToolCall(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	request := openai.ResponsesRequest{Model: "test-model", Tools: []openai.ResponsesTool{{Type: "web_search"}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, model: request.Model, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat: &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(_ context.Context, query string) (*api.WebSearchResponse, error) {
			return &api.WebSearchResponse{Results: []api.WebSearchResult{{Title: "Result", URL: "https://example.com", Content: "info"}}}, nil
		},
		followUpChat: func(_ context.Context, messages []api.Message, _ api.Tools) (api.ChatResponse, error) {
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "Here is the answer."}, Metrics: api.Metrics{PromptEvalCount: 7, EvalCount: 3}}, nil
		},
	}

	// Non-streaming response with both content and a web_search tool call.
	initial := api.ChatResponse{Done: true, Message: api.Message{Content: "Let me look this up.", ToolCalls: []api.ToolCall{{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "test"})}}}}, Metrics: api.Metrics{PromptEvalCount: 5, EvalCount: 2}}
	data, err := json.Marshal(initial)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}

	var response openai.ResponsesResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v: %s", err, recorder.Body.String())
	}

	// Output should be: [pre-search message, web_search_call, final message]
	if len(response.Output) != 3 {
		t.Fatalf("output count = %d, want 3: %#v", len(response.Output), response.Output)
	}
	if response.Output[0].Type != "message" || response.Output[0].Content[0].Text != "Let me look this up." {
		t.Fatalf("pre-search message = %#v", response.Output[0])
	}
	if response.Output[1].Type != "web_search_call" {
		t.Fatalf("web_search_call = %#v", response.Output[1])
	}
	if response.Output[2].Type != "message" || response.Output[2].Content[0].Text != "Here is the answer." {
		t.Fatalf("final message = %#v", response.Output[2])
	}
}

func TestWebSearchResponsesWriterNonStreamingSurfacesMixedToolCalls(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	request := openai.ResponsesRequest{Model: "test-model", Tools: []openai.ResponsesTool{{Type: "web_search"}, {Type: "function", Name: "get_weather", Description: ptr("weather"), Parameters: map[string]any{"type": "object"}}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, model: request.Model, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat: &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) {
			return &api.WebSearchResponse{}, nil
		},
		followUpChat: func(_ context.Context, messages []api.Message, _ api.Tools) (api.ChatResponse, error) {
			// Verify the assistant message only contains the web_search tool call,
			// not the get_weather tool call.
			if len(messages) < 2 {
				t.Fatalf("expected at least 2 messages, got %d", len(messages))
			}
			assistant := messages[len(messages)-2]
			if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].Function.Name != "web_search" {
				t.Fatalf("assistant message should only have web_search tool call, got %#v", assistant.ToolCalls)
			}
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "done"}}, nil
		},
	}

	// Non-streaming response with both web_search and get_weather tool calls.
	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{
		{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "weather"})}},
		{ID: "call_2", Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "SF"})}},
	}}}
	data, _ := json.Marshal(initial)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}

	var response openai.ResponsesResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v: %s", err, recorder.Body.String())
	}

	// Output should include a function_call item for get_weather.
	var hasFunctionCall bool
	for _, item := range response.Output {
		if item.Type == "function_call" && item.Name == "get_weather" {
			hasFunctionCall = true
		}
	}
	if !hasFunctionCall {
		t.Fatalf("mixed tool call (get_weather) was not surfaced: %#v", response.Output)
	}
}

func TestWebSearchResponsesWriterStreamingSurfacesMixedToolCalls(t *testing.T) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	stream := true
	request := openai.ResponsesRequest{Model: "test-model", Stream: &stream, Tools: []openai.ResponsesTool{{Type: "web_search"}, {Type: "function", Name: "get_weather", Description: ptr("weather"), Parameters: map[string]any{"type": "object"}}}}
	inner := &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, converter: openai.NewResponsesStreamConverter("resp_test", "msg_test", request.Model, request), model: request.Model, stream: true, responseID: "resp_test", itemID: "msg_test", request: request}
	writer := &WebSearchResponsesWriter{
		BaseWriter: BaseWriter{ResponseWriter: ctx.Writer}, inner: inner, req: request,
		chat: &api.ChatRequest{Model: request.Model, Tools: api.Tools{openai.WebSearchFunctionTool()}},
		search: func(context.Context, string) (*api.WebSearchResponse, error) {
			return &api.WebSearchResponse{}, nil
		},
		followUpChat: func(_ context.Context, messages []api.Message, _ api.Tools) (api.ChatResponse, error) {
			assistant := messages[len(messages)-2]
			if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].Function.Name != "web_search" {
				t.Fatalf("assistant message should only have web_search tool call, got %#v", assistant.ToolCalls)
			}
			return api.ChatResponse{Done: true, Message: api.Message{Role: "assistant", Content: "done"}}, nil
		},
	}

	// Streaming: initial response has both web_search and get_weather tool calls.
	initial := api.ChatResponse{Done: true, Message: api.Message{ToolCalls: []api.ToolCall{
		{ID: "call_1", Function: api.ToolCallFunction{Name: "web_search", Arguments: testArgs(map[string]any{"query": "weather"})}},
		{ID: "call_2", Function: api.ToolCallFunction{Name: "get_weather", Arguments: testArgs(map[string]any{"city": "SF"})}},
	}}}
	data, _ := json.Marshal(initial)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}

	body := recorder.Body.String()
	if !strings.Contains(body, "response.web_search_call.completed") {
		t.Fatalf("missing web search event: %s", body)
	}
	if !strings.Contains(body, "response.function_call_arguments") {
		t.Fatalf("mixed function call (get_weather) was not emitted: %s", body)
	}
	if !strings.Contains(body, "get_weather") {
		t.Fatalf("get_weather function name not found: %s", body)
	}

	output := completedResponseOutput(t, body)
	var hasFunctionCall, hasFinalMessage bool
	for _, item := range output {
		switch item["type"] {
		case "function_call":
			hasFunctionCall = item["name"] == "get_weather"
		case "message":
			content := item["content"].([]any)
			part := content[0].(map[string]any)
			hasFinalMessage = part["text"] == "done"
		}
	}
	if !hasFunctionCall || !hasFinalMessage {
		t.Fatalf("terminal output missing mixed call or final message: %#v", output)
	}
}

func completedResponseOutput(t *testing.T, body string) []map[string]any {
	t.Helper()
	for _, block := range strings.Split(body, "\n\n") {
		if !strings.HasPrefix(block, "event: response.completed\n") {
			continue
		}
		dataAt := strings.Index(block, "\ndata: ")
		if dataAt < 0 {
			t.Fatalf("response.completed event has no data: %s", block)
		}
		var payload struct {
			Response struct {
				Output []map[string]any `json:"output"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(block[dataAt+7:]), &payload); err != nil {
			t.Fatalf("decode response.completed: %v: %s", err, block)
		}
		return payload.Response.Output
	}
	t.Fatalf("response.completed event not found: %s", body)
	return nil
}

func ptr[T any](v T) *T { return &v }
