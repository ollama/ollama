package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/openai"
)

func summaryResponse(t *testing.T, summary string, retained []string) []byte {
	t.Helper()
	arguments, err := json.Marshal(map[string]any{"summary": summary, "retain_item_ids": retained})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"id": "resp_summary", "object": "response", "status": "completed", "model": "fixture",
		"output": []any{map[string]any{
			"id": "fc_summary", "type": "function_call", "status": "completed", "call_id": "call_summary",
			"name": openai.CreateSummaryToolName, "arguments": string(arguments),
		}},
		"usage": map[string]any{
			"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
			"input_tokens_details":  map[string]any{"cached_tokens": 0},
			"output_tokens_details": map[string]any{"reasoning_tokens": 0},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

type compactionUpstreamCapture struct {
	mu     sync.Mutex
	paths  []string
	bodies [][]byte
}

func (c *compactionUpstreamCapture) add(path string, body []byte) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.paths = append(c.paths, path)
	c.bodies = append(c.bodies, append([]byte(nil), body...))
	return len(c.bodies)
}

func (c *compactionUpstreamCapture) snapshot() ([]string, [][]byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	paths := append([]string(nil), c.paths...)
	bodies := make([][]byte, len(c.bodies))
	for i := range c.bodies {
		bodies[i] = append([]byte(nil), c.bodies[i]...)
	}
	return paths, bodies
}

func newCompactionTestServer(t *testing.T, handler func(int, http.ResponseWriter, *http.Request, []byte)) (*httptest.Server, *compactionUpstreamCapture) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	capture := &compactionUpstreamCapture{}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		attempt := capture.add(r.URL.Path, body)
		handler(attempt, w, r, body)
	}))
	t.Cleanup(upstream.Close)

	original := cloudProxyBaseURL
	cloudProxyBaseURL = upstream.URL
	t.Cleanup(func() { cloudProxyBaseURL = original })

	s := &Server{}
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	t.Cleanup(local.Close)
	return local, capture
}

func postCompactionRequest(t *testing.T, server *httptest.Server, path, body string) (*http.Response, []byte) {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, server.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return response, responseBody
}

func TestResponsesCompactUsesOrdinarySelectedCloudModel(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(summaryResponse(t, "Continue the task.", nil))
	})

	response, body := postCompactionRequest(t, local, "/v1/responses/compact", `{
		"model":"fixture:cloud",
		"instructions":"original agent instructions",
		"input":[{"type":"message","role":"user","content":"hello"}],
		"tools":[{"type":"function","name":"shell","description":"Run a command","strict":false,"parameters":{"type":"object"}}]
	}`)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.StatusCode, body)
	}
	var compacted openai.ResponsesCompactedResponse
	if err := json.Unmarshal(body, &compacted); err != nil {
		t.Fatal(err)
	}
	if compacted.Object != "response.compaction" || len(compacted.Output) != 1 || compacted.Output[0].Type != "compaction" {
		t.Fatalf("unexpected compact response: %+v", compacted)
	}

	paths, bodies := capture.snapshot()
	if len(paths) != 1 || paths[0] != "/v1/responses" {
		t.Fatalf("compaction must use one ordinary Responses inference call, paths=%v", paths)
	}
	if bytes.Contains(bodies[0], []byte("original agent instructions")) {
		t.Fatalf("top-level instructions leaked to compactor: %s", bodies[0])
	}
	var summaryRequest struct {
		Model  string                 `json:"model"`
		Stream bool                   `json:"stream"`
		Tools  []openai.ResponsesTool `json:"tools"`
	}
	if err := json.Unmarshal(bodies[0], &summaryRequest); err != nil {
		t.Fatal(err)
	}
	if summaryRequest.Model != "fixture" || summaryRequest.Stream {
		t.Fatalf("unexpected upstream summary request: %+v", summaryRequest)
	}
	if len(summaryRequest.Tools) != 1 || summaryRequest.Tools[0].Name != openai.CreateSummaryToolName {
		t.Fatalf("unexpected callable tools: %+v", summaryRequest.Tools)
	}
	if !bytes.Contains(bodies[0], []byte("shell")) {
		t.Fatalf("original tool metadata missing from transcript: %s", bodies[0])
	}
}

func TestResponsesCompactionTriggerReturnsCodexStream(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(summaryResponse(t, "Compact summary.", nil))
	})

	response, body := postCompactionRequest(t, local, "/v1/responses", `{
		"model":"fixture:cloud","stream":true,
		"input":[{"type":"message","role":"user","content":"hello"},{"type":"compaction_trigger"}]
	}`)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.StatusCode, body)
	}
	if got := response.Header.Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
		t.Fatalf("content-type=%q", got)
	}
	text := string(body)
	if strings.Count(text, "event: response.output_item.done") != 1 || strings.Count(text, `"type":"compaction"`) == 0 {
		t.Fatalf("missing single compaction output item: %s", text)
	}
	if strings.Count(text, "event: response.completed") != 1 {
		t.Fatalf("missing response.completed: %s", text)
	}
	paths, requests := capture.snapshot()
	if len(paths) != 1 || paths[0] != "/v1/responses" {
		t.Fatalf("paths=%v requests=%s", paths, requests)
	}
}

func TestResponsesCompactionRepairsMalformedSummaryOnce(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(attempt int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		if attempt == 1 {
			_, _ = w.Write([]byte(`{"id":"bad","object":"response","output":[{"type":"message","role":"assistant","content":[]}]}`))
			return
		}
		_, _ = w.Write(summaryResponse(t, "Repaired summary.", nil))
	})

	response, body := postCompactionRequest(t, local, "/v1/responses/compact", `{"model":"fixture:cloud","input":"hello"}`)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.StatusCode, body)
	}
	_, bodies := capture.snapshot()
	if len(bodies) != 2 {
		t.Fatalf("expected one repair retry, got %d requests", len(bodies))
	}
	if !bytes.Contains(bodies[1], []byte("previous create_summary call was invalid")) {
		t.Fatalf("repair request does not explain the validation error: %s", bodies[1])
	}
}

func TestResponsesCompactionFailsAfterOneRepair(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"bad","object":"response","output":[]}`))
	})

	response, body := postCompactionRequest(t, local, "/v1/responses/compact", `{"model":"fixture:cloud","input":"hello"}`)
	if response.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status=%d body=%s", response.StatusCode, body)
	}
	var errorResponse openai.ErrorResponse
	if err := json.Unmarshal(body, &errorResponse); err != nil {
		t.Fatal(err)
	}
	if errorResponse.Error.Code == nil || *errorResponse.Error.Code != "compaction_failed" {
		t.Fatalf("unexpected error: %+v", errorResponse)
	}
	paths, _ := capture.snapshot()
	if len(paths) != 2 {
		t.Fatalf("expected exactly two attempts, got %d", len(paths))
	}
}

func TestResponsesCompactionPayloadIsExpandedBeforeCloudPassthrough(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_next","object":"response","status":"completed","model":"fixture","output":[],"usage":null}`))
	})
	payload, err := json.Marshal(openai.OllamaCompactionPayload{
		Type: openai.OllamaCompactionPayloadType, Version: openai.OllamaCompactionPayloadVersion,
		Summary: "The build is ready.",
	})
	if err != nil {
		t.Fatal(err)
	}
	request, err := json.Marshal(map[string]any{
		"model": "fixture:cloud", "stream": false,
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "first retained user turn"},
			map[string]any{"type": "message", "role": "user", "content": "second retained user turn"},
			openai.ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(payload)},
			map[string]any{"type": "message", "role": "user", "content": "new turn"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	response, body := postCompactionRequest(t, local, "/v1/responses", string(request))
	if response.StatusCode != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.StatusCode, body)
	}
	paths, bodies := capture.snapshot()
	if len(paths) != 1 || paths[0] != "/v1/responses" {
		t.Fatalf("paths=%v", paths)
	}
	forwarded := string(bodies[0])
	if strings.Contains(forwarded, `"type":"compaction"`) {
		t.Fatalf("compaction boundary was forwarded: %s", forwarded)
	}
	for _, marker := range []string{"first retained user turn", "second retained user turn"} {
		if strings.Count(forwarded, marker) != 1 {
			t.Fatalf("retained user message %q was lost or duplicated: %s", marker, forwarded)
		}
	}
	if !strings.Contains(forwarded, "The build is ready.") || !strings.Contains(forwarded, "new turn") || !strings.Contains(forwarded, "ollama_compaction_summary") {
		t.Fatalf("expanded state is incomplete: %s", forwarded)
	}
}
