package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/internal/proxy"
	"github.com/ollama/ollama/openai"
)

func TestIsCompactionContextLimit(t *testing.T) {
	for _, tt := range []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"reported cloud error", 400, `{"error":{"message":"The prompt is too long: 1068408, model maximum context length: 1048576 (ref: 00000000-0000-4000-8000-000000000001)","code":null}}`, true},
		{"without reference", 400, `{"error":{"message":"The prompt is too long: 100, model maximum context length: 90"}}`, true},
		{"structured code", 400, `{"error":{"message":"input too large","code":"context_length_exceeded"}}`, true},
		{"structured 413", 413, `{"error":{"code":"context_length_exceeded"}}`, true},
		{"wrong status", 500, `{"error":{"code":"context_length_exceeded"}}`, false},
		{"rate limit", 429, `{"error":{"code":"context_length_exceeded"}}`, false},
		{"other 413", 413, `{"error":{"message":"request body too large"}}`, false},
		{"other invalid request", 400, `{"error":{"message":"invalid tool schema","code":"invalid_request_error"}}`, false},
		{"untrusted embedded phrase", 400, `{"error":{"message":"Invalid input contains: The prompt is too long: 100, model maximum context length: 90"}}`, false},
		{"missing counts", 400, `{"error":{"message":"The prompt is too long: unknown, model maximum context length: unknown"}}`, false},
		{"invalid JSON", 400, `upstream failed`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			response := &responsesInferenceRecorder{status: tt.status, body: *bytes.NewBufferString(tt.body)}
			if got := isCompactionContextLimit(response); got != tt.want {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResponsesCompactionOverflowRecovery(t *testing.T) {
	const overflow = `{"error":{"message":"The prompt is too long: 1068408, model maximum context length: 1048576 (ref: fixture)","type":"invalid_request_error","code":null}}`
	const invalid = `{"id":"bad","object":"response","output":[]}`
	for _, tt := range []struct {
		name        string
		path        string
		responses   []string
		statuses    []int
		wantStatus  int
		wantTrimmed bool
	}{
		{"standalone", "/v1/responses/compact", []string{overflow, ""}, []int{400, 200}, 200, true},
		{"trigger", "/v1/responses", []string{overflow, ""}, []int{400, 200}, 200, true},
		{"overflow then repair", "/v1/responses/compact", []string{overflow, invalid, ""}, []int{400, 200, 200}, 200, true},
		{"repair then overflow", "/v1/responses/compact", []string{invalid, overflow, ""}, []int{200, 400, 200}, 200, true},
		{"overflow retry exhausted", "/v1/responses/compact", []string{overflow, overflow}, []int{400, 400}, 400, true},
		{"both retries exhausted", "/v1/responses/compact", []string{overflow, invalid, invalid}, []int{400, 200, 200}, 500, true},
		{"unrelated error", "/v1/responses/compact", []string{`{"error":{"message":"invalid tool schema"}}`}, []int{400}, 400, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			local, capture := newCompactionTestServer(t, func(attempt int, w http.ResponseWriter, _ *http.Request, _ []byte) {
				w.Header().Set("Content-Type", "application/json")
				if attempt > len(tt.responses) {
					t.Errorf("unexpected attempt %d", attempt)
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				w.WriteHeader(tt.statuses[attempt-1])
				if body := tt.responses[attempt-1]; body != "" {
					_, _ = io.WriteString(w, body)
				} else {
					_, _ = w.Write(summaryResponse(t, "Continue from the latest result.", nil))
				}
			})
			input := `[
				{"type":"message","role":"user","content":"original goal"},
				{"type":"function_call","call_id":"old","name":"shell","arguments":"{}"},
				{"type":"function_call_output","call_id":"old","output":"` + strings.Repeat("old output ", 1000) + `"},
				{"type":"message","role":"assistant","content":"old result processed"},
				{"type":"message","role":"user","content":"latest request"},
				{"type":"function_call","call_id":"latest","name":"shell","arguments":"{}"},
				{"type":"function_call_output","call_id":"latest","output":"latest result"}
			]`
			stream := tt.path == "/v1/responses"
			if stream {
				input = strings.TrimSuffix(input, "]") + `,{"type":"compaction_trigger"}]`
			}
			request := fmt.Sprintf(`{"model":"fixture:cloud","stream":%t,"input":%s}`, stream, input)
			status, _, body := postCompactionRequest(t, local, tt.path, request)
			if status != tt.wantStatus {
				t.Fatalf("status=%d, want %d: %s", status, tt.wantStatus, body)
			}
			_, requests := capture.snapshot()
			if len(requests) != len(tt.responses) {
				t.Fatalf("made %d attempts, want %d", len(requests), len(tt.responses))
			}
			if tt.wantTrimmed {
				last := requests[len(requests)-1]
				if len(last) >= len(requests[0]) || bytes.Contains(last, []byte("old output")) {
					t.Fatal("overflow retry did not shrink the old transcript")
				}
				for _, marker := range []string{"original goal", "latest request", "latest result", "2 older transcript items were omitted"} {
					if !bytes.Contains(last, []byte(marker)) {
						t.Errorf("retry prompt missing %q", marker)
					}
				}
			}
			if status == http.StatusOK {
				if !bytes.Contains(body, []byte("2 older transcript items were omitted")) || !bytes.Contains(body, []byte("latest result")) {
					t.Fatalf("compaction output lost omission notice or active result: %s", body)
				}
			} else if status == http.StatusBadRequest && string(body) != tt.responses[len(tt.responses)-1] {
				t.Fatalf("did not preserve upstream error: %s", body)
			}
		})
	}
}

func TestResponsesCompactionOverflowWithoutRemovableHistory(t *testing.T) {
	const overflow = `{"error":{"code":"context_length_exceeded","message":"too many tokens"}}`
	local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, overflow)
	})
	status, _, body := postCompactionRequest(t, local, "/v1/responses/compact", `{"model":"fixture:cloud","input":"oversized user message"}`)
	if status != http.StatusBadRequest || string(body) != overflow {
		t.Fatalf("status=%d body=%s", status, body)
	}
	_, requests := capture.snapshot()
	if len(requests) != 1 {
		t.Fatalf("made %d requests without removable history", len(requests))
	}
}

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

func postCompactionRequest(t *testing.T, server *httptest.Server, path, body string) (int, http.Header, []byte) {
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
	return response.StatusCode, response.Header.Clone(), responseBody
}

func TestResponsesCompactUsesOrdinarySelectedCloudModel(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(summaryResponse(t, "Continue the task.", nil))
	})

	status, _, body := postCompactionRequest(t, local, "/v1/responses/compact", `{
		"model":"fixture:cloud",
		"instructions":"original agent instructions",
		"input":[{"type":"message","role":"user","content":"hello"}],
		"tools":[{"type":"function","name":"shell","description":"Run a command","strict":false,"parameters":{"type":"object"}}]
	}`)
	if status != http.StatusOK {
		t.Fatalf("status=%d body=%s", status, body)
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

func TestResponsesCompactionPreservesSelectedImageIntoNextTurn(t *testing.T) {
	const imageURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	local, capture := newCompactionTestServer(t, func(attempt int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		if attempt == 1 {
			_, _ = w.Write(summaryResponse(t, "Keep the source image available.", []string{"item_000001"}))
			return
		}
		_, _ = w.Write([]byte(`{"id":"resp_next","object":"response","status":"completed","model":"fixture","output":[],"usage":null}`))
	})

	status, _, body := postCompactionRequest(t, local, "/v1/responses/compact", `{
		"model":"fixture:cloud",
		"input":[{"type":"message","role":"user","content":[
			{"type":"input_text","text":"inspect this image"},
			{"type":"input_image","detail":"auto","image_url":"`+imageURL+`"}
		]}]
	}`)
	if status != http.StatusOK {
		t.Fatalf("compact status=%d body=%s", status, body)
	}
	var compacted openai.ResponsesCompactedResponse
	if err := json.Unmarshal(body, &compacted); err != nil {
		t.Fatal(err)
	}
	if len(compacted.Output) != 1 {
		t.Fatalf("unexpected compact response: %+v", compacted)
	}

	request, err := json.Marshal(map[string]any{
		"model": "fixture:cloud", "stream": false,
		"input": []any{
			compacted.Output[0],
			map[string]any{"type": "message", "role": "user", "content": "what was in it?"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	status, _, body = postCompactionRequest(t, local, "/v1/responses", string(request))
	if status != http.StatusOK {
		t.Fatalf("follow-up status=%d body=%s", status, body)
	}

	paths, bodies := capture.snapshot()
	if len(paths) != 2 || paths[0] != "/v1/responses" || paths[1] != "/v1/responses" {
		t.Fatalf("unexpected upstream requests: %v", paths)
	}
	var summaryRequest struct {
		Input []struct {
			Content json.RawMessage `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(bodies[0], &summaryRequest); err != nil {
		t.Fatal(err)
	}
	if len(summaryRequest.Input) != 2 {
		t.Fatalf("unexpected summary input: %s", bodies[0])
	}
	var blocks []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL string `json:"image_url"`
	}
	if err := json.Unmarshal(summaryRequest.Input[1].Content, &blocks); err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 2 || blocks[0].Type != "input_text" || blocks[1].Type != "input_image" || blocks[1].ImageURL != imageURL {
		t.Fatalf("compactor did not receive the source image as multimodal input: %+v", blocks)
	}
	if strings.Contains(blocks[0].Text, "iVBOR") {
		t.Fatalf("image bytes leaked into transcript text: %s", blocks[0].Text)
	}

	forwarded := string(bodies[1])
	if strings.Contains(forwarded, `"type":"compaction"`) {
		t.Fatalf("opaque compaction item reached the next model: %s", forwarded)
	}
	if strings.Count(forwarded, imageURL) != 1 || !strings.Contains(forwarded, `"type":"input_image"`) {
		t.Fatalf("selected image was not replayed exactly once on the next turn: %s", forwarded)
	}
}

func TestResponsesCompactionTriggerReturnsCodexStream(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(_ int, w http.ResponseWriter, _ *http.Request, _ []byte) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(summaryResponse(t, "Compact summary.", nil))
	})

	status, header, body := postCompactionRequest(t, local, "/v1/responses", `{
		"model":"fixture:cloud","stream":true,
		"input":[{"type":"message","role":"user","content":"hello"},{"type":"compaction_trigger"}]
	}`)
	if status != http.StatusOK {
		t.Fatalf("status=%d body=%s", status, body)
	}
	if got := header.Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
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

func TestResponsesCompactionPreservesStandaloneOutputIntoNextTurn(t *testing.T) {
	const imageURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
	for _, tt := range []struct {
		name      string
		stream    bool
		namespace string
		nullID    bool
	}{
		{name: "Codex trigger with namespaced handoff", stream: true, namespace: "workspace"},
		{name: "compact endpoint with null call ID", nullID: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			local, capture := newCompactionTestServer(t, func(attempt int, w http.ResponseWriter, _ *http.Request, _ []byte) {
				w.Header().Set("Content-Type", "application/json")
				if attempt%2 == 1 {
					// The compactor does not select the handoff for retention.
					_, _ = w.Write(summaryResponse(t, "Continue the task.", nil))
					return
				}
				_, _ = io.WriteString(w, `{"id":"resp_next","object":"response","status":"completed","model":"fixture","output":[],"usage":null}`)
			})
			endpoint, path := local, "/v1/responses/compact"
			if tt.stream {
				catalogPath := filepath.Join(t.TempDir(), proxy.CodexDesktopRoutingCatalogFilename)
				if err := os.WriteFile(catalogPath, []byte(`{"models":[{"slug":"fixture:cloud"}]}`), 0o600); err != nil {
					t.Fatal(err)
				}
				handler, err := proxy.NewCodexDesktop(proxy.CodexDesktopConfig{
					OllamaURL: local.URL, ChatGPTURL: local.URL, OpenAIURL: local.URL,
					RoutingCatalogPath: catalogPath,
				})
				if err != nil {
					t.Fatal(err)
				}
				endpoint = httptest.NewServer(handler)
				t.Cleanup(endpoint.Close)
				path = proxy.CodexDesktopPathPrefix + "/v1/responses"
			}

			output := []any{
				map[string]any{"type": "input_text", "text": "Use the supplied architecture diagram."},
				map[string]any{"type": "input_image", "detail": "auto", "image_url": imageURL},
			}
			standalone := map[string]any{"type": "function_call_output", "name": "handoff", "output": output}
			if tt.namespace != "" {
				standalone["namespace"] = tt.namespace
			}
			if tt.nullID {
				standalone["call_id"] = nil
			}
			input := []any{
				map[string]any{"type": "message", "role": "user", "content": "Implement the feature."},
				standalone,
				map[string]any{"type": "message", "role": "assistant", "content": "I have read the handoff."},
				map[string]any{"type": "message", "role": "user", "content": "Continue."},
			}
			for cycle := range 2 {
				compactInput := append([]any(nil), input...)
				if tt.stream {
					compactInput = append(compactInput, map[string]any{"type": "compaction_trigger"})
				}
				request, err := json.Marshal(map[string]any{"model": "fixture:cloud", "stream": tt.stream, "input": compactInput})
				if err != nil {
					t.Fatal(err)
				}
				status, header, body := postCompactionRequest(t, endpoint, path, string(request))
				if status != http.StatusOK {
					t.Fatalf("cycle %d compact status=%d body=%s", cycle, status, body)
				}
				var compacted openai.ResponsesCompactionItem
				if tt.stream {
					if !strings.HasPrefix(header.Get("Content-Type"), "text/event-stream") {
						t.Fatalf("unexpected stream content-type %q", header.Get("Content-Type"))
					}
					done := 0
					for _, line := range strings.Split(string(body), "\n") {
						data, ok := strings.CutPrefix(line, "data: ")
						if !ok || data == "[DONE]" {
							continue
						}
						var event struct {
							Type string                         `json:"type"`
							Item openai.ResponsesCompactionItem `json:"item"`
						}
						if err := json.Unmarshal([]byte(data), &event); err != nil {
							t.Fatal(err)
						}
						if event.Type == "response.output_item.done" {
							compacted = event.Item
							done++
						}
					}
					if done != 1 {
						t.Fatalf("expected one completed compaction item, got %d: %s", done, body)
					}
				} else {
					var response openai.ResponsesCompactedResponse
					if err := json.Unmarshal(body, &response); err != nil {
						t.Fatal(err)
					}
					if len(response.Output) != 1 {
						t.Fatalf("expected one compaction item: %s", body)
					}
					compacted = response.Output[0]
				}
				if compacted.Type != "compaction" {
					t.Fatalf("unexpected output item: %+v", compacted)
				}

				input = []any{compacted, map[string]any{"type": "message", "role": "user", "content": "Continue."}}
				request, err = json.Marshal(map[string]any{"model": "fixture:cloud", "stream": false, "input": input})
				if err != nil {
					t.Fatal(err)
				}
				status, _, body = postCompactionRequest(t, endpoint, strings.TrimSuffix(path, "/compact"), string(request))
				if status != http.StatusOK {
					t.Fatalf("cycle %d replay status=%d body=%s", cycle, status, body)
				}
				paths, bodies := capture.snapshot()
				if len(paths) != 2*(cycle+1) || paths[len(paths)-1] != "/v1/responses" {
					t.Fatalf("unexpected upstream requests: %v", paths)
				}
				var forwarded struct {
					Input []map[string]any `json:"input"`
				}
				if err := json.Unmarshal(bodies[len(bodies)-1], &forwarded); err != nil {
					t.Fatal(err)
				}
				outputs := 0
				var summaryCallID string
				for _, item := range forwarded.Input {
					if item["type"] == "compaction" {
						t.Fatalf("opaque compaction item reached upstream: %+v", item)
					}
					if item["type"] == "function_call" {
						if item["name"] != "ollama_compaction_summary" {
							t.Fatalf("unexpected synthetic function call: %+v", item)
						}
						summaryCallID, _ = item["call_id"].(string)
						continue
					}
					if item["type"] != "function_call_output" {
						continue
					}
					if summaryCallID != "" && item["call_id"] == summaryCallID {
						continue
					}
					outputs++
					if item["call_id"] != nil || item["name"] != "handoff" || !reflect.DeepEqual(item["output"], output) {
						t.Fatalf("standalone output changed during replay: %+v", item)
					}
					if namespace, _ := item["namespace"].(string); namespace != tt.namespace {
						t.Fatalf("namespace=%q, want %q", namespace, tt.namespace)
					}
				}
				if outputs != 1 {
					t.Fatalf("expected one replayed standalone output, got %d: %s", outputs, bodies[len(bodies)-1])
				}
			}
		})
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

	status, _, body := postCompactionRequest(t, local, "/v1/responses/compact", `{"model":"fixture:cloud","input":"hello"}`)
	if status != http.StatusOK {
		t.Fatalf("status=%d body=%s", status, body)
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

	status, _, body := postCompactionRequest(t, local, "/v1/responses/compact", `{"model":"fixture:cloud","input":"hello"}`)
	if status != http.StatusInternalServerError {
		t.Fatalf("status=%d body=%s", status, body)
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

	status, _, body := postCompactionRequest(t, local, "/v1/responses", string(request))
	if status != http.StatusOK {
		t.Fatalf("status=%d body=%s", status, body)
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
