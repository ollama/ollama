package middleware

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

func historyRequest(t *testing.T, router http.Handler, body, auth string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	return w
}

func historyResponseID(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var response struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(response.ID, "resp_") {
		t.Fatalf("invalid response ID: %q", response.ID)
	}
	return response.ID
}

func TestResponsesHistoryLocalContinuation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var requests []api.ChatRequest
	router := gin.New()
	router.POST("/v1/responses", ResponsesHistoryMiddleware(), ResponsesMiddleware(), func(c *gin.Context) {
		var req api.ChatRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			t.Fatal(err)
		}
		requests = append(requests, req)
		response := api.ChatResponse{Model: req.Model, CreatedAt: time.Now(), Done: true, Message: api.Message{Role: "assistant", Content: "DONE"}}
		if len(requests) == 1 {
			response.Message.Content = ""
			response.Message.Thinking = "I should call the tool."
			response.Message.ToolCalls = []api.ToolCall{{ID: "call_check", Function: api.ToolCallFunction{Name: "check", Arguments: api.NewToolCallFunctionArguments()}}}
		}
		c.JSON(http.StatusOK, response)
	})
	first := historyRequest(t, router, `{"model":"test","input":"Call check, then reply DONE.","instructions":"old instructions"}`, "")
	id := historyResponseID(t, first)
	var result openai.ResponsesResponse
	if err := json.Unmarshal(first.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.Store {
		t.Fatal("responses should be stored by default")
	}
	var callID string
	for _, item := range result.Output {
		if item.Type == "function_call" {
			callID = item.CallID
		}
	}
	if callID == "" {
		t.Fatal("no tool call")
	}
	second := historyRequest(t, router, fmt.Sprintf(`{"model":"test","previous_response_id":%q,"input":[{"type":"function_call_output","call_id":%q,"output":"passed"}],"instructions":"new instructions"}`, id, callID), "")
	secondID := historyResponseID(t, second)
	if secondID == id {
		t.Fatal("response IDs reused")
	}
	if err := json.Unmarshal(second.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.PreviousResponseID == nil || *result.PreviousResponseID != id {
		t.Fatalf("previous response ID = %v", result.PreviousResponseID)
	}
	got := requests[1].Messages
	if len(got) != 4 || got[0].Content != "new instructions" || got[1].Content != "Call check, then reply DONE." || got[2].Role != "assistant" || len(got[2].ToolCalls) != 1 || got[3].Role != "tool" || got[3].Content != "passed" {
		t.Fatalf("incorrect continued messages: %+v", got)
	}
	if got[2].Thinking != "I should call the tool." {
		t.Fatalf("lost reasoning: %+v", got[2])
	}
	third := historyRequest(t, router, fmt.Sprintf(`{"model":"test","previous_response_id":%q,"input":"Thanks"}`, secondID), "")
	historyResponseID(t, third)
	if got := requests[2].Messages; len(got) != 5 || got[0].Role != "user" || got[3].Content != "DONE" || got[4].Content != "Thanks" {
		t.Fatalf("incorrect third turn: %+v", got)
	}
	// A second branch from the first turn must not include the first branch.
	branch := historyRequest(t, router, fmt.Sprintf(`{"model":"test","previous_response_id":%q,"input":"Different follow-up"}`, id), "")
	historyResponseID(t, branch)
	if got := requests[3].Messages; len(got) != 3 || got[2].Content != "Different follow-up" {
		t.Fatalf("branch mutated history: %+v", got)
	}
}

func TestResponsesHistoryRawPassthrough(t *testing.T) {
	gin.SetMode(gin.TestMode)
	var requests []map[string]json.RawMessage
	router := gin.New()
	router.POST("/v1/responses", ResponsesHistoryMiddleware(), func(c *gin.Context) {
		var req map[string]json.RawMessage
		if err := c.ShouldBindJSON(&req); err != nil {
			t.Fatal(err)
		}
		requests = append(requests, req)
		if _, ok := req["previous_response_id"]; ok {
			t.Fatal("previous ID must not reach the upstream")
		}
		if string(req["store"]) != "false" {
			t.Fatal("upstream storage must be disabled")
		}
		c.Data(http.StatusOK, "application/json", []byte(`{"id":"upstream_reused_id","object":"response","status":"completed","store":false,"output":[{"id":"reason_1","type":"reasoning","encrypted_content":"opaque"},{"id":"fc_1","type":"function_call","call_id":"call_1","name":"check","arguments":"{}","custom":"preserve"}]}`))
	})
	id := historyResponseID(t, historyRequest(t, router, `{"model":"test:cloud","input":"check","custom_request":{"a":1},"instructions":"old"}`, ""))
	second := historyRequest(t, router, fmt.Sprintf(`{"model":"test:cloud","previous_response_id":%q,"input":[{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_image","image_url":"data:image/png;base64,AAAA"}]}],"custom_request":{"a":2}}`, id), "")
	if next := historyResponseID(t, second); next == id || next == "upstream_reused_id" {
		t.Fatal("must assign unique local response IDs")
	}
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(requests[1]["input"], &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 4 || string(items[1]["encrypted_content"]) != `"opaque"` || string(items[2]["custom"]) != `"preserve"` || !bytes.Contains(items[3]["output"], []byte("input_image")) {
		t.Fatalf("lost wire items: %s", requests[1]["input"])
	}
	if _, ok := requests[1]["instructions"]; ok {
		t.Fatal("prior instructions were carried over")
	}
	if string(requests[1]["custom_request"]) != `{"a":2}` {
		t.Fatalf("lost custom request field: %s", requests[1]["custom_request"])
	}
}

func TestResponsesHistoryStreaming(t *testing.T) {
	for _, crlf := range []bool{false, true} {
		for _, chunkSize := range []int{1, 10000} {
			t.Run(fmt.Sprintf("crlf=%t/chunk=%d", crlf, chunkSize), func(t *testing.T) {
				gin.SetMode(gin.TestMode)
				history := newResponsesHistory()
				router := gin.New()
				router.POST("/v1/responses", history.middleware(), func(c *gin.Context) {
					c.Header("Content-Type", "text/event-stream")
					stream := "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"old\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
						"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"response_id\":\"old\",\"delta\":\"hello\"}\n\n" +
						"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"old\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":\"hello\"}]}}\n\n" + "data: [DONE]\n\n"
					if crlf {
						stream = strings.ReplaceAll(stream, "\n", "\r\n")
					}
					for len(stream) > 0 {
						n := min(chunkSize, len(stream))
						if _, err := c.Writer.Write([]byte(stream[:n])); err != nil {
							t.Fatal(err)
						}
						c.Writer.Flush()
						stream = stream[n:]
					}
				})
				w := historyRequest(t, router, `{"input":"hi","stream":true}`, "")
				if w.Code != 200 || !w.Flushed {
					t.Fatalf("stream not flushed: %d", w.Code)
				}
				var id string
				for _, line := range strings.Split(w.Body.String(), "\n") {
					if !strings.HasPrefix(line, "data: {") {
						continue
					}
					var event struct {
						Type       string `json:"type"`
						ResponseID string `json:"response_id"`
						Response   *struct {
							ID    string `json:"id"`
							Store bool   `json:"store"`
						} `json:"response"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
						t.Fatal(err)
					}
					if event.Response != nil {
						if id == "" {
							id = event.Response.ID
						}
						if event.Response.ID != id || !event.Response.Store {
							t.Fatalf("inconsistent response: %+v", event.Response)
						}
					} else if event.ResponseID != id {
						t.Fatalf("inconsistent event ID: %q", event.ResponseID)
					}
				}
				if id == "old" || id == "" {
					t.Fatalf("bad response ID: %q", id)
				}
				stored, ok := history.get(responsesHistoryKey{id: id, credential: sha256Credential("")})
				if !ok || !bytes.Contains(stored, []byte("hello")) {
					t.Fatalf("stream not stored: %s", stored)
				}
			})
		}
	}
}

func sha256Credential(s string) [32]byte { return sha256.Sum256([]byte(s)) }

func TestResponsesHistoryStoreAndIsolation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	history := newResponsesHistory()
	calls := 0
	router := gin.New()
	router.POST("/v1/responses", history.middleware(), func(c *gin.Context) {
		calls++
		c.Data(200, "application/json", []byte(`{"id":"old","status":"completed","output":[]}`))
	})
	id := historyResponseID(t, historyRequest(t, router, `{"input":"hi","store":false}`, ""))
	w := historyRequest(t, router, fmt.Sprintf(`{"previous_response_id":%q,"input":"follow-up"}`, id), "")
	if w.Code != 404 || calls != 1 {
		t.Fatalf("store:false retained history: %d %d", w.Code, calls)
	}
	id = historyResponseID(t, historyRequest(t, router, `{"input":"hi","store":true}`, "Bearer a"))
	w = historyRequest(t, router, fmt.Sprintf(`{"previous_response_id":%q}`, id), "Bearer b")
	if w.Code != 404 || calls != 2 {
		t.Fatalf("cross-credential history exposed: %d %d", w.Code, calls)
	}
	w = historyRequest(t, router, fmt.Sprintf(`{"previous_response_id":%q,"store":false}`, id), "Bearer a")
	next := historyResponseID(t, w)
	if calls != 3 {
		t.Fatalf("continuation not allowed with store:false")
	}
	w = historyRequest(t, router, fmt.Sprintf(`{"previous_response_id":%q}`, next), "Bearer a")
	if w.Code != 404 {
		t.Fatalf("store:false continuation was saved")
	}
	for _, body := range []string{`{"previous_response_id":17}`, `{"previous_response_id":{}}`, `{"store":"yes"}`, `{"background":true}`, `{"conversation":"conv_1"}`, `{"input":2}`, `null`} {
		if w := historyRequest(t, router, body, ""); w.Code != 400 {
			t.Fatalf("expected 400 for %s, got %d: %s", body, w.Code, w.Body.String())
		}
	}
	historyResponseID(t, historyRequest(t, router, `{"previous_response_id":null,"store":null,"input":"hi"}`, ""))
}

func TestResponsesHistoryLimitsAndExpiry(t *testing.T) {
	h := newResponsesHistory()
	now := time.Now()
	h.now = func() time.Time { return now }
	h.maxEntries, h.maxBytes = 2, 8
	key := func(s string) responsesHistoryKey { return responsesHistoryKey{id: s} }
	if !h.put(key("a"), []byte(`[]`)) || !h.put(key("b"), []byte(`[1]`)) || !h.put(key("c"), []byte(`[2]`)) {
		t.Fatal("put failed")
	}
	if _, ok := h.get(key("a")); ok {
		t.Fatal("oldest entry not evicted")
	}
	if h.bytes != 6 || len(h.entries) != 2 {
		t.Fatalf("bad accounting: %d %d", h.bytes, len(h.entries))
	}
	if h.put(key("large"), []byte(`123456789`)) {
		t.Fatal("oversized entry stored")
	}
	now = now.Add(h.ttl)
	if _, ok := h.get(key("b")); ok {
		t.Fatal("expired entry returned")
	}
	if !h.put(key("d"), []byte(`[]`)) || h.bytes != 2 || len(h.entries) != 1 {
		t.Fatal("expired entries not evicted")
	}
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Go(func() { h.put(key("concurrent"), []byte(`[]`)); h.get(key("concurrent")) })
	}
	wg.Wait()
}

func TestResponsesHistoryZstdAndErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/v1/responses", ResponsesHistoryMiddleware(), func(c *gin.Context) {
		if c.GetHeader("Content-Encoding") != "" {
			t.Fatal("encoding not removed")
		}
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			t.Fatal(err)
		}
		if int64(len(body)) != c.Request.ContentLength {
			t.Fatal("incorrect request content length")
		}
		c.Data(400, "application/json", []byte(`{"error":{"message":"model not found"}}`))
	})
	enc, _ := zstd.NewWriter(nil)
	defer enc.Close()
	body := enc.EncodeAll([]byte(`{"input":"hi"}`), nil)
	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(body))
	req.Header.Set("Content-Encoding", "zstd")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	if w.Code != 400 || w.Body.String() != `{"error":{"message":"model not found"}}` {
		t.Fatalf("error not preserved: %d %s", w.Code, w.Body.String())
	}
}

func TestResponsesHistoryIncompleteAndFailedStreams(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tt := range []struct {
		name, response string
		stream, flush  bool
		wantStatus     int
		wantError      bool
	}{
		{"malformed JSON", `not JSON`, false, false, 502, true},
		{"incomplete event", "data: {\"type\":\"response.completed\",", true, false, 502, true},
		{"truncated flushed stream", "data: {\"type\":\"response.created\",\"response\":{\"id\":\"old\",\"status\":\"in_progress\"}}\n\n", true, true, 200, true},
		{"upstream error event", "event: error\ndata: {\"type\":\"error\",\"message\":\"upstream failure\"}\n\n", true, true, 200, false},
		{"failed response", `{"id":"failed","status":"failed","error":{"message":"failure"},"output":[]}`, false, false, 200, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			h := newResponsesHistory()
			router := gin.New()
			router.POST("/v1/responses", h.middleware(), func(c *gin.Context) {
				if tt.stream {
					c.Header("Content-Type", "text/event-stream")
				}
				_, _ = c.Writer.Write([]byte(tt.response))
				if tt.flush {
					c.Writer.Flush()
				}
			})
			w := historyRequest(t, router, fmt.Sprintf(`{"input":"hi","stream":%t}`, tt.stream), "")
			if w.Code != tt.wantStatus {
				t.Fatalf("status %d, want %d: %s", w.Code, tt.wantStatus, w.Body.String())
			}
			if h.order.Len() != 0 {
				t.Fatal("failed or incomplete response cached")
			}
			if tt.wantError && !bytes.Contains(w.Body.Bytes(), []byte("error")) {
				t.Fatalf("failure not surfaced: %s", w.Body.String())
			}
		})
	}
}

func TestResponsesHistoryStorageCapacityReported(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := newResponsesHistory()
	h.maxBytes = 1
	router := gin.New()
	router.POST("/v1/responses", h.middleware(), func(c *gin.Context) {
		c.Header("Content-Length", "12345")
		c.Data(200, "application/json", []byte(`{"id":"old","status":"completed","output":[]}`))
	})
	w := historyRequest(t, router, `{"input":"hi"}`, "")
	historyResponseID(t, w)
	if !bytes.Contains(w.Body.Bytes(), []byte(`"store":false`)) || h.order.Len() != 0 {
		t.Fatalf("incorrect storage status: %s", w.Body.String())
	}
	if w.Header().Get("Content-Length") != "" {
		t.Fatal("stale upstream content length")
	}
}
