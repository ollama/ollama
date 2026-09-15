package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/internal/proxy"
	"github.com/ollama/ollama/openai"
)

func TestResponsesHistoryCloudAndCodexContinuation(t *testing.T) {
	for _, codex := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("codex=%t/stream=%t", codex, stream), func(t *testing.T) {
				local, capture := newCompactionTestServer(t, func(attempt int, w http.ResponseWriter, r *http.Request, body []byte) {
					var req struct {
						Model    string            `json:"model"`
						Store    bool              `json:"store"`
						Previous string            `json:"previous_response_id"`
						Input    []json.RawMessage `json:"input"`
					}
					if err := json.Unmarshal(body, &req); err != nil {
						t.Error(err)
						w.WriteHeader(500)
						return
					}
					if req.Model != "fixture" || req.Store || req.Previous != "" || r.Header.Get("Accept-Encoding") != "identity" {
						t.Errorf("incorrect upstream request: %s", body)
					}
					if attempt == 1 {
						response := `{"id":"upstream_id","object":"response","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_check","name":"check","arguments":"{}"}]}`
						if stream {
							w.Header().Set("Content-Type", "text/event-stream")
							_, _ = fmt.Fprint(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"upstream_id\",\"status\":\"in_progress\",\"output\":[]}}\n\n")
							w.(http.Flusher).Flush()
							_, _ = fmt.Fprintf(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":%s}\n\n", response)
							w.(http.Flusher).Flush()
						} else {
							w.Header().Set("Content-Type", "application/json")
							_, _ = io.WriteString(w, response)
						}
						return
					}
					if len(req.Input) != 3 || !strings.Contains(string(req.Input[0]), "reply DONE") || !strings.Contains(string(req.Input[1]), `"call_id":"call_check"`) || !strings.Contains(string(req.Input[2]), `"output":"passed"`) {
						t.Errorf("lost continued context: %s", body)
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"upstream_id","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"DONE"}]}]}`)
				})
				endpoint, path := local, "/v1/responses"
				if codex {
					catalog := filepath.Join(t.TempDir(), proxy.CodexDesktopRoutingCatalogFilename)
					if err := os.WriteFile(catalog, []byte(`{"models":[{"slug":"fixture:cloud"}]}`), 0o600); err != nil {
						t.Fatal(err)
					}
					handler, err := proxy.NewCodexDesktop(proxy.CodexDesktopConfig{OllamaURL: local.URL, ChatGPTURL: local.URL, OpenAIURL: local.URL, RoutingCatalogPath: catalog})
					if err != nil {
						t.Fatal(err)
					}
					endpoint = httptest.NewServer(handler)
					t.Cleanup(endpoint.Close)
					path = proxy.CodexDesktopPathPrefix + "/v1/responses"
				}
				status, _, body := postCompactionRequest(t, endpoint, path, fmt.Sprintf(`{"model":"fixture:cloud","input":"Call check, then reply DONE.","stream":%t}`, stream))
				if status != 200 {
					t.Fatalf("first status=%d body=%s", status, body)
				}
				var first openai.ResponsesResponse
				if stream {
					for _, line := range strings.Split(string(body), "\n") {
						data, ok := strings.CutPrefix(line, "data: ")
						if !ok {
							continue
						}
						var event struct {
							Type     string                   `json:"type"`
							Response openai.ResponsesResponse `json:"response"`
						}
						if err := json.Unmarshal([]byte(data), &event); err != nil {
							t.Fatal(err)
						}
						if event.Type == "response.completed" {
							first = event.Response
						}
					}
				} else if err := json.Unmarshal(body, &first); err != nil {
					t.Fatal(err)
				}
				if first.ID == "" || first.ID == "upstream_id" || !first.Store {
					t.Fatalf("incorrect stored response: %+v", first)
				}
				status, _, body = postCompactionRequest(t, endpoint, path, fmt.Sprintf(`{"model":"fixture:cloud","previous_response_id":%q,"input":[{"type":"function_call_output","call_id":"call_check","output":"passed"}]}`, first.ID))
				if status != 200 {
					t.Fatalf("follow-up status=%d body=%s", status, body)
				}
				var second openai.ResponsesResponse
				if err := json.Unmarshal(body, &second); err != nil {
					t.Fatal(err)
				}
				if second.PreviousResponseID == nil || *second.PreviousResponseID != first.ID || !strings.Contains(string(body), "DONE") {
					t.Fatalf("incorrect follow-up: %s", body)
				}
				status, _, body = postCompactionRequest(t, endpoint, path, `{"model":"fixture:cloud","previous_response_id":"resp_missing","input":"hi"}`)
				if status != 404 || !strings.Contains(string(body), "previous_response_not_found") {
					t.Fatalf("unknown ID not rejected: %d %s", status, body)
				}
				paths, _ := capture.snapshot()
				if len(paths) != 2 {
					t.Fatalf("unexpected upstream calls: %v", paths)
				}
			})
		}
	}
}

func TestResponsesHistoryAfterCompaction(t *testing.T) {
	local, capture := newCompactionTestServer(t, func(attempt int, w http.ResponseWriter, _ *http.Request, body []byte) {
		w.Header().Set("Content-Type", "application/json")
		if attempt == 1 {
			_, _ = w.Write(summaryResponse(t, "Remember the original goal.", nil))
			return
		}
		if strings.Contains(string(body), `"type":"compaction_trigger"`) || strings.Contains(string(body), `"name":"create_summary"`) {
			t.Errorf("continuation triggered compaction again: %s", body)
		}
		if !strings.Contains(string(body), "Remember the original goal.") || !strings.Contains(string(body), "Continue now") {
			t.Errorf("missing compacted history: %s", body)
		}
		_, _ = io.WriteString(w, `{"id":"after_compaction","status":"completed","output":[]}`)
	})
	status, _, body := postCompactionRequest(t, local, "/v1/responses", `{"model":"fixture:cloud","stream":true,"input":[{"type":"message","role":"user","content":"Original goal"},{"type":"compaction_trigger"}]}`)
	if status != 200 {
		t.Fatalf("compaction failed: %d %s", status, body)
	}
	var id string
	for _, line := range strings.Split(string(body), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var event struct {
			Type     string `json:"type"`
			Response struct {
				ID    string `json:"id"`
				Store bool   `json:"store"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			t.Fatal(err)
		}
		if event.Type == "response.completed" {
			id = event.Response.ID
			if !event.Response.Store {
				t.Fatal("compaction not stored")
			}
		}
	}
	if id == "" {
		t.Fatal("no completed compaction response")
	}
	status, _, body = postCompactionRequest(t, local, "/v1/responses", fmt.Sprintf(`{"model":"fixture:cloud","previous_response_id":%q,"input":"Continue now"}`, id))
	if status != 200 {
		t.Fatalf("continuation failed: %d %s", status, body)
	}
	paths, _ := capture.snapshot()
	if len(paths) != 2 {
		t.Fatalf("unexpected upstream requests: %v", paths)
	}
}
