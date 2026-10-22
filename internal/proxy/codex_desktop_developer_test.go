package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestCodexDesktopPreservesInstructionMessagesForOllama(t *testing.T) {
	const input = `[
        {"type":"message","role":"system","content":"Base policy"},
        {"role":"developer","content":"Use the scratch project."},
        {"role":"user","content":"Read the fixture"},
        {"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{}"},
        {"type":"function_call_output","call_id":"call_1","output":"fixture contents"},
        {"type":"message","id":"dev_2","role":"developer","content":[{"type":"input_text","text":"Preserve the fixture."}],"metadata":{"turn":"later"}},
        {"role":"user","content":"Continue"}
    ]`
	var want []map[string]any
	if err := json.Unmarshal([]byte(input), &want); err != nil {
		t.Fatal(err)
	}
	for _, endpoint := range []string{"/v1/responses", "/v1/responses/compact"} {
		t.Run(endpoint, func(t *testing.T) {
			var body []byte
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ = io.ReadAll(r.Body)
				w.WriteHeader(http.StatusOK)
			}))
			defer upstream.Close()
			handler := newTestCodexDesktop(t, upstream.URL, "http://127.0.0.1:1", writeCatalog(t, "qwen3.5:latest"))
			request := httptest.NewRequest(http.MethodPost, CodexDesktopPathPrefix+endpoint,
				strings.NewReader(`{"model":"qwen3.5:latest","input":`+input+`}`))
			request.RemoteAddr = "127.0.0.1:1234"
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", response.Code, response.Body.String())
			}
			var got struct {
				Input []map[string]any `json:"input"`
			}
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Input, want) {
				t.Fatalf("gateway changed instruction roles, contents, metadata, or tool history: %s", body)
			}
		})
	}
}
