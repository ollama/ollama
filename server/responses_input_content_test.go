package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestResponsesCloudPreservesInputTypes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "false")
	for _, reject := range []bool{false, true} {
		t.Run(fmt.Sprintf("upstream_rejects=%t", reject), func(t *testing.T) {
			input := `[
				{"role":"user","content":"Before."},
				{"type":"reasoning","id":"rs_test","content":[{"type":"reasoning_text","text":"Keep <note> intact."}]},
				{"type":"custom_tool_call","id":"fc_test","call_id":"call_test","name":"read_note","input":"note.txt"},
				{"role":"assistant","content":[{"type":"refusal","refusal":"I cannot delete the note."}]},
				{"type":"custom_tool_call_output","call_id":"call_test","output":"note contents"},
				{"role":"user","content":"Continue."}
			]`
			status, response := http.StatusOK, `{"id":"resp_test","object":"response","status":"completed","output":[]}`
			if reject {
				input = strings.TrimSuffix(input, "]") + `,{"type":"future_item","payload":"keep-me"}]`
				status, response = http.StatusBadRequest, `{"error":{"type":"invalid_request_error","message":"unknown input item type: future_item"}}`
			}
			var want any
			if err := json.Unmarshal([]byte(input), &want); err != nil {
				t.Fatal(err)
			}
			calls := 0
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					return
				}
				var request struct {
					Model string `json:"model"`
					Input any    `json:"input"`
				}
				if err := json.Unmarshal(body, &request); err != nil {
					t.Error(err)
				}
				if r.URL.Path != "/v1/responses" || request.Model != "test" || !reflect.DeepEqual(want, request.Input) {
					t.Errorf("cloud request changed: path=%s body=%s", r.URL.Path, body)
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(status)
				_, _ = io.WriteString(w, response)
			}))
			defer upstream.Close()
			original := cloudProxyBaseURL
			cloudProxyBaseURL = upstream.URL
			t.Cleanup(func() { cloudProxyBaseURL = original })
			s := &Server{}
			router, err := s.GenerateRoutes()
			if err != nil {
				t.Fatal(err)
			}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test:cloud","input":`+input+`}`))
			req.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)
			if calls != 1 || recorder.Code != status || recorder.Body.String() != response {
				t.Fatalf("cloud response changed: calls=%d status=%d body=%s", calls, recorder.Code, recorder.Body.String())
			}
		})
	}
}
