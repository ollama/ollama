package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

// streamThenFail mirrors server.streamResponse when an error arrives after the
// first chunk was written: the error goes out in-band as {"error": "..."}.
func streamThenFail(first string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Content-Type", "application/x-ndjson")
		c.Writer.Write([]byte(first + "\n"))
		json.NewEncoder(c.Writer).Encode(gin.H{"error": "model runner has unexpectedly stopped"})
	}
}

func TestStreamErrorAfterFirstChunkIsForwarded(t *testing.T) {
	gin.SetMode(gin.TestMode)

	cases := []struct {
		name    string
		path    string
		mw      gin.HandlerFunc
		body    string
		first   string
		content string
	}{
		{
			name:    "chat",
			path:    "/api/chat",
			mw:      ChatMiddleware(),
			body:    `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`,
			first:   `{"model":"test-model","message":{"role":"assistant","content":"Hel"},"done":false}`,
			content: "Hel",
		},
		{
			name:    "completions",
			path:    "/api/generate",
			mw:      CompletionsMiddleware(),
			body:    `{"model":"test-model","stream":true,"prompt":"hi"}`,
			first:   `{"model":"test-model","response":"Hel","done":false}`,
			content: "Hel",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			router := gin.New()
			router.POST(tc.path, tc.mw, streamThenFail(tc.first))

			req, _ := http.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			events := strings.Split(strings.TrimSpace(resp.Body.String()), "\n\n")
			if len(events) != 2 {
				t.Fatalf("expected the content chunk and one error event, got %d events:\n%s", len(events), resp.Body.String())
			}
			if !strings.Contains(events[0], tc.content) {
				t.Fatalf("first event lost the content: %s", events[0])
			}

			var payload struct {
				Error *struct {
					Message string `json:"message"`
					Type    string `json:"type"`
				} `json:"error"`
			}
			if err := json.Unmarshal([]byte(strings.TrimPrefix(events[1], "data: ")), &payload); err != nil {
				t.Fatalf("error event is not JSON: %v: %s", err, events[1])
			}
			if payload.Error == nil || payload.Error.Message != "model runner has unexpectedly stopped" {
				t.Fatalf("error event does not carry the error: %s", events[1])
			}
		})
	}
}
