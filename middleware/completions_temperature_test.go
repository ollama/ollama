package middleware

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestCompletionsMiddleware_Temperature(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tests := []struct {
		name    string
		field   string
		wantSet bool
		want    float64
	}{
		{name: "omitted"},
		{name: "null", field: `,"temperature":null`},
		{name: "zero", field: `,"temperature":0`, wantSet: true, want: 0},
		{name: "nonzero", field: `,"temperature":0.5`, wantSet: true, want: 0.5},
	}

	for _, stream := range []bool{false, true} {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("stream=%t/%s", stream, tt.name), func(t *testing.T) {
				var captured *api.GenerateRequest
				router := gin.New()
				router.POST("/v1/completions", CompletionsMiddleware(), func(c *gin.Context) {
					var req api.GenerateRequest
					if err := json.NewDecoder(c.Request.Body).Decode(&req); err != nil {
						t.Fatal(err)
					}
					captured = &req
					c.Status(http.StatusNoContent)
				})

				body := fmt.Sprintf(`{"model":"test-model","prompt":"Hello","stream":%t%s}`, stream, tt.field)
				req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				resp := httptest.NewRecorder()
				router.ServeHTTP(resp, req)
				if resp.Code != http.StatusNoContent || captured == nil {
					t.Fatalf("request did not reach handler: status=%d body=%s", resp.Code, resp.Body)
				}
				if captured.Stream == nil || *captured.Stream != stream {
					t.Fatalf("stream option not preserved: %v", captured.Stream)
				}
				value, set := captured.Options["temperature"]
				if set != tt.wantSet || (set && value != tt.want) {
					t.Errorf("temperature option = %v (present: %v), want %v (present: %v)", value, set, tt.want, tt.wantSet)
				}
			})
		}
	}
}
