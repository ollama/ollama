package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestThinkingCompatibilityScope(t *testing.T) {
	gin.SetMode(gin.TestMode)
	lookup := func(name string) *model.Thinking {
		if name == "generic" {
			return &model.Thinking{Values: []any{false, "medium", "xhigh"}, Default: "medium"}
		}
		return nil
	}
	for _, tt := range []struct {
		name                    string
		middleware              func(...ThinkingLookup) gin.HandlerFunc
		extra                   string
		wantGeneric, wantLegacy any
		legacyError             bool
	}{
		{"chat xhigh", ChatMiddleware, `"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"xhigh"`, "xhigh", "max", false},
		{"chat minimal", ChatMiddleware, `"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"minimal"`, "minimal", "low", false},
		{"chat future", ChatMiddleware, `"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"future"`, "future", nil, true},
		{"chat off", ChatMiddleware, `"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"`, false, false, false},
		{"chat nested precedence", ChatMiddleware, `"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"low","reasoning":{"effort":"xhigh"}`, "xhigh", "max", false},
		{"responses xhigh", ResponsesMiddleware, `"input":"hi","reasoning":{"effort":"xhigh"}`, "xhigh", "max", false},
		{"responses future", ResponsesMiddleware, `"input":"hi","reasoning":{"effort":"future"}`, "future", nil, true},
		{"responses off", ResponsesMiddleware, `"input":"hi","reasoning":{"effort":"none"}`, false, false, false},
		{"responses native precedence", ResponsesMiddleware, `"input":"hi","think":false,"reasoning":{"effort":"xhigh"}`, false, false, false},
		{"anthropic adaptive", AnthropicMessagesMiddleware, `"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"}`, "xhigh", "high", false},
		{"anthropic unknown", AnthropicMessagesMiddleware, `"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"output_config":{"effort":"future"}`, "future", nil, false},
		{"anthropic off precedence", AnthropicMessagesMiddleware, `"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"thinking":{"type":"disabled"},"output_config":{"effort":"xhigh"}`, false, false, false},
	} {
		for _, name := range []string{"generic", "legacy"} {
			t.Run(tt.name+"/"+name, func(t *testing.T) {
				var captured api.ChatRequest
				router := gin.New()
				router.POST("/", tt.middleware(lookup), func(c *gin.Context) {
					if err := json.NewDecoder(c.Request.Body).Decode(&captured); err != nil {
						t.Fatal(err)
					}
					c.Status(http.StatusOK)
				})
				req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"model":"`+name+`",`+tt.extra+`}`))
				req.Header.Set("Content-Type", "application/json")
				rec := httptest.NewRecorder()
				router.ServeHTTP(rec, req)
				if name == "legacy" && tt.legacyError {
					if rec.Code != http.StatusBadRequest {
						t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
					}
					return
				}
				if rec.Code != http.StatusOK {
					t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
				}
				want := tt.wantGeneric
				if name == "legacy" {
					want = tt.wantLegacy
				}
				var got any
				if captured.Think != nil {
					got = captured.Think.Value
				}
				if got != want {
					t.Fatalf("think=%#v, want %#v", got, want)
				}
			})
		}
	}
}
