package server

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestGenerateHandlerRejectsTrailingJSONGarbage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &Server{}
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = &http.Request{
		Method: http.MethodPost,
		Body:   io.NopCloser(strings.NewReader(`{"model":"m","prompt":"hi","stream":false} NOT JSON HERE`)),
	}

	s.GenerateHandler(c)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "error") {
		t.Fatalf("expected error response, got %s", w.Body.String())
	}
}

func TestGenerateHandlerAcceptsTrailingWhitespace(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &Server{}
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = &http.Request{
		Method: http.MethodPost,
		Body:   io.NopCloser(strings.NewReader("{\"model\":\"\",\"prompt\":\"hi\"}\n  \n")),
	}

	s.GenerateHandler(c)

	if w.Code == http.StatusBadRequest && strings.Contains(w.Body.String(), "after top-level value") {
		t.Fatalf("trailing whitespace should be allowed, got %d: %s", w.Code, w.Body.String())
	}
}
