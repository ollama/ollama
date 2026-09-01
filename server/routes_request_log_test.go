package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestInferenceRequestLoggerMiddlewareWritesReplayArtifacts(t *testing.T) {
	gin.SetMode(gin.TestMode)

	logDir := t.TempDir()
	requestLogger := &inferenceRequestLogger{dir: logDir}

	const route = "/v1/chat/completions"
	const requestBody = `{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`
	const responseBody = `{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"hi"}}]}`

	var bodySeenByHandler string

	r := gin.New()
	r.POST(route, requestLogger.middleware(route), func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			t.Fatalf("failed to read body in handler: %v", err)
		}

		bodySeenByHandler = string(body)
		c.Data(http.StatusOK, "application/json", []byte(responseBody))
	})

	req := httptest.NewRequest(http.MethodPost, route, strings.NewReader(requestBody))
	req.Host = "127.0.0.1:11434"
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	if bodySeenByHandler != requestBody {
		t.Fatalf("handler body mismatch:\nexpected: %s\ngot: %s", requestBody, bodySeenByHandler)
	}

	bodyFiles, err := filepath.Glob(filepath.Join(logDir, "*_v1_chat_completions_body.json"))
	if err != nil {
		t.Fatalf("failed to glob body logs: %v", err)
	}
	if len(bodyFiles) != 1 {
		t.Fatalf("expected 1 body log, got %d (%v)", len(bodyFiles), bodyFiles)
	}

	curlFiles, err := filepath.Glob(filepath.Join(logDir, "*_v1_chat_completions_request.sh"))
	if err != nil {
		t.Fatalf("failed to glob curl logs: %v", err)
	}
	if len(curlFiles) != 1 {
		t.Fatalf("expected 1 curl log, got %d (%v)", len(curlFiles), curlFiles)
	}

	bodyData, err := os.ReadFile(bodyFiles[0])
	if err != nil {
		t.Fatalf("failed to read body log: %v", err)
	}
	if string(bodyData) != requestBody {
		t.Fatalf("body log mismatch:\nexpected: %s\ngot: %s", requestBody, string(bodyData))
	}

	curlData, err := os.ReadFile(curlFiles[0])
	if err != nil {
		t.Fatalf("failed to read curl log: %v", err)
	}

	curlString := string(curlData)
	if !strings.Contains(curlString, "http://127.0.0.1:11434"+route) {
		t.Fatalf("curl log does not contain expected route URL: %s", curlString)
	}

	bodyFileName := filepath.Base(bodyFiles[0])
	if !strings.Contains(curlString, "@\"${SCRIPT_DIR}/"+bodyFileName+"\"") {
		t.Fatalf("curl log does not reference sibling body file: %s", curlString)
	}

	if w.Body.String() != responseBody {
		t.Fatalf("client response mismatch:\nexpected: %s\ngot: %s", responseBody, w.Body.String())
	}

	responseFiles, err := filepath.Glob(filepath.Join(logDir, "*_v1_chat_completions_response.json"))
	if err != nil {
		t.Fatalf("failed to glob response logs: %v", err)
	}
	if len(responseFiles) != 1 {
		t.Fatalf("expected 1 response log, got %d (%v)", len(responseFiles), responseFiles)
	}

	responseData, err := os.ReadFile(responseFiles[0])
	if err != nil {
		t.Fatalf("failed to read response log: %v", err)
	}
	if string(responseData) != responseBody {
		t.Fatalf("response log mismatch:\nexpected: %s\ngot: %s", responseBody, string(responseData))
	}
}

func TestInferenceRequestLoggerMiddlewareCapturesStreamedResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	logDir := t.TempDir()
	requestLogger := &inferenceRequestLogger{dir: logDir}

	const route = "/api/chat"
	chunks := []string{
		`{"message":{"role":"assistant","content":"he"},"done":false}` + "\n",
		`{"message":{"role":"assistant","content":"llo"},"done":false}` + "\n",
		`{"message":{"role":"assistant","content":""},"done":true}` + "\n",
	}

	r := gin.New()
	r.POST(route, requestLogger.middleware(route), func(c *gin.Context) {
		c.Header("Content-Type", "application/x-ndjson")
		c.Status(http.StatusOK)
		for _, chunk := range chunks {
			if _, err := c.Writer.Write([]byte(chunk)); err != nil {
				t.Fatalf("failed to write chunk: %v", err)
			}
			c.Writer.Flush()
		}
	})

	req := httptest.NewRequest(http.MethodPost, route, strings.NewReader(`{"model":"test-model"}`))
	req.Host = "127.0.0.1:11434"
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	want := strings.Join(chunks, "")
	if w.Body.String() != want {
		t.Fatalf("client response mismatch:\nexpected: %s\ngot: %s", want, w.Body.String())
	}

	responseFiles, err := filepath.Glob(filepath.Join(logDir, "*_api_chat_response.txt"))
	if err != nil {
		t.Fatalf("failed to glob response logs: %v", err)
	}
	if len(responseFiles) != 1 {
		t.Fatalf("expected 1 response log, got %d (%v)", len(responseFiles), responseFiles)
	}

	responseData, err := os.ReadFile(responseFiles[0])
	if err != nil {
		t.Fatalf("failed to read response log: %v", err)
	}
	if string(responseData) != want {
		t.Fatalf("response log mismatch:\nexpected: %s\ngot: %s", want, string(responseData))
	}
}

func TestInferenceRequestLoggerMiddlewareCapturesErrorResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)

	logDir := t.TempDir()
	requestLogger := &inferenceRequestLogger{dir: logDir}

	const route = "/api/generate"
	const errorBody = `{"error":"model not found"}`

	r := gin.New()
	r.POST(route, requestLogger.middleware(route), func(c *gin.Context) {
		c.Data(http.StatusNotFound, "application/json", []byte(errorBody))
	})

	req := httptest.NewRequest(http.MethodPost, route, strings.NewReader(`{"model":"missing"}`))
	req.Host = "127.0.0.1:11434"
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected status 404, got %d", w.Code)
	}

	responseFiles, err := filepath.Glob(filepath.Join(logDir, "*_api_generate_response.json"))
	if err != nil {
		t.Fatalf("failed to glob response logs: %v", err)
	}
	if len(responseFiles) != 1 {
		t.Fatalf("expected 1 response log, got %d (%v)", len(responseFiles), responseFiles)
	}

	responseData, err := os.ReadFile(responseFiles[0])
	if err != nil {
		t.Fatalf("failed to read response log: %v", err)
	}
	if string(responseData) != errorBody {
		t.Fatalf("response log mismatch:\nexpected: %s\ngot: %s", errorBody, string(responseData))
	}
}

func TestNewInferenceRequestLoggerCreatesDirectory(t *testing.T) {
	requestLogger, err := newInferenceRequestLogger()
	if err != nil {
		t.Fatalf("expected no error creating request logger: %v", err)
	}
	t.Cleanup(func() {
		_ = os.RemoveAll(requestLogger.dir)
	})

	if requestLogger == nil || requestLogger.dir == "" {
		t.Fatalf("expected request logger directory to be set")
	}

	info, err := os.Stat(requestLogger.dir)
	if err != nil {
		t.Fatalf("expected directory to exist: %v", err)
	}
	if !info.IsDir() {
		t.Fatalf("expected %q to be a directory", requestLogger.dir)
	}
}

func TestSanitizeRouteForFilename(t *testing.T) {
	tests := []struct {
		route string
		want  string
	}{
		{route: "/api/generate", want: "api_generate"},
		{route: "/v1/chat/completions", want: "v1_chat_completions"},
		{route: "/v1/messages", want: "v1_messages"},
	}

	for _, tt := range tests {
		if got := sanitizeRouteForFilename(tt.route); got != tt.want {
			t.Fatalf("sanitizeRouteForFilename(%q) = %q, want %q", tt.route, got, tt.want)
		}
	}
}
