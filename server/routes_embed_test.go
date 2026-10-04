package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
)

type embedTestBackend struct {
	llm.LlamaServer
	fn func(context.Context, string) ([]float32, int, error)
}

func (r *embedTestBackend) Embedding(ctx context.Context, text string) ([]float32, int, error) {
	return r.fn(ctx, text)
}

func newEmbedTestServer(t *testing.T, fn func(context.Context, string) ([]float32, int, error)) *Server {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	gin.SetMode(gin.TestMode)
	runner := &mockRunner{LlamaServer: &embedTestBackend{fn: fn}}
	s := newServerWithMockRunner(t, runner)
	createMinimalGGUFModel(t, s, "test-model", nil, "", nil)
	return s
}

func embedRequest(handler gin.HandlerFunc, body string) *httptest.ResponseRecorder {
	router := gin.New()
	router.POST("/v1/embeddings", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), handler)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(body)))
	return w
}

func TestOpenAIEmbeddingsFormats(t *testing.T) {
	s := newEmbedTestServer(t, func(_ context.Context, text string) ([]float32, int, error) {
		return []float32{3, 4, 0}, 5, nil
	})
	for _, format := range []string{"", "float", "base64", "FLOAT", "BASE64", "Float", "Base64"} {
		t.Run("format="+format, func(t *testing.T) {
			w := embedRequest(s.OpenAIEmbeddingsHandler, fmt.Sprintf(`{"model":"test-model","input":["A","B","C"],"encoding_format":%q}`, format))
			if w.Code != 200 {
				t.Fatalf("%d: %s", w.Code, w.Body.String())
			}
			var resp openai.EmbeddingList
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatal(err)
			}
			if resp.Model != "test-model" || resp.Object != "list" || len(resp.Data) != 3 || resp.Usage.PromptTokens != 15 || resp.Usage.TotalTokens != 15 {
				t.Fatalf("unexpected response: %+v", resp)
			}
			for i, item := range resp.Data {
				if item.Index != i || item.Object != "embedding" {
					t.Fatalf("unexpected item: %+v", item)
				}
				if strings.EqualFold(format, "base64") {
					value, ok := item.Embedding.(string)
					if !ok {
						t.Fatalf("expected base64: %T", item.Embedding)
					}
					decoded, err := base64.StdEncoding.DecodeString(value)
					if err != nil || len(decoded) != 12 {
						t.Fatalf("invalid base64: %q %v", value, err)
					}
					for j, want := range []float32{0.6, 0.8, 0} {
						if got := math.Float32frombits(binary.LittleEndian.Uint32(decoded[j*4:])); math.Abs(float64(got-want)) > 1e-6 {
							t.Fatalf("value %d: %v != %v", j, got, want)
						}
					}
				} else {
					values, ok := item.Embedding.([]any)
					if !ok || len(values) != 3 {
						t.Fatalf("expected 3 floats: %v", item.Embedding)
					}
					for j, want := range []float64{0.6, 0.8, 0} {
						if math.Abs(values[j].(float64)-want) > 1e-6 {
							t.Fatalf("unexpected float: %v", values)
						}
					}
				}
			}
		})
	}
}

func TestOpenAIEmbeddingsValidation(t *testing.T) {
	s := &Server{}
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	for _, tc := range []struct {
		name, body, message string
		status              int
	}{
		{"missing body", "", "EOF", 400},
		{"malformed", "{", "unexpected EOF", 400},
		{"missing input", `{"model":"test-model"}`, "invalid input", 400},
		{"null input", `{"model":"test-model","input":null}`, "invalid input", 400},
		{"empty list", `{"model":"test-model","input":[]}`, "invalid input", 400},
		{"mixed list", `{"model":"test-model","input":["hello",123]}`, "invalid input type", 400},
		{"number", `{"model":"test-model","input":123}`, "invalid input type", 400},
		{"object", `{"model":"test-model","input":{}}`, "invalid input type", 400},
		{"invalid format", `{"model":"test-model","input":"A","encoding_format":"hex"}`, "encoding_format", 400},
		{"invalid json format", `{"model":"test-model","input":"A","encoding_format":"json"}`, "encoding_format", 400},
		{"invalid arbitrary format", `{"model":"test-model","input":"A","encoding_format":"invalid_format"}`, "encoding_format", 400},
		{"missing model", `{"model":"does-not-exist","input":"A"}`, "not found", 404},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := embedRequest(s.OpenAIEmbeddingsHandler, tc.body)
			if w.Code != tc.status {
				t.Fatalf("%d: %s", w.Code, w.Body.String())
			}
			var resp openai.ErrorResponse
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(resp.Error.Message, tc.message) || resp.Error.Type != openai.NewError(tc.status, tc.message).Error.Type {
				t.Fatalf("unexpected error: %+v", resp)
			}
		})
	}
}

func TestEmbedNativeAndOpenAIInputSemantics(t *testing.T) {
	var mu sync.Mutex
	var inputs []string
	s := newEmbedTestServer(t, func(_ context.Context, text string) ([]float32, int, error) {
		mu.Lock()
		inputs = append(inputs, text)
		mu.Unlock()
		return []float32{3, 4}, 2, nil
	})
	for _, tc := range []struct {
		name, input string
		openAI      bool
		count       int
	}{
		{"native empty string", `""`, false, 0},
		{"native null", `null`, false, 0},
		{"native empty list", `[]`, false, 0},
		{"OpenAI empty string", `""`, true, 1},
		{"native empty string list", `[""]`, false, 1},
		{"OpenAI single", `"Hello"`, true, 1},
		{"OpenAI batch", `["Hello","World"]`, true, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inputs = nil
			handler := s.EmbedHandler
			if tc.openAI {
				handler = s.OpenAIEmbeddingsHandler
			}
			w := createRequest(t, handler, json.RawMessage(fmt.Sprintf(`{"model":"test-model","input":%s}`, tc.input)))
			if w.Code != 200 {
				t.Fatalf("%d: %s", w.Code, w.Body.String())
			}
			if len(inputs) != tc.count {
				t.Fatalf("inputs = %q, expected %d", inputs, tc.count)
			}
			if tc.count == 1 && tc.input == `""` && inputs[0] != "" {
				t.Fatalf("input changed: %q", inputs)
			}
			if !tc.openAI {
				var resp api.EmbedResponse
				if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
					t.Fatal(err)
				}
				if len(resp.Embeddings) != tc.count {
					t.Fatalf("unexpected native response: %+v", resp)
				}
				if tc.count == 0 && (resp.TotalDuration != 0 || resp.LoadDuration != 0) {
					t.Fatalf("empty native response gained timing fields: %+v", resp)
				}
			}
		})
	}
}

func TestEmbedDimensionsAndBatchOrdering(t *testing.T) {
	// A waits for B and C: completion order differs from request order.
	release := make(chan struct{})
	var mu sync.Mutex
	completed := 0
	s := newEmbedTestServer(t, func(_ context.Context, text string) ([]float32, int, error) {
		if text == "A" {
			<-release
		} else {
			mu.Lock()
			completed++
			if completed == 2 {
				close(release)
			}
			mu.Unlock()
		}
		vector := make([]float32, 256)
		vector[int(text[0]-'A')] = 1
		return vector, int(text[0]-'A') + 1, nil
	})
	w := embedRequest(s.OpenAIEmbeddingsHandler, `{"model":"test-model","input":["A","B","C"],"dimensions":128}`)
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	var resp openai.EmbeddingList
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 3 || resp.Usage.PromptTokens != 6 {
		t.Fatalf("unexpected response: %+v", resp)
	}
	for i, item := range resp.Data {
		values := item.Embedding.([]any)
		if item.Index != i || len(values) != 128 || values[i].(float64) != 1 {
			t.Fatalf("ordering/dimensions changed: %+v", item)
		}
	}
}

func TestEmbedExecutionErrors(t *testing.T) {
	for _, tc := range []struct {
		name    string
		err     error
		status  int
		message string
	}{
		{"plain", errors.New("  runner failed \n"), 400, "runner failed"},
		{"status", api.StatusError{StatusCode: 503, ErrorMessage: "  runner unavailable \n"}, 503, "runner unavailable"},
		{"empty status message", api.StatusError{StatusCode: 503, ErrorMessage: " \n"}, 503, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newEmbedTestServer(t, func(context.Context, string) ([]float32, int, error) { return nil, 0, tc.err })
			for _, compat := range []bool{false, true} {
				handler := s.EmbedHandler
				if compat {
					handler = s.OpenAIEmbeddingsHandler
				}
				w := createRequest(t, handler, api.EmbedRequest{Model: "test-model", Input: "hello"})
				if w.Code != tc.status {
					t.Fatalf("%d: %s", w.Code, w.Body.String())
				}
				if compat {
					var resp openai.ErrorResponse
					if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
						t.Fatal(err)
					}
					if resp.Error.Message != (api.StatusError{ErrorMessage: tc.message}).Error() {
						t.Fatalf("%+v", resp)
					}
				} else {
					var resp map[string]string
					if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
						t.Fatal(err)
					}
					if resp["error"] != tc.message {
						t.Fatalf("%+v", resp)
					}
				}
			}
		})
	}
}

func TestEmbedScheduleErrors(t *testing.T) {
	for _, tc := range []struct {
		err     error
		status  int
		message string
	}{
		{errCapabilities, 400, errCapabilities.Error()},
		{errRequired, 400, errRequired.Error()},
		{errTypicalPUnsupported, 400, errTypicalPUnsupported.Error()},
		{fmt.Errorf("wrapped: %w", context.Canceled), 499, "request canceled"},
		{ErrMaxQueue, 503, ErrMaxQueue.Error()},
		{fmt.Errorf("wrapped: %w", os.ErrNotExist), 404, `model "test-model" not found, try pulling it first`},
		{errors.New("failure"), 500, "failure"},
	} {
		e := scheduleEmbedError("test-model", tc.err)
		if e.status != tc.status || e.Error() != tc.message {
			t.Errorf("%v: %+v", tc.err, e)
		}
	}
}

func TestParseEmbedInput(t *testing.T) {
	for _, tc := range []struct {
		input   any
		want    []string
		invalid bool
	}{
		{nil, nil, false},
		{"", nil, false},
		{"A", []string{"A"}, false},
		{[]any{}, nil, false},
		{[]any{"", "B"}, []string{"", "B"}, false},
		{[]any{"A", 123}, nil, true},
		{123, nil, true},
		{map[string]any{}, nil, true},
	} {
		got, err := parseEmbedInput(tc.input)
		if (err != nil) != tc.invalid || !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%#v: %q, %v", tc.input, got, err)
		}
	}
}

func TestEmbedNativeTruncation(t *testing.T) {
	for _, truncate := range []bool{true, false} {
		t.Run(fmt.Sprint(truncate), func(t *testing.T) {
			var seen string
			s := newEmbedTestServer(t, func(_ context.Context, text string) ([]float32, int, error) {
				seen = text
				return []float32{3, 4}, 2, nil
			})
			w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
				Model: "test-model", Input: strings.Repeat("word ", 8200), Truncate: &truncate,
			})
			if !truncate {
				if w.Code != 400 || !strings.Contains(w.Body.String(), "the input length exceeds the context length") || seen != "" {
					t.Fatalf("overflow behavior changed: %d %s, input %q", w.Code, w.Body.String(), seen)
				}
				return
			}
			// mockRunner.Detokenize returns an empty string; successful execution
			// therefore proves that the oversized input was tokenized and truncated.
			if w.Code != 200 || seen != "" {
				t.Fatalf("truncation: %d %s", w.Code, w.Body.String())
			}
			var resp api.EmbedResponse
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatal(err)
			}
			if len(resp.Embeddings) != 1 || resp.PromptEvalCount != 2 {
				t.Fatalf("unexpected response: %+v", resp)
			}
		})
	}
}

func TestEmbedRuntimeOOMExpiresRunners(t *testing.T) {
	s := newEmbedTestServer(t, func(context.Context, string) ([]float32, int, error) {
		return nil, 0, errors.New("out of memory")
	})
	ref := &runnerRef{refCount: 1, sessionDuration: time.Hour}
	s.sched.loadedMu.Lock()
	s.sched.loaded["oom-test"] = ref
	s.sched.loadedMu.Unlock()
	w := createRequest(t, s.EmbedHandler, api.EmbedRequest{Model: "test-model", Input: "hello"})
	if w.Code != 400 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	ref.refMu.Lock()
	defer ref.refMu.Unlock()
	if ref.sessionDuration != 0 {
		t.Fatal("runtime OOM did not expire the loaded runner")
	}
}

func TestEmbedNativeRetryAndDimensions(t *testing.T) {
	var calls []string
	s := newEmbedTestServer(t, func(_ context.Context, text string) ([]float32, int, error) {
		calls = append(calls, text)
		if len(calls) == 1 {
			return nil, 0, api.StatusError{StatusCode: 400, ErrorMessage: "context overflow"}
		}
		return []float32{3, 4, 12}, 7, nil
	})
	text := strings.Repeat("word ", 100)
	w := createRequest(t, s.EmbedHandler, api.EmbedRequest{
		Model: "test-model", Input: text, Dimensions: 2,
		Options: map[string]any{"num_batch": 4},
	})
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	if len(calls) != 2 || calls[0] != text || calls[1] != "" {
		t.Fatalf("unexpected retry inputs: %q", calls)
	}
	var resp api.EmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.PromptEvalCount != 7 || len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) != 2 {
		t.Fatalf("unexpected response: %+v", resp)
	}
	for i, want := range []float32{0.6, 0.8} {
		if math.Abs(float64(resp.Embeddings[0][i]-want)) > 1e-6 {
			t.Fatalf("shortened embedding was not renormalized: %v", resp.Embeddings)
		}
	}
}

func TestOpenAIEmbeddingsModelErrorsMatchNative(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	s := &Server{}
	for _, name := range []string{"", "bad name", "test:cloud:local", "missing", "missing:local"} {
		t.Run(name, func(t *testing.T) {
			request := api.EmbedRequest{Model: name, Input: "hello"}
			native := createRequest(t, s.EmbedHandler, request)
			compat := createRequest(t, s.OpenAIEmbeddingsHandler, request)
			if native.Code != compat.Code || native.Code < 400 {
				t.Fatalf("statuses differ: native=%d compatibility=%d", native.Code, compat.Code)
			}
			var nativeError map[string]string
			if err := json.Unmarshal(native.Body.Bytes(), &nativeError); err != nil {
				t.Fatal(err)
			}
			var compatError openai.ErrorResponse
			if err := json.Unmarshal(compat.Body.Bytes(), &compatError); err != nil {
				t.Fatal(err)
			}
			if compatError.Error.Message != nativeError["error"] {
				t.Fatalf("errors differ: %s / %s", native.Body, compat.Body)
			}
		})
	}
}

func TestOpenAIEmbeddingsRegisteredRoute(t *testing.T) {
	s := newEmbedTestServer(t, func(context.Context, string) ([]float32, int, error) { return []float32{3, 4}, 1, nil })
	router, err := s.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "http://localhost/v1/embeddings", strings.NewReader(`{"model":"test-model","input":"hello","encoding_format":"base64"}`))
	router.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	var resp openai.EmbeddingList
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if _, ok := resp.Data[0].Embedding.(string); !ok {
		t.Fatalf("expected base64: %+v", resp)
	}
}

func TestOpenAIEmbeddingsCloudPassthrough(t *testing.T) {
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_NO_CLOUD", "0")
	const response = `{"object":"list","data":[],"model":"remote","extra":"preserved"}`
	captured := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected cloud path: %s", r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		captured <- string(body)
		w.Header().Set("Content-Type", "application/json")
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
	w := httptest.NewRecorder()
	router.ServeHTTP(w, httptest.NewRequest("POST", "http://localhost/v1/embeddings", strings.NewReader(`{"model":"test-model:cloud","input":"hello","encoding_format":"cloud-specific","extra":true}`)))
	if w.Code != 200 || w.Body.String() != response {
		t.Fatalf("cloud response changed: %d %s", w.Code, w.Body.String())
	}
	var request map[string]any
	if err := json.Unmarshal([]byte(<-captured), &request); err != nil {
		t.Fatal(err)
	}
	if request["model"] != "test-model" || request["encoding_format"] != "cloud-specific" || request["extra"] != true {
		t.Fatalf("cloud request changed: %+v", request)
	}
}
