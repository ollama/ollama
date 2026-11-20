package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

type tokenizerMockRunner struct {
	mockRunner
	addSpecial bool
}

func (m *tokenizerMockRunner) TokenizePrompt(_ context.Context, s string, addSpecial bool) ([]int, error) {
	m.addSpecial = addSpecial
	tokens := []int{}
	for range strings.Fields(s) {
		tokens = append(tokens, len(tokens)+10)
	}
	return tokens, nil
}

// callTokenize runs handler like net/http does: the request context ends when
// the handler returns.
func callTokenize(t *testing.T, handler gin.HandlerFunc, body any) *httptest.ResponseRecorder {
	t.Helper()
	var data []byte
	if body != nil {
		var err error
		if data, err = json.Marshal(body); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequestWithContext(ctx, http.MethodPost, "/", strings.NewReader(string(data)))
	c.Request.Header.Set("Content-Type", "application/json")
	handler(c)
	return w
}

func decodeTokenizeResponse[T any](t *testing.T, w *httptest.ResponseRecorder) T {
	t.Helper()
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var resp T
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	return resp
}

func requireTokenizeError(t *testing.T, w *httptest.ResponseRecorder, status int, contains string) {
	t.Helper()
	if w.Code != status || !strings.Contains(w.Body.String(), contains) {
		t.Fatalf("got %d %s, want %d containing %q", w.Code, w.Body.String(), status, contains)
	}
}

func requireRunnerReleased(t *testing.T, runner *runnerRef) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		runner.refMu.Lock()
		refs := runner.refCount
		runner.refMu.Unlock()
		if refs == 0 {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("runner still holds %d references", refs)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestTokenizeGGUFUsesLoadedRunnerOnly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	mock := &tokenizerMockRunner{}
	mock.DetokenizeFn = func(_ context.Context, tokens []int) (string, error) {
		return strings.Repeat("x", len(tokens)), nil
	}
	s := newServerWithMockRunner(t, &mock.mockRunner)
	createMinimalGGUFModel(t, s, "gguf-tok", nil, "{{ .Prompt }}", nil)
	m, err := s.getExistingModelForRunner("gguf-tok", "")
	if err != nil {
		t.Fatal(err)
	}
	key := schedulerModelKey(m)

	unusable := map[string]*runnerRef{
		"not loaded": nil,
		"loading":    {llama: mock, Options: &api.Options{}, loading: true},
		"unloaded":   {llama: mock},
	}
	for name, runner := range unusable {
		t.Run(name, func(t *testing.T) {
			s.sched.loadedMu.Lock()
			if runner == nil {
				delete(s.sched.loaded, key)
			} else {
				runner.model.Store(m)
				s.sched.loaded[key] = runner
			}
			s.sched.loadedMu.Unlock()

			w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "gguf-tok", Prompt: "a b"})
			requireTokenizeError(t, w, http.StatusBadRequest, "must be loaded")
			w = callTokenize(t, s.DetokenizeHandler, api.DetokenizeRequest{Model: "gguf-tok", Tokens: []int{1}})
			requireTokenizeError(t, w, http.StatusBadRequest, "must be loaded")
			if len(s.sched.pendingReqCh) != 0 {
				t.Fatal("tokenization scheduled a model load")
			}
		})
	}

	runner := &runnerRef{llama: mock, Options: &api.Options{}, sessionDuration: time.Hour}
	runner.model.Store(m)
	s.sched.loadedMu.Lock()
	s.sched.loaded[key] = runner
	s.sched.loadedMu.Unlock()

	t.Run("tokenize", func(t *testing.T) {
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "gguf-tok", Prompt: "a b c", AddSpecialTokens: true})
		resp := decodeTokenizeResponse[api.TokenizeResponse](t, w)
		if !slices.Equal(resp.Tokens, []int{10, 11, 12}) || resp.Count != 3 || resp.Model != "gguf-tok" || !mock.addSpecial {
			t.Fatalf("response = %+v, addSpecial = %t", resp, mock.addSpecial)
		}
		requireRunnerReleased(t, runner)
	})
	t.Run("empty prompt", func(t *testing.T) {
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "gguf-tok", Prompt: ""})
		if body := w.Body.String(); !strings.Contains(body, `"tokens":[]`) || !strings.Contains(body, `"count":0`) {
			t.Fatalf("empty prompt response = %s", body)
		}
		requireRunnerReleased(t, runner)
	})
	t.Run("detokenize", func(t *testing.T) {
		w := callTokenize(t, s.DetokenizeHandler, api.DetokenizeRequest{Model: "gguf-tok", Tokens: []int{1, 2}})
		if got := decodeTokenizeResponse[api.DetokenizeResponse](t, w); got.Prompt != "xx" {
			t.Fatalf("detokenize = %+v", got)
		}
		requireRunnerReleased(t, runner)
	})
}

// tokenizePeer serves Ollama's tokenize routes, or acts like a peer without
// them.
func tokenizePeer(t *testing.T, routes bool, gotModel *string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		*gotModel = body.Model
		switch {
		case !routes:
			w.Header().Set("Content-Type", "text/html")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte("<html>not found</html>"))
		case body.Model == "missing":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":"model 'missing' not found"}`))
		case body.Model == "unauthorized":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":"unauthorized"}`))
		case r.URL.Path == "/api/tokenize":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"` + body.Model + `","tokens":[7,8],"count":2}`))
		case r.URL.Path == "/api/detokenize":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"model":"` + body.Model + `","prompt":"hi"}`))
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func serveTokenize(t *testing.T, h http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(string(data)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Origin", "http://localhost")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

func TestTokenizeCloudModels(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	for _, routes := range []bool{true, false} {
		t.Run(fmt.Sprintf("routes=%t", routes), func(t *testing.T) {
			var gotModel string
			original := cloudProxyBaseURL
			cloudProxyBaseURL = tokenizePeer(t, routes, &gotModel).URL
			t.Cleanup(func() { cloudProxyBaseURL = original })

			s := &Server{}
			router, err := s.GenerateRoutes()
			if err != nil {
				t.Fatal(err)
			}

			w := serveTokenize(t, router, "/api/tokenize", api.TokenizeRequest{Model: "gpt-oss:20b:cloud", Prompt: "hi"})
			if !routes {
				requireTokenizeError(t, w, http.StatusNotImplemented, "not yet available")
				if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
					t.Fatalf("missing route content type = %q", ct)
				}
				if got := w.Header().Get("Access-Control-Allow-Origin"); got == "" {
					t.Fatal("missing route response dropped the server's CORS headers")
				}
				w = serveTokenize(t, router, "/api/detokenize", api.DetokenizeRequest{Model: "gpt-oss:20b:cloud", Tokens: []int{1}})
				requireTokenizeError(t, w, http.StatusNotImplemented, "not yet available")
				return
			}
			resp := decodeTokenizeResponse[api.TokenizeResponse](t, w)
			if gotModel != "gpt-oss:20b" || !slices.Equal(resp.Tokens, []int{7, 8}) || resp.Count != 2 {
				t.Fatalf("cloud model %q, response %+v", gotModel, resp)
			}
			w = serveTokenize(t, router, "/api/detokenize", api.DetokenizeRequest{Model: "gpt-oss:20b:cloud", Tokens: []int{7, 8}})
			if got := decodeTokenizeResponse[api.DetokenizeResponse](t, w); got.Prompt != "hi" {
				t.Fatalf("detokenize = %+v", got)
			}
		})
	}
}

func TestTokenizeRemoteModels(t *testing.T) {
	gin.SetMode(gin.TestMode)
	home := t.TempDir()
	setTestHome(t, home)
	writeTestOllamaPrivateKey(t, home)
	t.Setenv("OLLAMA_REMOTES", "127.0.0.1")

	var gotModel string
	s := Server{}
	createRemote := func(t *testing.T, name, from string, routes bool) {
		t.Helper()
		w := createRequest(t, s.CreateHandler, api.CreateRequest{
			Model:      name,
			RemoteHost: tokenizePeer(t, routes, &gotModel).URL,
			From:       from,
			Info:       map[string]any{"capabilities": []string{"completion"}},
			Stream:     &stream,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("create remote model: %d %s", w.Code, w.Body.String())
		}
	}

	t.Run("forwarded", func(t *testing.T) {
		createRemote(t, "remote-tok", "upstream-model", true)
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "remote-tok", Prompt: "hi"})
		resp := decodeTokenizeResponse[api.TokenizeResponse](t, w)
		if gotModel != "upstream-model" || resp.Model != "remote-tok" || !slices.Equal(resp.Tokens, []int{7, 8}) {
			t.Fatalf("remote model %q, response %+v", gotModel, resp)
		}
		w = callTokenize(t, s.DetokenizeHandler, api.DetokenizeRequest{Model: "remote-tok", Tokens: []int{7, 8}})
		if got := decodeTokenizeResponse[api.DetokenizeResponse](t, w); got.Prompt != "hi" || got.Model != "remote-tok" {
			t.Fatalf("detokenize = %+v", got)
		}

		// An explicit :local never leaves the machine.
		w = callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "remote-tok:latest:local", Prompt: "hi"})
		requireTokenizeError(t, w, http.StatusNotFound, "not found")
	})
	t.Run("host without routes", func(t *testing.T) {
		createRemote(t, "remote-old", "upstream-model", false)
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "remote-old", Prompt: "hi"})
		requireTokenizeError(t, w, http.StatusNotImplemented, "not yet available")
	})
	t.Run("model missing on host", func(t *testing.T) {
		createRemote(t, "remote-missing", "missing", true)
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "remote-missing", Prompt: "hi"})
		requireTokenizeError(t, w, http.StatusNotFound, "not found")
	})
	t.Run("host not in OLLAMA_REMOTES", func(t *testing.T) {
		createRemote(t, "remote-denied", "upstream-model", true)
		t.Setenv("OLLAMA_REMOTES", "example.com")
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "remote-denied", Prompt: "hi"})
		requireTokenizeError(t, w, http.StatusBadRequest, "cannot run this remote model")
	})
	t.Run("unauthorized", func(t *testing.T) {
		createRemote(t, "remote-auth", "unauthorized", true)
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "remote-auth", Prompt: "hi"})
		requireTokenizeError(t, w, http.StatusUnauthorized, "signin_url")
	})
}

func TestTokenizeRequestValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	s := Server{}

	for _, tc := range []struct {
		name     string
		handler  gin.HandlerFunc
		body     any
		status   int
		contains string
	}{
		{"missing body", s.TokenizeHandler, nil, http.StatusBadRequest, "missing request body"},
		{"missing model", s.TokenizeHandler, api.TokenizeRequest{Prompt: "hi"}, http.StatusBadRequest, "model is required"},
		{"unknown model", s.TokenizeHandler, api.TokenizeRequest{Model: "missing", Prompt: "hi"}, http.StatusNotFound, "not found"},
		{"missing tokens", s.DetokenizeHandler, api.DetokenizeRequest{Model: "missing"}, http.StatusBadRequest, "tokens are required"},
		{"too large", s.TokenizeHandler, api.TokenizeRequest{Model: "missing", Prompt: strings.Repeat("a", maxTokenizeBodyBytes)}, http.StatusRequestEntityTooLarge, "too large"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			requireTokenizeError(t, callTokenize(t, tc.handler, tc.body), tc.status, tc.contains)
		})
	}
}

const testMLXTokenizerJSON = `{
	"model": {"type": "BPE", "vocab": {"h": 0, "i": 1, "hi": 2, "<s>": 3}, "merges": ["h i"]},
	"added_tokens": [{"id": 3, "content": "<s>", "special": true}]
}`

func createMLXTokenizerModel(t *testing.T, name, arch string, addBOS bool) {
	t.Helper()
	var layers []manifest.Layer
	for file, data := range map[string]string{
		"config.json":           `{"architectures":["` + arch + `"]}`,
		"tokenizer.json":        testMLXTokenizerJSON,
		"tokenizer_config.json": fmt.Sprintf(`{"bos_token":"<s>","add_bos_token":%t}`, addBOS),
	} {
		layers = append(layers, manifest.Layer{
			MediaType: manifest.MediaTypeImageJSON,
			Digest:    createTestBlob(t, []byte(data)),
			Size:      int64(len(data)),
			Name:      file,
		})
	}
	createSafetensorsTestModel(t, name, model.ConfigV2{ModelFormat: "safetensors", Capabilities: []string{"completion"}}, layers)
}

func TestTokenizeMLXInProcess(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	createMLXTokenizerModel(t, "mlx-tok", "Qwen3ForCausalLM", true)
	createMLXTokenizerModel(t, "mlx-nobos", "Qwen3ForCausalLM", false)
	createMLXTokenizerModel(t, "mlx-unregistered", "NoSuchArchitecture", true)

	// No scheduler: MLX tokenization must not need a runner.
	s := Server{}
	for _, tc := range []struct {
		model      string
		addSpecial bool
		want       []int
	}{
		{"mlx-tok", false, []int{2, 2}},
		{"mlx-tok", true, []int{3, 2, 2}},
		{"mlx-nobos", true, []int{2, 2}},
	} {
		t.Run(fmt.Sprintf("%s add_special_tokens=%t", tc.model, tc.addSpecial), func(t *testing.T) {
			w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: tc.model, Prompt: "hihi", AddSpecialTokens: tc.addSpecial})
			resp := decodeTokenizeResponse[api.TokenizeResponse](t, w)
			if !slices.Equal(resp.Tokens, tc.want) || resp.Count != len(tc.want) {
				t.Fatalf("tokens = %v, want %v", resp.Tokens, tc.want)
			}
		})
	}

	t.Run("detokenize", func(t *testing.T) {
		w := callTokenize(t, s.DetokenizeHandler, api.DetokenizeRequest{Model: "mlx-tok", Tokens: []int{2, 2}})
		if got := decodeTokenizeResponse[api.DetokenizeResponse](t, w); got.Prompt != "hihi" {
			t.Fatalf("detokenize = %+v", got)
		}
		for _, id := range []int{-1, 4} {
			w := callTokenize(t, s.DetokenizeHandler, api.DetokenizeRequest{Model: "mlx-tok", Tokens: []int{id}})
			requireTokenizeError(t, w, http.StatusBadRequest, "outside the vocabulary")
		}
	})
	t.Run("unregistered architecture", func(t *testing.T) {
		w := callTokenize(t, s.TokenizeHandler, api.TokenizeRequest{Model: "mlx-unregistered", Prompt: "hi"})
		requireTokenizeError(t, w, http.StatusBadRequest, "does not support tokenization")
	})
}
