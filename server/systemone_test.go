package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/systemone"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

type systemOneTestRunner struct {
	mockRunner
	request llm.ScoreRequest
	err     error
	calls   int
}

func (r *systemOneTestRunner) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	r.calls++
	r.request = input
	return llm.ScoreResponse{Logits: [][]float32{{0, 2}}, InputTokens: 123}, r.err
}

func TestSystemOneHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	config := model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion"}}
	createSafetensorsTestModel(t, "decision-test", config, nil)
	config.Renderer = "gemma4"
	createSafetensorsTestModel(t, "wrong-renderer", config, nil)

	for _, tt := range []struct {
		name   string
		body   string
		err    error
		status int
		calls  int
		expire bool
	}{
		{"success", `{"model":"decision-test","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"runner validation", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, api.StatusError{StatusCode: 400, ErrorMessage: "prompt too long"}, 400, 1, false},
		{"runner failure", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("runner failed"), 500, 1, false},
		{"runtime OOM", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("MLX: failed to allocate memory"), 500, 1, true},
		{"invalid schema", `{"model":"decision-test","state":"x","questions":{}}`, nil, 400, 0, false},
		{"missing model", `{"model":"missing","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 404, 0, false},
		{"cloud", `{"model":"decision:cloud","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"wrong architecture", `{"model":"wrong-renderer","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"bad JSON", `{`, nil, 400, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			runner := &systemOneTestRunner{err: tt.err}
			ref := &runnerRef{llama: runner, refCount: 1, sessionDuration: time.Hour}
			s := newServerWithMockRunner(t, &runner.mockRunner)
			s.sched.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				ref.model = req.model
				ref.modelKey = schedulerModelKey(req.model)
				s.sched.loadedMu.Lock()
				s.sched.loaded[ref.modelKey] = ref
				s.sched.loadedMu.Unlock()
				req.successCh <- ref
				return false
			}
			router := gin.New()
			router.POST("/v1/systemone", s.SystemOneHandler)
			w := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(w, req)
			if w.Code != tt.status || runner.calls != tt.calls {
				t.Fatalf("status=%d calls=%d body=%s", w.Code, runner.calls, w.Body)
			}
			if tt.calls > 0 {
				want := time.Hour
				if tt.expire {
					want = 0
				}
				ref.refMu.Lock()
				duration := ref.sessionDuration
				ref.refMu.Unlock()
				if duration != want {
					t.Fatalf("runner keep-alive=%v, want %v", duration, want)
				}
			}
			if tt.status == 200 {
				var response struct {
					Answers map[string]struct {
						Noul float64 `json:"noul"`
					} `json:"answers"`
					Usage systemone.Usage `json:"usage"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				if response.Answers["refund"].Noul < 0.88 || response.Usage.InputTokens != 123 || response.Usage.OutputTokens != 0 {
					t.Fatalf("incorrect scoring response: %s", w.Body)
				}
				if runner.request.MaxTokens != systemone.MaxPromptTokens || !strings.HasSuffix(runner.request.Rows[0].Prompt, "<think>\n\n</think>\n\n") {
					t.Fatal("scorer did not receive the trained prompt contract")
				}
			}
		})
	}
}
