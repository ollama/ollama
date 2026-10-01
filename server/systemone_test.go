package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
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
	if len(input.Fields) > 0 {
		return llm.ScoreResponse{Logits: [][]float32{{2, 0}}, InputTokens: 123}, r.err
	}
	return llm.ScoreResponse{Logits: [][]float32{{0, 2}}, InputTokens: 123, OutputTokens: 2}, r.err
}

func TestSystemOneHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	config := model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion", "decision"}}
	createSafetensorsTestModel(t, "safetensors-decision", config, nil)
	for _, modelConfig := range []struct {
		name, architecture, renderer, system, template string
		contextLength                                  int
		undeclared                                     bool
	}{
		{"renamed-clef", "qwen35", "", "Ignored for the joint head", "", 1024, false},
		{"decision-test", "qwen35", "qwen3.5", "Model-specific scoring instructions.", "", 1024, false},
		{"gguf-decision", "qwen35", "", "Native model scoring instructions.", "", 4096, false},
		{"go-template", "qwen35", "", "Model-specific scoring instructions.", "custom:{{ range .Messages }}{{ .Role }}:{{ .Content }}\n{{ end }}answer:", 1024, false},
		{"no-system", "qwen35", "qwen3.5", "", "", 1024, false},
		{"gguf-undeclared", "qwen35", "qwen3.5", "", "", 1024, true},
		{"gguf-other-architecture", "llama", "", "Model-specific scoring instructions.", "custom:{{ range .Messages }}{{ .Role }}:{{ .Content }}\n{{ end }}answer:", 1024, false},
	} {
		kv := gguftest.KV{"general.architecture": modelConfig.architecture}
		if modelConfig.template == "" {
			kv["tokenizer.chat_template"] = "{{ messages }}"
		}
		if modelConfig.name == "renamed-clef" {
			kv[modelConfig.architecture+".decision.type"] = "clef"
		}
		_, digest := createBinFile(t, kv, nil)
		caps := []string{"completion", "decision"}
		if modelConfig.name == "renamed-clef" {
			caps = []string{"decision", "vision"}
		}
		if modelConfig.undeclared {
			caps = []string{"completion"}
		}
		configLayer, err := createConfigLayer(model.ConfigV2{
			ModelFormat: "gguf", ModelFamily: modelConfig.architecture,
			Renderer: modelConfig.renderer, Capabilities: caps,
		})
		if err != nil {
			t.Fatal(err)
		}
		layers := []manifest.Layer{{MediaType: "application/vnd.ollama.image.model", Digest: digest}}
		for _, layer := range []struct{ content, mediaType string }{
			{modelConfig.system, "application/vnd.ollama.image.system"},
			{fmt.Sprintf(`{"num_ctx":%d}`, modelConfig.contextLength), "application/vnd.ollama.image.params"},
		} {
			l, err := manifest.NewLayer(strings.NewReader(layer.content), layer.mediaType)
			if err != nil {
				t.Fatal(err)
			}
			layers = append(layers, l)
		}
		if modelConfig.template != "" {
			l, err := manifest.NewLayer(strings.NewReader(modelConfig.template), "application/vnd.ollama.image.template")
			if err != nil {
				t.Fatal(err)
			}
			layers = append(layers, l)
		}
		if err := manifest.WriteManifest(model.ParseName(modelConfig.name), *configLayer, layers); err != nil {
			t.Fatal(err)
		}
	}

	const prefix = `{"model":"decision-test","state":"`
	const suffix = `","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`
	stateLimit := (64 << 10) - len(prefix) - len(suffix)
	for _, tt := range []struct {
		name   string
		body   string
		err    error
		status int
		calls  int
		expire bool
	}{
		{"success", `{"model":"decision-test","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"Clef image above text body limit", `{"model":"renamed-clef","state":"x","images":["` + base64.StdEncoding.EncodeToString(make([]byte, 70<<10)) + `"],"questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 200, 1, false},
		{"over image body limit", `{"model":"renamed-clef","images":["` + strings.Repeat("A", 32<<20) + `"]}`, nil, 413, 0, false},
		{"images require a supported encoding", `{"model":"decision-test","state":"x","images":["aW1hZ2U="],"questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"Clef videos unsupported", `{"model":"renamed-clef","state":"x","videos":["video.mp4"],"questions":{"refund":{"type":"noul"}}}`, nil, 400, 0, false},
		{"candidate videos unsupported", `{"model":"decision-test","state":"x","videos":["video.mp4"],"questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"Clef null state and default instructions", `{"model":"renamed-clef","state":null,"questions":{"refund":{"type":"noul"}}}`, nil, 200, 1, false},
		{"Clef joint head", `{"model":"renamed-clef","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"GGUF success without renderer", `{"model":"gguf-decision","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"Modelfile template", `{"model":"go-template","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"no system prompt", `{"model":"no-system","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"template failure", `{"model":"gguf-decision","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, errors.New("invalid model template"), 500, 0, false},
		{"GGUF capability without Qwen architecture", `{"model":"gguf-other-architecture","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 200, 1, false},
		{"GGUF missing capability", `{"model":"gguf-undeclared","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"runner validation", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, api.StatusError{StatusCode: 400, ErrorMessage: "prompt too long"}, 400, 1, false},
		{"runner failure", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("runner failed"), 500, 1, false},
		{"runtime OOM", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("out of memory"), 500, 1, true},
		{"invalid schema", `{"model":"decision-test","state":"x","questions":{}}`, nil, 400, 0, false},
		{"invalid Clef schema", `{"model":"renamed-clef","state":"x","questions":{}}`, nil, 400, 0, false},
		{"omitted model", `{"state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"blank model", `{"model":" ","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"missing model", `{"model":"missing","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 404, 0, false},
		{"cloud", `{"model":"decision:cloud","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"safetensors unsupported", `{"model":"safetensors-decision","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"bad JSON", `{`, nil, 400, 0, false},
		{"at body limit", prefix + strings.Repeat("x", stateLimit) + suffix, nil, 200, 1, false},
		{"over body limit", prefix + strings.Repeat("x", stateLimit+1) + suffix, nil, 413, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			runner := &systemOneTestRunner{err: tt.err}
			runner.TemplateFn = func(_ context.Context, req llm.ChatRequest) (string, error) {
				if len(req.Messages) != 2 || req.Messages[0].Role != "system" || req.Messages[0].Content != "Native model scoring instructions." || req.Messages[1].Role != "user" {
					t.Fatalf("model system prompt was not passed to the native template: %+v", req.Messages)
				}
				if req.Think == nil || req.Think.Bool() {
					t.Fatal("scoring must disable thinking")
				}
				return "native:" + req.Messages[1].Content, tt.err
			}
			ref := &runnerRef{llama: runner, refCount: 1, sessionDuration: time.Hour}
			s := newServerWithMockRunner(t, &runner.mockRunner)
			s.sched.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				runner.contextLength = req.opts.NumCtx
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
			req.ContentLength = -1
			req.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(w, req)
			if w.Code != tt.status || runner.calls != tt.calls {
				t.Fatalf("status=%d calls=%d body=%s", w.Code, runner.calls, w.Body)
			}
			if tt.calls == 0 && tt.err == nil && ref.model != nil {
				t.Fatal("loaded a runner for a rejected request")
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
			if tt.name == "Clef image above text body limit" && (len(runner.request.Images) != 1 || len(runner.request.Images[0]) != 70<<10) {
				t.Fatal("images were not passed to scoring")
			}
			if tt.status == 200 {
				var response struct {
					Answers map[string]struct {
						Noul float64 `json:"noul"`
					} `json:"answers"`
					Usage decision.Usage `json:"usage"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				outputTokens := 2
				if len(runner.request.Fields) > 0 {
					outputTokens = 0
				}
				if response.Answers["refund"].Noul < 0.88 || response.Usage.InputTokens != 123 || response.Usage.OutputTokens != outputTokens {
					t.Fatalf("incorrect scoring response: %s", w.Body)
				}
				if len(runner.request.Fields) > 0 {
					if len(runner.request.Rows) != 0 || len(runner.request.Fields) != 1 || runner.request.MaxTokens != 2048 {
						t.Fatal("invalid joint scoring request")
					}
					return
				}
				prompt := runner.request.Rows[0].Prompt
				wantContext := 1024
				if ref.model.HasGoTemplate {
					if !strings.HasPrefix(prompt, "custom:system:Model-specific scoring instructions.\nuser:") || !strings.HasSuffix(prompt, "answer:") {
						t.Fatalf("scorer did not receive the Modelfile template output: %q", prompt)
					}
				} else if ref.model.Config.Renderer == "" {
					wantContext = 4096
					if !strings.HasPrefix(prompt, `native:{"context":`) {
						t.Fatalf("scorer did not receive the native template output: %q", prompt)
					}
				} else {
					wantPrefix := "<|im_start|>user\n"
					if ref.model.System != "" {
						wantPrefix = "<|im_start|>system\nModel-specific scoring instructions.<|im_end|>\n<|im_start|>user\n"
					}
					if !strings.HasPrefix(prompt, wantPrefix) || !strings.HasSuffix(prompt, "<think>\n\n</think>\n\n") {
						t.Fatalf("scorer did not receive the model's rendered prompt: %q", prompt)
					}
				}
				if runner.request.MaxTokens != wantContext {
					t.Fatalf("scoring context = %d, want model num_ctx %d", runner.request.MaxTokens, wantContext)
				}
			}
		})
	}
}
