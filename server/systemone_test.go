package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/systemone"
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
	return llm.ScoreResponse{Logits: [][]float32{{0, 2}}, InputTokens: 123}, r.err
}

func TestSystemOneHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	config := model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"decision"}}
	createSafetensorsTestModel(t, "decision-test", config, nil)
	config.Capabilities = []string{"completion"}
	createSafetensorsTestModel(t, "chat-model", config, nil)
	config.Renderer = "gemma4"
	createSafetensorsTestModel(t, "gemma-model", config, nil)
	for _, modelConfig := range []struct {
		name, architecture, renderer string
	}{
		{"gguf-decision", "qwen35", ""},
		{"gguf-llama", "llama", ""},
		{"gguf-renderer", "qwen35", "gemma4"},
	} {
		_, digest := createBinFile(t, gguftest.KV{
			"general.architecture":    modelConfig.architecture,
			"tokenizer.chat_template": "{{ messages }}",
		}, nil)
		configLayer, err := createConfigLayer(model.ConfigV2{
			ModelFormat: "gguf", ModelFamily: modelConfig.architecture,
			Renderer: modelConfig.renderer, Capabilities: []string{"decision"},
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := manifest.WriteManifest(model.ParseName(modelConfig.name), *configLayer, []manifest.Layer{{
			MediaType: "application/vnd.ollama.image.model", Digest: digest,
		}}); err != nil {
			t.Fatal(err)
		}
	}

	const prefix = `{"model":"decision-test","state":"`
	const suffix = `","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`
	stateLimit := (64 << 10) - len(prefix) - len(suffix)
	// Each model's own template renders its prompts: Qwen 3.5 and Gemma 4
	// through Ollama's renderers, GGUF chat templates through the runner.
	const qwen, gemma, native = "<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|turn>model\n", "native template"
	for _, tt := range []struct {
		name   string
		body   string
		err    error
		status int
		calls  int
		expire bool
		prompt string
	}{
		{"success", `{"model":"decision-test","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false, qwen},
		{"GGUF chat template", `{"model":"gguf-decision","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false, native},
		{"GGUF other architecture", `{"model":"gguf-llama","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false, native},
		{"GGUF with renderer", `{"model":"gguf-renderer","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false, gemma},
		{"other renderer", `{"model":"gemma-model","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false, gemma},
		{"chat model", `{"model":"chat-model","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false, qwen},
		{"template failure", `{"model":"gguf-llama","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 500, 0, false, ""},
		{"runner validation", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, api.StatusError{StatusCode: 400, ErrorMessage: "prompt too long"}, 400, 1, false, ""},
		{"runner failure", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("runner failed"), 500, 1, false, ""},
		{"runtime OOM", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("MLX: failed to allocate memory"), 500, 1, true, ""},
		{"invalid schema", `{"model":"decision-test","state":"x","questions":{}}`, nil, 400, 0, false, ""},
		{"missing model", `{"model":"missing","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 404, 0, false, ""},
		{"cloud", `{"model":"decision:cloud","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false, ""},
		{"bad JSON", `{`, nil, 400, 0, false, ""},
		{"at body limit", prefix + strings.Repeat("x", stateLimit) + suffix, nil, 200, 1, false, qwen},
		{"over body limit", prefix + strings.Repeat("x", stateLimit+1) + suffix, nil, 413, 0, false, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			// The loaded context is smaller than the public prompt ceiling.
			runner := &systemOneTestRunner{mockRunner: mockRunner{contextLength: 1024, Template: native}, err: tt.err}
			if tt.name == "template failure" {
				runner.TemplateFn = func(context.Context, llm.ChatRequest) (string, error) {
					return "", errors.New("template failed")
				}
			}
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
			req.ContentLength = -1
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
				if runner.request.MaxTokens != 1024 || !strings.HasSuffix(runner.request.Rows[0].Prompt, tt.prompt) {
					t.Fatalf("scorer received prompt %q with budget %d", runner.request.Rows[0].Prompt, runner.request.MaxTokens)
				}
				if tt.prompt == native && (runner.ChatRequest.Think == nil || runner.ChatRequest.Think.Bool()) {
					t.Fatal("native templates must render with thinking off")
				}
			}
		})
	}
}

// Decision models answer through /v1/systemone only: create doesn't give them
// generate, even when their template supports tools and thinking, and they
// can still be unloaded.
func TestDecisionModelDoesNotGenerate(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	s := newServerWithMockRunner(t, &mockRunner{})
	createMinimalGGUFModel(t, s, "decider", gguftest.KV{
		"tokenizer.chat_template": `{% if tools %}<tool_call>{% endif %}<think>{{ messages[0]['content'] }}</think>`,
	}, "", map[string]any{"capabilities": []any{"decision"}})

	w := createRequest(t, s.ShowHandler, api.ShowRequest{Model: "decider"})
	var show api.ShowResponse
	if err := json.Unmarshal(w.Body.Bytes(), &show); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(show.Capabilities, []model.Capability{model.CapabilityDecision}) {
		t.Fatalf("capabilities = %v, want only decision", show.Capabilities)
	}

	w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "decider", Prompt: "hi"})
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "does not support generate") {
		t.Errorf("generate = %d %s", w.Code, w.Body)
	}
	w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: "decider", Messages: []api.Message{{Role: "user", Content: "hi"}}})
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "does not support chat") {
		t.Errorf("chat = %d %s", w.Code, w.Body)
	}
	w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "decider", KeepAlive: &api.Duration{Duration: 0}})
	if w.Code != http.StatusOK {
		t.Errorf("unload = %d %s", w.Code, w.Body)
	}
}
