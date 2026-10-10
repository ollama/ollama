package server

import (
	"bytes"
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
	request          llm.ScoreRequest
	err              error
	calls            int
	native           bool
	state, questions json.RawMessage
}

// SystemOne answers like llama-server does for the decision models it supports.
func (r *systemOneTestRunner) SystemOne(ctx context.Context, state, questions json.RawMessage, images []api.ImageData) (json.RawMessage, json.RawMessage, error) {
	r.calls++
	r.native = true
	r.state, r.questions = state, questions
	r.request = llm.ScoreRequest{Images: images}
	var ids map[string]json.RawMessage
	if err := json.Unmarshal(questions, &ids); err != nil {
		return nil, nil, err
	}
	answers := map[string]any{}
	for id := range ids {
		answers[id] = map[string]any{"type": "noul", "noul": 0.9}
	}
	data, _ := json.Marshal(answers)
	return data, json.RawMessage(`{"input_tokens":123,"output_tokens":0}`), r.err
}

func (r *systemOneTestRunner) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	r.calls++
	r.request = input
	if len(input.Fields) > 0 {
		return llm.ScoreResponse{Logits: [][]float32{{2, 0}}, InputTokens: 123}, r.err
	}
	return llm.ScoreResponse{Logits: [][]float32{{0, 2}}, InputTokens: 123, OutputTokens: 2}, r.err
}

func TestDecisionModelRejectsCompletion(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	// No scheduler: capability rejection must happen before a runner is loaded.
	s := &Server{}
	for _, cfg := range []struct {
		name         string
		capabilities []string
	}{
		{"decision-only", []string{"decision"}},
		{"decision-vision", []string{"decision", "vision"}},
	} {
		createSafetensorsTestModel(t, cfg.name, model.ConfigV2{
			ModelFormat: "safetensors", Capabilities: cfg.capabilities,
		}, nil)
		for _, tc := range []struct {
			name    string
			handler gin.HandlerFunc
			body    any
		}{
			{"generate", s.GenerateHandler, api.GenerateRequest{Model: cfg.name, Prompt: "hello"}},
			{"chat", s.ChatHandler, api.ChatRequest{Model: cfg.name, Messages: []api.Message{{Role: "user", Content: "hello"}}}},
		} {
			t.Run(cfg.name+"/"+tc.name, func(t *testing.T) {
				w := createRequest(t, tc.handler, tc.body)
				want := fmt.Sprintf("does not support %s", tc.name)
				if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), want) {
					t.Fatalf("status=%d body=%s, want 400 containing %q", w.Code, w.Body, want)
				}
			})
		}
	}
}

func TestSystemOneHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	config := model.ConfigV2{ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion", "decision"}}
	params, err := manifest.NewLayer(strings.NewReader(`{"num_ctx":8192}`), "application/vnd.ollama.image.params")
	if err != nil {
		t.Fatal(err)
	}
	createSafetensorsTestModel(t, "safetensors-decision", config, []manifest.Layer{params})
	config.Capabilities = []string{"decision"}
	createSafetensorsTestModel(t, "safetensors-decision-only", config, []manifest.Layer{params})
	config.Renderer = "tev1"
	createSafetensorsTestModel(t, "safetensors-tev1", config, []manifest.Layer{params})
	config.Renderer = "strands"
	createSafetensorsTestModel(t, "safetensors-strands", config, []manifest.Layer{params})
	config.Renderer = "clef"
	createSafetensorsTestModel(t, "safetensors-clef-text", config, []manifest.Layer{params})
	config.Capabilities = []string{"decision", "vision"}
	createSafetensorsTestModel(t, "safetensors-clef", config, []manifest.Layer{params})
	config.Renderer = "qwen3.5"
	config.Capabilities = []string{"completion"}
	createSafetensorsTestModel(t, "safetensors-undeclared", config, nil)
	config.Capabilities = []string{"completion", "decision"}
	system, err := manifest.NewLayer(strings.NewReader("Model-specific scoring instructions."), "application/vnd.ollama.image.system")
	if err != nil {
		t.Fatal(err)
	}
	createSafetensorsTestModel(t, "decision-test", config, []manifest.Layer{params, system})
	goTemplate, err := manifest.NewLayer(strings.NewReader("custom:{{ range .Messages }}{{ .Role }}:{{ .Content }}\n{{ end }}answer:"), "application/vnd.ollama.image.template")
	if err != nil {
		t.Fatal(err)
	}
	config.Renderer = ""
	createSafetensorsTestModel(t, "go-template", config, []manifest.Layer{params, system, goTemplate})
	for _, gguf := range []struct{ name, architecture, decisionType, renderer string }{
		{"renamed-clef", "clef", "clef", ""},
		{"laya-native", "modern-bert", "laya", ""},
		{"gguf-tev1", "qwen35", "", "tev1"},
	} {
		kv := gguftest.KV{"general.architecture": gguf.architecture}
		caps := []string{"decision"}
		if gguf.decisionType != "" {
			kv[gguf.architecture+".decision.type"] = gguf.decisionType
		}
		if gguf.decisionType == "clef" {
			caps = append(caps, "vision")
		}
		_, digest := createBinFile(t, kv, nil)
		configLayer, err := createConfigLayer(model.ConfigV2{ModelFormat: "gguf", ModelFamily: gguf.architecture, Renderer: gguf.renderer, Capabilities: caps})
		if err != nil {
			t.Fatal(err)
		}
		if err := manifest.WriteManifest(model.ParseName(gguf.name), *configLayer, []manifest.Layer{{MediaType: "application/vnd.ollama.image.model", Digest: digest}}); err != nil {
			t.Fatal(err)
		}
	}

	const prefix = `{"model":"decision-test","state":"`
	const suffix = `","questions":{"refund":{"type":"noul","instructions":"Refund requested?","criteria":{}}}}`
	stateLimit := (64 << 10) - len(prefix) - len(suffix)
	image := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte("image"), 16<<10))
	const imagePrefix = `{"model":"safetensors-clef","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}},"images":["`
	largeText := strings.Replace(prefix+strings.Repeat("x", stateLimit+1)+suffix, "decision-test", "safetensors-clef", 1)
	nativeQuestions := func(n int) string {
		questions := []string{`"refund":{"type":"noul","instructions":"q"}`}
		for i := 1; i < n; i++ {
			questions = append(questions, fmt.Sprintf(`"q%d":{"type":"noul","instructions":"q"}`, i))
		}
		return `{"model":"laya-native","state":"x","questions":{` + strings.Join(questions, ",") + `}}`
	}
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
		{"Clef null state and default instructions", `{"model":"safetensors-clef","state":null,"questions":{"refund":{"type":"noul"}}}`, nil, 200, 1, false},
		{"Clef GGUF null state and default instructions", `{"model":"renamed-clef","state":null,"questions":{"refund":{"type":"noul"},"tone":{"type":"noul","instructions":""},"urgent":{"type":"noul","instructions":"Is it urgent?"}}}`, nil, 200, 1, false},
		{"Clef joint head", `{"model":"safetensors-clef","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"llama.cpp decision model", `{"model":"laya-native","state":{"total":1250.0},"questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"Modelfile template", `{"model":"go-template","state":"refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}}`, nil, 200, 1, false},
		{"runner validation", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, api.StatusError{StatusCode: 400, ErrorMessage: "prompt too long"}, 400, 1, false},
		{"runner failure", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("runner failed"), 500, 1, false},
		{"runtime OOM", `{"model":"decision-test","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, errors.New("out of memory"), 500, 1, true},
		{"invalid schema", `{"model":"decision-test","state":"x","questions":{}}`, nil, 400, 0, false},
		{"invalid Clef schema", `{"model":"safetensors-clef","state":"x","questions":{}}`, nil, 400, 0, false},
		{"GGUF without a llama.cpp decision type", `{"model":"gguf-tev1","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"llama.cpp 65 questions", nativeQuestions(65), nil, 400, 0, false},
		{"omitted model", `{"state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"blank model", `{"model":" ","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"missing model", `{"model":"missing","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 404, 0, false},
		{"cloud", `{"model":"decision:cloud","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"safetensors success", `{"model":"safetensors-decision","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 200, 1, false},
		{"safetensors decision only", `{"model":"safetensors-decision-only","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 200, 1, false},
		{"Strands pointer head", `{"model":"safetensors-strands","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 200, 1, false},
		{"Clef decision only text", `{"model":"safetensors-clef-text","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 200, 1, false},
		{"Clef decision only image", strings.Replace(imagePrefix, "safetensors-clef", "safetensors-clef-text", 1) + `aW1hZ2U="]}`, nil, 400, 0, false},
		{"Clef image above text limit", imagePrefix + image + `"]}`, nil, 200, 1, false},
		{"other model image", `{"model":"decision-test","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}},"images":["aW1hZ2U="]}`, nil, 400, 0, false},
		{"invalid image base64", imagePrefix + `!not-base64"]}`, nil, 400, 0, false},
		{"image data URL", imagePrefix + `data:image/png;base64,aW1hZ2U="]}`, nil, 400, 0, false},
		{"Tev1 safetensors", `{"model":"safetensors-tev1","state":"x","questions":{"refund":{"type":"noul","instructions":"q"}}}`, nil, 200, 1, false},
		{"Tev1 too many candidates", `{"model":"safetensors-tev1","state":"x","questions":{"refund":{"type":"score","instructions":"q","criteria":["x"` + strings.Repeat(`,"x"`, 24) + `]}}}`, nil, 400, 0, false},
		{"Tev1 empty description", `{"model":"safetensors-tev1","state":"x","questions":{"refund":{"type":"choice","instructions":"q","criteria":{"a":"","b":"B"}}}}`, nil, 400, 0, false},
		{"safetensors missing capability", `{"model":"safetensors-undeclared","state":"x","questions":{"x":{"type":"noul","instructions":"q"}}}`, nil, 400, 0, false},
		{"bad JSON", `{`, nil, 400, 0, false},
		{"at text limit", prefix + strings.Repeat("<", stateLimit) + suffix, nil, 200, 1, false},
		{"over text limit", prefix + strings.Repeat("x", stateLimit+1) + suffix, nil, 413, 0, false},
		{"image does not relax text limit", strings.TrimSuffix(largeText, "}") + `,"images":["aW1hZ2U="]}`, nil, 413, 0, false},
		{"over transport limit", imagePrefix + strings.Repeat("A", 32<<20) + `"]}`, nil, 413, 0, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			runner := &systemOneTestRunner{err: tt.err}
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
			if tt.name == "Clef GGUF null state and default instructions" && (string(runner.state) != `"null"` ||
				string(runner.questions) != `{"refund":{"type":"noul","instructions":"refund","criteria":null},"tone":{"type":"noul","instructions":"tone","criteria":null},"urgent":{"type":"noul","instructions":"Is it urgent?","criteria":null}}`) {
				t.Fatalf("Clef defaults were not filled in: state %s, questions %s", runner.state, runner.questions)
			}
			if tt.name == "GGUF without a llama.cpp decision type" && !strings.Contains(w.Body.String(), "not a llama.cpp decision model") {
				t.Fatalf("expected a llama.cpp decision model error: %s", w.Body)
			}
			if tt.name == "Clef decision only image" && !strings.Contains(w.Body.String(), "vision") {
				t.Fatalf("expected missing vision capability: %s", w.Body)
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
				var sent decision.Request
				if err := json.Unmarshal([]byte(tt.body), &sent); err != nil {
					t.Fatal(err)
				}
				if len(runner.request.Images) != len(sent.Images) {
					t.Fatalf("runner received %d images, want %d", len(runner.request.Images), len(sent.Images))
				}
				for i := range sent.Images {
					if !bytes.Equal(runner.request.Images[i], sent.Images[i]) {
						t.Fatalf("image %d changed during request compilation", i)
					}
				}
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
				if len(runner.request.Fields) > 0 || runner.native {
					outputTokens = 0
				}
				if response.Answers["refund"].Noul < 0.88 || response.Usage.InputTokens != 123 || response.Usage.OutputTokens != outputTokens {
					t.Fatalf("incorrect scoring response: %s", w.Body)
				}
				if runner.native {
					// llama-server builds the prompt; the request reaches it as sent.
					if tt.name == "llama.cpp decision model" && (string(runner.state) != `{"total":1250.0}` || !strings.Contains(w.Body.String(), `"model":"laya-native"`)) {
						t.Fatalf("request did not pass through: state %s, response %s", runner.state, w.Body)
					}
					return
				}
				if ref.model.Config.Renderer == "strands" && (len(runner.request.PointerRows) != 1 || len(runner.request.Rows) != 0 || !strings.HasSuffix(runner.request.PointerRows[0].Prompt, "<answer>")) {
					t.Fatalf("Strands did not receive pointer inputs: %+v", runner.request)
				}
				var prompt string
				if len(runner.request.Rows) > 0 {
					prompt = runner.request.Rows[0].Prompt
				}
				if ref.model.Config.Renderer == "tev1" && (!strings.Contains(prompt, `"state": "x", "question": "q", "options":`) || strings.Contains(prompt, "Requested field:")) {
					t.Fatalf("Tev1 did not receive its per-question prompt: %q", prompt)
				}
				if len(runner.request.Fields) > 0 {
					if len(runner.request.Rows) != 0 || len(runner.request.Fields) != 1 || len(runner.request.Segments) < 3 {
						t.Fatal("Clef scoring must preserve the segmented schema")
					}
				} else if len(runner.request.PointerRows) > 0 {
					if len(runner.request.Rows) != 0 {
						t.Fatal("pointer head must bypass chat templates")
					}
				} else if ref.model.HasGoTemplate {
					if !strings.HasPrefix(prompt, "custom:system:Model-specific scoring instructions.\nuser:") || !strings.HasSuffix(prompt, "answer:") {
						t.Fatalf("scorer did not receive the Modelfile template output: %q", prompt)
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
				if runner.request.MaxTokens != 8192 {
					t.Fatalf("scoring context = %d, want model num_ctx 8192", runner.request.MaxTokens)
				}
			}
		})
	}
}
