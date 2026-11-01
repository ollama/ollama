package server

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/renderers"
)

// speakingRunner records the speech request a handler sends it.
type speakingRunner struct {
	mockRunner
	req llm.SpeechRequest
}

func (r *speakingRunner) Speech(_ context.Context, req llm.SpeechRequest, emit func([]byte) error) error {
	r.req = req
	return emit([]byte{0, 0})
}

// The handler gives the runner the model's SpeechPrompter, which the
// llama-server runner needs to speak; renderers without one pass none.
func TestSpeechHandlerPassesSpeechPrompter(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, tt := range []struct {
		renderer string
		voices   []string
	}{
		{"lfm3-audio", renderers.LFM3SpeechVoices()},
		{"lfm3", nil},
	} {
		t.Run(tt.renderer, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			runner := &speakingRunner{}
			s := &Server{sched: &Scheduler{
				pendingReqCh:    make(chan *LlmRequest, 1),
				finishedReqCh:   make(chan *LlmRequest, 1),
				expiredCh:       make(chan *runnerRef, 1),
				unloadedCh:      make(chan any, 1),
				loaded:          make(map[string]*runnerRef),
				newServerFn:     newMockServer(&runner.mockRunner),
				getGpuFn:        getGpuFn,
				getSystemInfoFn: getSystemInfoFn,
				waitForRecovery: 250 * time.Millisecond,
				loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
					req.successCh <- &runnerRef{llama: runner}
					return false
				},
			}}
			go s.sched.Run(t.Context())

			// A loadable GGUF; the renderer, not the architecture, selects speech.
			_, digest := createBinFile(t, gguftest.KV{
				"general.architecture":          "qwen3",
				"qwen3.block_count":             uint32(1),
				"qwen3.context_length":          uint32(8192),
				"qwen3.embedding_length":        uint32(4096),
				"qwen3.attention.head_count":    uint32(32),
				"qwen3.attention.head_count_kv": uint32(8),
				"tokenizer.ggml.tokens":         []string{""},
				"tokenizer.ggml.scores":         []float32{0},
				"tokenizer.ggml.token_type":     []int32{0},
			}, []*gguftest.Tensor{
				{Name: "token_embd.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
				{Name: "output.weight", Shape: []uint64{1}, WriterTo: bytes.NewReader(make([]byte, 4))},
			})
			if w := createRequest(t, s.CreateHandler, api.CreateRequest{
				Model: "speaker", Files: map[string]string{"file.gguf": digest},
				Renderer: tt.renderer, Capabilities: []string{"completion", "speech"}, Stream: &stream,
			}); w.Code != http.StatusOK {
				t.Fatalf("create: %d %s", w.Code, w.Body)
			}

			// Speech sets write deadlines, which a ResponseRecorder can't.
			router := gin.New()
			router.POST("/v1/audio/speech", s.SpeechHandler)
			srv := httptest.NewServer(router)
			defer srv.Close()
			body, _ := json.Marshal(map[string]string{"model": "speaker", "input": "Hello there.", "voice": "male-french"})
			resp, err := http.Post(srv.URL+"/v1/audio/speech", "application/json", bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if _, err := io.ReadAll(resp.Body); err != nil || resp.StatusCode != http.StatusOK {
				t.Fatalf("status %d, read error %v", resp.StatusCode, err)
			}
			if runner.req.Input != "Hello there." || runner.req.Voice != "male-french" {
				t.Errorf("request %+v, want the client's input and voice", runner.req)
			}
			var voices []string
			if runner.req.Prompter != nil {
				voices = runner.req.Prompter.SpeechVoices()
			}
			if !slices.Equal(voices, tt.voices) {
				t.Errorf("prompter voices %v, want %v", voices, tt.voices)
			}
		})
	}
}
