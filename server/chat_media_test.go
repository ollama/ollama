package server

import (
	"bytes"
	"context"
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

// A RIFF/WAVE header is all it takes for llm.DetectMediaKind to see a clip;
// none of this reaches a model.
var clipBytes = []byte("RIFF\x24\x00\x00\x00WAVEfmt ")

func mediaTestModel(t *testing.T, audio bool) *Model {
	t.Helper()
	textModelPath, _ := createBinFile(t, map[string]any{
		"general.architecture": "llama",
	}, nil)
	projectorKV := map[string]any{
		"general.architecture":    "clip",
		"clip.vision.block_count": uint32(1),
	}
	if audio {
		projectorKV["clip.has_audio_encoder"] = true
	}
	projectorPath, _ := createBinFile(t, projectorKV, nil)

	m := &Model{
		Name:           "test",
		ModelPath:      textModelPath,
		ProjectorPaths: []string{projectorPath},
	}
	loadTestMetadata(t, m)
	return m
}

func TestMessageMediaCapabilitiesRejectsAudioOnTextModel(t *testing.T) {
	m := mediaTestModel(t, false)
	_, _, _, err := (&Server{}).scheduleRunnerForModel(context.Background(), m, messageMediaCapabilities([]api.Message{{
		Role:    "user",
		Content: "Transcribe this.",
		Images:  []api.ImageData{clipBytes},
	}}), nil, nil, nil)
	if err == nil || !bytes.Contains([]byte(err.Error()), []byte("audio")) {
		t.Fatalf("audio sent to a model with %v: err = %v, want an audio capability error", m.Capabilities(), err)
	}
}

func TestMessageMediaCapabilitiesKeepsAudioForAudioModel(t *testing.T) {
	m := mediaTestModel(t, true)
	if err := m.CheckCapabilities(model.CapabilityAudio); err != nil {
		t.Fatalf("fixture has no audio capability: %v", err)
	}
	if err := m.CheckCapabilities(messageMediaCapabilities([]api.Message{{
		Role:    "user",
		Content: "Transcribe this.",
		Images:  []api.ImageData{clipBytes},
	}})...); err != nil {
		t.Fatalf("audio sent to an audio model: %v", err)
	}
}

func TestMessageMediaCapabilitiesKeepsImages(t *testing.T) {
	m := mediaTestModel(t, true)
	if err := m.CheckCapabilities(model.CapabilityVision); err != nil {
		t.Fatalf("fixture has no vision capability: %v", err)
	}
	// One message holding both kinds is fine as long as the model has the
	// encoders for them.
	if err := m.CheckCapabilities(messageMediaCapabilities([]api.Message{{
		Role:    "user",
		Content: "Look and listen.",
		Images:  []api.ImageData{[]byte("\x89PNG\r\n\x1a\n"), clipBytes},
	}})...); err != nil {
		t.Fatalf("image and audio on a model that takes both: %v", err)
	}
}

// A vision projector with an audio encoder is the case where the model can
// take a clip; the text-only fixture above is what /v1/audio and /api/chat
// were letting through. /api/generate built its media list straight from
// req.Images without asking for the encoder, so a clip sat in the same
// field an image would and got dropped after the model was already loaded.
func TestGenerateRejectsAudioOnTextModelBeforeLoading(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	loads := 0
	mock := mockRunner{}
	s := &Server{
		modelCaches: &modelCaches{show: newModelShowCache()},
		sched: &Scheduler{
			pendingReqCh:    make(chan *LlmRequest, 1),
			finishedReqCh:   make(chan *LlmRequest, 1),
			expiredCh:       make(chan *runnerRef, 1),
			unloadedCh:      make(chan any, 1),
			loaded:          make(map[string]*runnerRef),
			newServerFn:     newMockServer(&mock),
			getGpuFn:        getGpuFn,
			getSystemInfoFn: getSystemInfoFn,
			loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				loads++
				req.successCh <- &runnerRef{llama: &mock}
				return false
			},
		},
	}
	createMinimalGGUFModel(t, s, "text-only", gguftest.KV{
		"general.architecture":          "llama",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(8192),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, "{{ .Prompt }}", map[string]any{"capabilities": []any{"completion"}})

	go s.sched.Run(t.Context())

	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model:  "text-only",
		Prompt: "Transcribe this.",
		Images: []api.ImageData{clipBytes},
		Stream: &stream,
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("clip on a text-only model: status = %d, want 400: %s", w.Code, w.Body.String())
	}
	if !bytes.Contains(w.Body.Bytes(), []byte("audio")) {
		t.Fatalf("body = %s, want the audio capability error", w.Body.String())
	}
	if loads != 0 {
		t.Fatalf("runner loaded %d times, want 0: the request should be rejected before scheduling", loads)
	}
}

func TestGenerateKeepsImagesForVisionModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	mock := mockRunner{}
	s := &Server{
		modelCaches: &modelCaches{show: newModelShowCache()},
		sched: &Scheduler{
			pendingReqCh:    make(chan *LlmRequest, 1),
			finishedReqCh:   make(chan *LlmRequest, 1),
			expiredCh:       make(chan *runnerRef, 1),
			unloadedCh:      make(chan any, 1),
			loaded:          make(map[string]*runnerRef),
			newServerFn:     newMockServer(&mock),
			getGpuFn:        getGpuFn,
			getSystemInfoFn: getSystemInfoFn,
			loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
				req.successCh <- &runnerRef{llama: &mock}
				return false
			},
		},
	}
	go s.sched.Run(t.Context())

	createMinimalGGUFModel(t, s, "vlm", gguftest.KV{
		"general.architecture":          "llama",
		"llama.block_count":             uint32(1),
		"llama.context_length":          uint32(8192),
		"llama.embedding_length":        uint32(4096),
		"llama.attention.head_count":    uint32(32),
		"llama.attention.head_count_kv": uint32(8),
		"tokenizer.ggml.tokens":         []string{""},
		"tokenizer.ggml.scores":         []float32{0},
		"tokenizer.ggml.token_type":     []int32{0},
	}, "{{ .Prompt }}", map[string]any{"capabilities": []any{"completion", "vision"}})

	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model:  "vlm",
		Prompt: "What is this?",
		Images: []api.ImageData{[]byte("\x89PNG\r\n\x1a\n")},
		Stream: &stream,
	})
	if w.Code == http.StatusBadRequest && bytes.Contains(w.Body.Bytes(), []byte("vision")) {
		t.Fatalf("image on a vision model was rejected: %s", w.Body.String())
	}
}
