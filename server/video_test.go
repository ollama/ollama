package server

import (
	"bytes"
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestGenerateWithVideo(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	stream := false
	for _, mode := range []string{"rendered", "native", "native implicit", "template", "role template", "suffix", "raw", "raw implicit", "empty prompt"} {
		t.Run(mode, func(t *testing.T) {
			mock := &mockRunner{CompletionResponse: llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop}}
			nativeTemplateCalled := false
			mock.TemplateFn = func(_ context.Context, request llm.ChatRequest) (string, error) {
				nativeTemplateCalled = true
				if len(request.Messages[0].Videos) != 0 || len(request.Messages[0].Images) != 0 {
					t.Fatal("native template duplicated binary media")
				}
				return request.Messages[0].Content, nil
			}
			s := newServerWithMockRunner(t, mock)
			kv := gguftest.KV{"llama.vision.block_count": uint32(1)}
			if strings.HasPrefix(mode, "native") {
				kv["tokenizer.chat_template"] = "{{ messages[0]['content'] }}"
			}
			name := "video-" + strings.ReplaceAll(mode, " ", "-")
			baseTemplate := "{{ .Prompt }}"
			if strings.HasPrefix(mode, "native") {
				baseTemplate = ""
			}
			if mode == "suffix" {
				baseTemplate = "{{ if .Suffix }}{{ .Prompt }}SUFFIX{{ .Suffix }}{{ else }}{{ .Prompt }}{{ end }}"
			}
			createMinimalGGUFModel(t, s, name, kv, baseTemplate, nil)
			request := api.GenerateRequest{Model: name, Prompt: "describe [img] then [img]", Images: []api.ImageData{[]byte("image")}, Videos: []api.VideoData{[]byte("video")}, Stream: &stream}
			if mode == "native implicit" || mode == "raw implicit" {
				request.Prompt = "describe"
			}
			if mode == "template" {
				request.Template = "{{ .Prompt }}"
			}
			if mode == "role template" {
				request.Template = "{{ range .Messages }}<role>{{ .Role }}:{{ .Content }}</role>{{ end }}"
				request.Prompt = "describe"
			}
			if mode == "suffix" {
				request.Suffix = "end"
			}
			if strings.HasPrefix(mode, "raw") {
				request.Raw = true
			}
			if mode == "empty prompt" {
				request.Prompt = ""
			}
			response := createRequest(t, s.GenerateHandler, request)
			if response.Code != http.StatusOK {
				t.Fatalf("status %d: %s", response.Code, response.Body.String())
			}
			if strings.HasPrefix(mode, "native") && !nativeTemplateCalled {
				t.Fatal("native fixture bypassed llama-server template")
			}
			if strings.Count(mock.CompletionRequest.Prompt, "[img-0]") != 1 || strings.Count(mock.CompletionRequest.Prompt, "[img-1]") != 1 {
				t.Fatalf("unattached media in %q", mock.CompletionRequest.Prompt)
			}
			if (mode == "native implicit" || mode == "raw implicit") && mock.CompletionRequest.Prompt != "describe[img-0][img-1]" {
				t.Fatalf("native media moved before text: %q", mock.CompletionRequest.Prompt)
			}
			if mode == "role template" && !strings.HasPrefix(mock.CompletionRequest.Prompt, "<role>user:describe[img-0][img-1]") {
				t.Fatalf("video escaped user role: %q", mock.CompletionRequest.Prompt)
			}
			media := mock.CompletionRequest.Media
			if len(media) != 2 || media[0].ID != 0 || media[1].ID != 1 || media[1].Kind != llm.MediaKindVideo || !bytes.Equal(media[1].Data, []byte("video")) {
				t.Fatalf("media changed: %#v", media)
			}
		})
	}
}

func TestVideoInputRejectsUnsupportedRunners(t *testing.T) {
	for _, config := range []model.ConfigV2{
		{ModelFormat: "safetensors", Capabilities: []string{"vision"}},
		{ModelFamilies: []string{"mllama"}, Capabilities: []string{"vision"}},
		{ModelFormat: "gguf", Capabilities: []string{"completion"}},
	} {
		m := &Model{Config: config}
		if err := validateVideoInput(m, []api.Message{{Videos: []api.VideoData{[]byte("video")}}}); err == nil {
			t.Fatalf("accepted unsupported runner: %#v", config)
		}
		if err := validateVideoInput(m, []api.Message{{Content: "text"}}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCloneMessagesIsolatesVideoBytes(t *testing.T) {
	original := []api.Message{{Videos: []api.VideoData{[]byte("video")}}}
	cloned := cloneMessages(original)
	cloned[0].Videos[0][0] = 'X'
	if string(original[0].Videos[0]) != "video" {
		t.Fatal("cached video bytes were shared")
	}
}

func TestVideoHandlersRejectBeforeLoading(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	for _, tc := range []struct {
		name, runner, format   string
		families, capabilities []string
	}{
		{"mlx", manifest.RunnerMLX, manifest.FormatSafetensors, []string{"qwen3vl"}, []string{"completion", "vision"}},
		{"mllama", manifest.RunnerGGML, manifest.FormatGGUF, []string{"mllama"}, []string{"completion", "vision"}},
		{"text", manifest.RunnerGGML, manifest.FormatGGUF, []string{"llama"}, []string{"completion"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			writeShowManifestVariant(t, tc.name, tc.runner, tc.format, model.ConfigV2{ModelFormat: tc.format, ModelFamilies: tc.families, Capabilities: tc.capabilities}, map[string]any{"general.architecture": "llama"})
			s := &Server{} // A runner load would panic: validation must precede scheduling.
			stream := false
			generate := createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: tc.name, Prompt: "describe", Videos: []api.VideoData{[]byte("video")}, Stream: &stream})
			if generate.Code != http.StatusBadRequest {
				t.Fatalf("generate status %d: %s", generate.Code, generate.Body.String())
			}
			chat := createRequest(t, s.ChatHandler, api.ChatRequest{Model: tc.name, Messages: []api.Message{{Role: "user", Content: "describe", Videos: []api.VideoData{[]byte("video")}}}, Stream: &stream})
			if chat.Code != http.StatusBadRequest {
				t.Fatalf("chat status %d: %s", chat.Code, chat.Body.String())
			}
		})
	}
}

func TestGenerateVideoTemplateKeepsHistoryRoles(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	stream := false
	mock := &mockRunner{CompletionResponse: llm.CompletionResponse{Done: true, DoneReason: llm.DoneReasonStop}}
	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "video-base", gguftest.KV{"llama.vision.block_count": uint32(1)}, "{{ .Prompt }}", nil)
	created := createRequest(t, s.CreateHandler, api.CreateRequest{Model: "video-history", From: "video-base", System: "system", Renderer: "qwen3-vl-instruct", Messages: []api.Message{
		{Role: "user", Content: "history", Videos: []api.VideoData{[]byte("history-video")}},
		{Role: "assistant", Content: "prior response"},
	}, Stream: &stream})
	if created.Code != http.StatusOK {
		t.Fatalf("create %d: %s", created.Code, created.Body.String())
	}
	response := createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: "video-history", Prompt: "current", Images: []api.ImageData{[]byte("image")}, Videos: []api.VideoData{[]byte("current-video")}, Template: "{{ .System }}{{ range .Messages }}<role>{{ .Role }}:{{ .Content }}</role>{{ end }}", Stream: &stream})
	if response.Code != http.StatusOK {
		t.Fatalf("generate %d: %s", response.Code, response.Body.String())
	}
	prompt := mock.CompletionRequest.Prompt
	for _, want := range []string{"system<role>system:system</role><role>user:history[img-0]</role>", "<role>user:current[img-1][img-2]</role>"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("missing %q in %q", want, prompt)
		}
	}
	media := mock.CompletionRequest.Media
	if len(media) != 3 || string(media[0].Data) != "history-video" || media[0].Kind != llm.MediaKindVideo || string(media[1].Data) != "image" || string(media[2].Data) != "current-video" {
		t.Fatalf("history media order changed: %#v", media)
	}
}
