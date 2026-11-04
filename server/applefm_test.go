package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/go-cmp/cmp"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/applefm"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func stubAppleFMAvailable(t *testing.T, err error) {
	t.Helper()
	old := appleFMAvailable
	appleFMAvailable = func(context.Context) error { return err }
	t.Cleanup(func() { appleFMAvailable = old })
}

// newAppleFMServer registers the model in a fresh model store and serves it
// with mock as the runner.
func newAppleFMServer(t *testing.T, mock *mockRunner) *Server {
	t.Helper()
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	stubAppleFMAvailable(t, nil)
	if err := registerAppleFoundationModel(t.Context()); err != nil {
		t.Fatal(err)
	}
	return newServerWithMockRunner(t, mock)
}

func TestRegisterAppleFoundationModel(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	stubAppleFMAvailable(t, nil)

	// A registration from an older release is brought up to date.
	writeShowManifestVariant(t, appleFMModelName, applefm.RunnerName, applefm.ModelFormat, model.ConfigV2{
		ModelFormat:  applefm.ModelFormat,
		ModelFamily:  applefm.ModelFormat,
		Capabilities: []string{"completion"},
		ContextLen:   1024,
	}, nil)

	if err := registerAppleFoundationModel(t.Context()); err != nil {
		t.Fatal(err)
	}

	resp, err := GetModelInfo(api.ShowRequest{Model: appleFMModelName})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Details.Format != applefm.ModelFormat {
		t.Errorf("format = %q", resp.Details.Format)
	}
	if !slices.Equal(resp.Capabilities, []model.Capability{model.CapabilityCompletion, model.CapabilityVision}) {
		t.Errorf("capabilities = %v", resp.Capabilities)
	}
	if got := resp.ModelInfo["applefm.context_length"]; got != applefm.ContextLength {
		t.Errorf("context length = %v", got)
	}
}

func TestRegisterAppleFoundationModelUnavailable(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName(appleFMModelName)

	stubAppleFMAvailable(t, nil)
	if err := registerAppleFoundationModel(t.Context()); err != nil {
		t.Fatal(err)
	}

	// Turning off Apple Intelligence removes the earlier registration.
	stubAppleFMAvailable(t, errors.New("Apple Intelligence is not enabled"))
	if err := registerAppleFoundationModel(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := manifest.ParseNamedManifest(name); err == nil {
		t.Fatal("model still registered while unavailable")
	}
}

func TestRegisterAppleFoundationModelKeepsOtherModel(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	writeShowManifestVariant(t, appleFMModelName, "", manifest.FormatGGUF, model.ConfigV2{ModelFormat: "gguf"}, nil)
	name := model.ParseName(appleFMModelName)
	before, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}

	for _, available := range []error{nil, errors.New("unavailable")} {
		stubAppleFMAvailable(t, available)
		if err := registerAppleFoundationModel(t.Context()); err != nil {
			t.Fatal(err)
		}
		after, err := manifest.ParseNamedManifest(name)
		if err != nil {
			t.Fatalf("available=%v: %v", available, err)
		}
		if after.Config.Digest != before.Config.Digest {
			t.Fatalf("available=%v: existing model was replaced", available)
		}
	}
}

func TestChatHandlerAppleFMSendsAllMessages(t *testing.T) {
	mock := mockRunner{
		TemplateFn: func(context.Context, llm.ChatRequest) (string, error) {
			t.Error("chat template rendered for truncation")
			return "", nil
		},
		ChatFn: func(_ context.Context, _ llm.ChatRequest, fn func(llm.ChatResponse)) error {
			fn(llm.ChatResponse{Message: api.Message{Role: "assistant", Content: "ok"}, Done: true})
			return nil
		},
	}
	s := newAppleFMServer(t, &mock)

	msgs := []api.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "old one two three four"},
		{Role: "assistant", Content: "old answer one two"},
		{Role: "user", Content: "new ask"},
	}
	stream := false
	w := createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    appleFMModelName,
		Messages: msgs,
		Stream:   &stream,
		// fm manages the context window, so Ollama must not truncate.
		Options: map[string]any{"num_ctx": 6},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	if !slices.EqualFunc(mock.ChatRequest.Messages, msgs, func(a, b api.Message) bool {
		return a.Role == b.Role && a.Content == b.Content
	}) {
		t.Errorf("messages = %+v", mock.ChatRequest.Messages)
	}
}

func TestGenerateHandlerAppleFMSendsMessages(t *testing.T) {
	mock := mockRunner{
		CompletionFn: func(context.Context, llm.CompletionRequest, func(llm.CompletionResponse)) error {
			t.Error("rendered prompt sent to completion")
			return nil
		},
		ChatFn: func(_ context.Context, _ llm.ChatRequest, fn func(llm.ChatResponse)) error {
			fn(llm.ChatResponse{Message: api.Message{Role: "assistant", Content: "BANANA"}, Done: true})
			return nil
		},
	}
	s := newAppleFMServer(t, &mock)

	stream := false
	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model:  appleFMModelName,
		System: "only say BANANA",
		Prompt: "describe this",
		Images: []api.ImageData{[]byte("\x89PNG\r\n\x1a\n")},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}

	want := []api.Message{
		{Role: "system", Content: "only say BANANA"},
		{Role: "user", Content: "describe this", Images: []api.ImageData{[]byte("\x89PNG\r\n\x1a\n")}},
	}
	if diff := cmp.Diff(want, mock.ChatRequest.Messages); diff != "" {
		t.Errorf("messages mismatch (-want +got):\n%s", diff)
	}

	var resp api.GenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Response != "BANANA" || !resp.Done {
		t.Errorf("response = %+v", resp)
	}
}

func TestGenerateHandlerAppleFMRejectsPromptFields(t *testing.T) {
	s := newAppleFMServer(t, &mockRunner{})

	stream := false
	for name, req := range map[string]api.GenerateRequest{
		"context":  {Model: appleFMModelName, Prompt: "hi", Context: []int{1, 2}, Stream: &stream},
		"template": {Model: appleFMModelName, Prompt: "hi", Template: "{{ .Prompt }}", Stream: &stream},
	} {
		t.Run(name, func(t *testing.T) {
			w := createRequest(t, s.GenerateHandler, req)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("status = %d: %s", w.Code, w.Body.String())
			}
		})
	}
}
