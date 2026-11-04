//go:build integration && generate

package integration

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestGenerateNativeJinjaDebugRender(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("requires a test-managed server started with OLLAMA_GO_TEMPLATE=0")
	}
	t.Setenv("OLLAMA_GO_TEMPLATE", "0")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	pullOrSkip(ctx, t, client, smol)

	modelName := createGenerateJinjaModel(ctx, t, client, smol)
	defer func() {
		if err := client.Delete(context.Background(), &api.DeleteRequest{Model: modelName}); err != nil {
			t.Logf("failed to delete %s: %v", modelName, err)
		}
	}()

	t.Run("generate matches chat", func(t *testing.T) {
		prompt := "Reply with one short sentence about blue skies."
		generateRender := debugGenerateRender(ctx, t, client, api.GenerateRequest{
			Model:  modelName,
			Prompt: prompt,
		})
		chatRender := debugChatRender(ctx, t, client, api.ChatRequest{
			Model: modelName,
			Messages: []api.Message{
				{Role: "user", Content: prompt},
			},
		})

		if generateRender == "" {
			t.Fatal("generate debug render was empty")
		}
		if strings.Contains(generateRender, "GO_TEMPLATE_SENTINEL") {
			t.Fatalf("generate used Go TEMPLATE instead of native chat template: %q", generateRender)
		}
		if generateRender != chatRender {
			t.Fatalf("generate render does not match chat render\ngenerate: %q\nchat:     %q", generateRender, chatRender)
		}
	})

	t.Run("context prepends prior tokens", func(t *testing.T) {
		first := generateOnce(ctx, t, client, api.GenerateRequest{
			Model:  modelName,
			Prompt: "Say hello.",
			Options: map[string]any{
				"num_predict": 1,
				"temperature": 0,
			},
		})
		if len(first.Context) == 0 {
			t.Fatal("generate response did not include context")
		}

		next := api.GenerateRequest{
			Model:  modelName,
			Prompt: "Say goodbye.",
		}
		withoutContext := debugGenerateRender(ctx, t, client, next)
		next.Context = first.Context
		withContext := debugGenerateRender(ctx, t, client, next)

		if withContext == withoutContext {
			t.Fatal("context debug render did not change the prompt")
		}
		if !strings.HasSuffix(withContext, withoutContext) {
			t.Fatalf("context render should preserve current native template as suffix\nwith context: %q\nwithout:      %q", withContext, withoutContext)
		}
	})

	t.Run("images use completion media markers", func(t *testing.T) {
		resp := debugGenerate(ctx, t, client, api.GenerateRequest{
			Model:  modelName,
			Prompt: "compare [img] and [img]",
			Images: []api.ImageData{
				[]byte("first-test-image"),
				[]byte("second-test-image"),
			},
		})

		if resp.DebugInfo == nil {
			t.Fatal("missing debug info")
		}
		if resp.DebugInfo.ImageCount != 2 {
			t.Fatalf("image count = %d, want 2", resp.DebugInfo.ImageCount)
		}
		rendered := resp.DebugInfo.RenderedTemplate
		if !strings.Contains(rendered, "[img-0]") || !strings.Contains(rendered, "[img-1]") {
			t.Fatalf("rendered template missing image markers: %q", rendered)
		}
	})
}

func TestGenerateNativeJinjaImages(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_EXISTING") != "" {
		t.Skip("requires a test-managed server started with OLLAMA_GO_TEMPLATE=0")
	}
	t.Setenv("OLLAMA_GO_TEMPLATE", "0")
	skipUnderMinVRAM(t, 6)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	_, _, ollamaHome := decodeTestImages(t)

	t.Run("actual image processing", func(t *testing.T) {
		for _, baseModel := range testModels([]string{"gemma3:4b"}) {
			t.Run(baseModel, func(t *testing.T) {
				setupVisionModel(ctx, t, client, baseModel)

				modelName := createGenerateJinjaModel(ctx, t, client, baseModel)
				defer func() {
					if err := client.Delete(context.Background(), &api.DeleteRequest{Model: modelName}); err != nil {
						t.Logf("failed to delete %s: %v", modelName, err)
					}
				}()

				req := api.GenerateRequest{
					Model:  modelName,
					Prompt: "Describe what you see in this image briefly.",
					Images: []api.ImageData{ollamaHome},
					Options: map[string]any{
						"seed":        42,
						"temperature": 0.0,
					},
					KeepAlive: &api.Duration{Duration: 10 * time.Second},
				}

				resp := debugGenerate(ctx, t, client, req)
				if resp.DebugInfo == nil {
					t.Fatal("missing debug info")
				}
				if resp.DebugInfo.ImageCount != 1 {
					t.Fatalf("image count = %d, want 1", resp.DebugInfo.ImageCount)
				}
				rendered := resp.DebugInfo.RenderedTemplate
				if strings.Contains(rendered, "GO_TEMPLATE_SENTINEL") {
					t.Fatalf("generate with images used Go TEMPLATE instead of native chat template: %q", rendered)
				}
				if !strings.Contains(rendered, "[img-0]") {
					t.Fatalf("rendered template missing image marker: %q", rendered)
				}

				DoGenerate(ctx, t, client, req, []string{
					"llama", "animal", "build", "model", "open", "cartoon", "character",
				}, 120*time.Second, 30*time.Second)
			})
		}
	})
}

func createGenerateJinjaModel(ctx context.Context, t *testing.T, client *api.Client, baseModel string) string {
	t.Helper()

	modelName := fmt.Sprintf("test-generate-jinja-%d", time.Now().UnixNano())
	stream := false
	err := client.Create(ctx, &api.CreateRequest{
		Model:    modelName,
		From:     baseModel,
		Stream:   &stream,
		Template: `GO_TEMPLATE_SENTINEL {{ if .System }}{{ .System }} {{ end }}{{ if .Prompt }}{{ .Prompt }}{{ end }}{{ range .Messages }}{{ .Content }}{{ end }}`,
	}, func(api.ProgressResponse) error {
		return nil
	})
	if err != nil {
		t.Fatalf("create %s from %s: %v", modelName, baseModel, err)
	}
	return modelName
}

func debugGenerateRender(ctx context.Context, t *testing.T, client *api.Client, req api.GenerateRequest) string {
	t.Helper()
	resp := debugGenerate(ctx, t, client, req)
	if resp.DebugInfo == nil {
		t.Fatal("missing debug info")
	}
	return resp.DebugInfo.RenderedTemplate
}

func debugGenerate(ctx context.Context, t *testing.T, client *api.Client, req api.GenerateRequest) api.GenerateResponse {
	t.Helper()

	stream := false
	req.Stream = &stream
	req.DebugRenderOnly = true

	var final api.GenerateResponse
	if err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
		final = resp
		return nil
	}); err != nil {
		if strings.Contains(err.Error(), "requires more system memory") {
			t.Skipf("model is too large for this test system: %v", err)
		}
		t.Fatalf("generate debug render failed: %v", err)
	}
	return final
}

func debugChatRender(ctx context.Context, t *testing.T, client *api.Client, req api.ChatRequest) string {
	t.Helper()

	stream := false
	req.Stream = &stream
	req.DebugRenderOnly = true

	var final api.ChatResponse
	if err := client.Chat(ctx, &req, func(resp api.ChatResponse) error {
		final = resp
		return nil
	}); err != nil {
		if strings.Contains(err.Error(), "requires more system memory") {
			t.Skipf("model is too large for this test system: %v", err)
		}
		t.Fatalf("chat debug render failed: %v", err)
	}
	if final.DebugInfo == nil {
		t.Fatal("missing debug info")
	}
	return final.DebugInfo.RenderedTemplate
}

func generateOnce(ctx context.Context, t *testing.T, client *api.Client, req api.GenerateRequest) api.GenerateResponse {
	t.Helper()

	stream := false
	req.Stream = &stream

	var final api.GenerateResponse
	if err := client.Generate(ctx, &req, func(resp api.GenerateResponse) error {
		final = resp
		return nil
	}); err != nil {
		if strings.Contains(err.Error(), "requires more system memory") {
			t.Skipf("model is too large for this test system: %v", err)
		}
		t.Fatalf("generate failed: %v", err)
	}
	return final
}
