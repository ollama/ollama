//go:build integration

package integration

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/video/videotest"
)

func registerVideoCases(models []string) {
	registerModelIntegrationCases("video-describe", models, runVideoDescribe)
}

// runVideoDescribe sends a generated clip of a red square sliding across a
// black frame and checks that the model sees it.
func runVideoDescribe(t *testing.T, model string) {
	if runtime.GOOS != "darwin" {
		t.Skip("video input is macOS-only")
	}
	skipUnderMinVRAM(t, 6)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	pullOrSkip(ctx, t, client, model)
	requireCapability(ctx, t, client, model, "video")
	clip, err := videotest.Clip{Width: 320, Height: 240, Frames: 30, FPS: 10}.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	req := api.ChatRequest{
		Model: model,
		Think: &api.ThinkValue{Value: false},
		Messages: []api.Message{
			{Role: "user", Content: "What color is the square in this video?", Images: []api.ImageData{clip}},
		},
		Stream:    &stream,
		Options:   map[string]any{"temperature": 0.0, "seed": 42},
		KeepAlive: &api.Duration{Duration: 10 * time.Second},
	}
	preloadGenerateModel(ctx, t, client, api.GenerateRequest{Model: model})
	skipIfNotGPULoaded(ctx, t, client, model, 80)
	DoChat(ctx, t, client, req, []string{"red"}, 120*time.Second, 30*time.Second)
}
