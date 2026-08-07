package mlxrunner

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	sample "github.com/ollama/ollama/x/mlxrunner/sample"
)

// circlePNG synthesizes a fixture: a solid red circle on white, centered at
// (cx, cy) in a 384×512 landscape-free canvas.
func circlePNG(t *testing.T, w, h, cx, cy, r int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx, dy := x-cx, y-cy
			if dx*dx+dy*dy <= r*r {
				img.SetRGBA(x, y, color.RGBA{R: 200, G: 30, B: 30, A: 255})
			} else {
				img.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// TestVisionEndToEnd loads a real gemma4 MLX checkpoint and asks it about a
// synthesized red circle. Gated on model availability: enable with
// OLLAMA_VISION_E2E=1 (model override via OLLAMA_VISION_E2E_MODEL).
func TestVisionEndToEnd(t *testing.T) {
	if os.Getenv("OLLAMA_VISION_E2E") == "" {
		t.Skip("set OLLAMA_VISION_E2E=1 to run the vision end-to-end smoke test")
	}
	modelName := os.Getenv("OLLAMA_VISION_E2E_MODEL")
	if modelName == "" {
		modelName = "gemma4:12b-nvfp4"
	}
	if root, err := model.Open(modelName); err != nil {
		t.Skipf("model %s not available: %v", modelName, err)
	} else {
		root.Close()
	}

	worker, err := mlxthread.Start("mlxrunner-test", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	defer worker.Stop(context.Background(), func() {
		mlx.Sweep()
		mlx.ClearCache()
	})

	r := &Runner{Requests: make(chan Request), mlxThread: worker}
	if err := worker.Do(context.Background(), func() error {
		return r.Load(modelName)
	}); err != nil {
		t.Fatalf("load %s: %v", modelName, err)
	}

	ask := func(t *testing.T, question string, imgData []byte) string {
		t.Helper()
		opts := api.DefaultOptions()
		opts.NumPredict = 12
		opts.Temperature = 0
		opts.ImageMaxTokens = 280 // one ladder rung down: faster, ample for a circle

		request := Request{
			CompletionRequest: CompletionRequest{
				Prompt:  "<bos><|turn>user\n[img-0] " + question + "<turn|>\n<|turn>model\n",
				Options: opts,
				Media:   []llm.MediaData{{Data: imgData, ID: 0, Kind: llm.MediaKindImage}},
			},
			Responses: make(chan CompletionResponse),
		}
		request.SamplerOpts = sample.Options{
			Temperature: opts.Temperature,
			TopP:        opts.TopP,
			MinP:        opts.MinP,
			TopK:        opts.TopK,
			Seed:        opts.Seed,
			UseSeed:     opts.Seed >= 0,
		}

		if err := r.Prepare(&request); err != nil {
			t.Fatalf("Prepare: %v", err)
		}
		if len(request.VisionInputs) != 1 || len(request.VisionSpans) != 1 {
			t.Fatalf("expected 1 vision input/span, got %d/%d", len(request.VisionInputs), len(request.VisionSpans))
		}
		t.Logf("prompt tokens: %d, soft tokens: %d, span: %v",
			len(request.Tokens), request.VisionInputs[0].SoftTokens(), request.VisionSpans[0])

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		request.Ctx = ctx

		var sb strings.Builder
		done := make(chan struct{})
		go func() {
			defer close(done)
			for resp := range request.Responses {
				sb.WriteString(resp.Content)
				if resp.Error != nil {
					t.Errorf("runner error: %v", resp.Error)
					return
				}
				if resp.Done {
					return
				}
			}
		}()

		if err := worker.Do(ctx, func() error {
			defer close(request.Responses)
			return r.TextGenerationPipeline(ctx, request)
		}); err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		<-done
		t.Logf("model output: %q", sb.String())
		return strings.ToLower(sb.String())
	}

	t.Run("shape", func(t *testing.T) {
		out := ask(t, "What shape is in this image? Answer with one word.",
			circlePNG(t, 384, 384, 192, 192, 120))
		if !strings.Contains(out, "circle") {
			t.Fatalf("expected the model to name the circle, got %q", out)
		}
	})

	// Left-of-center circle on a wide canvas: catches transposed position
	// tables or swapped RoPE axes, which a shape question survives.
	t.Run("spatial", func(t *testing.T) {
		out := ask(t, "Is the red circle in the left half or the right half of this image? Answer left or right.",
			circlePNG(t, 768, 384, 130, 192, 90))
		if !strings.Contains(out, "left") || strings.Contains(out, "right") {
			t.Fatalf("expected the model to place the circle on the left, got %q", out)
		}
	})
}
