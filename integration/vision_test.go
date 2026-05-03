//go:build integration

package integration

import (
	"context"
	"encoding/base64"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// Default set of vision models to test. When OLLAMA_TEST_MODEL is set,
// only that model is tested (with a capability check for vision).
var defaultVisionModels = []string{
	"nemotron3:33b",
	"gemma4",
	"gemma3",
	"llama3.2-vision",
	"qwen2.5vl",
	"qwen3-vl:8b",
}

// decodeTestImages returns the test images.
func decodeTestImages(t *testing.T) (abbeyRoad, docs, ollamaHome api.ImageData) {
	t.Helper()
	var err error
	abbeyRoad, err = base64.StdEncoding.DecodeString(imageEncoding)
	if err != nil {
		t.Fatalf("decode abbey road image: %v", err)
	}
	docs, err = base64.StdEncoding.DecodeString(imageEncodingDocs)
	if err != nil {
		t.Fatalf("decode docs image: %v", err)
	}
	ollamaHome, err = base64.StdEncoding.DecodeString(imageEncodingOllamaHome)
	if err != nil {
		t.Fatalf("decode ollama home image: %v", err)
	}
	return
}

// skipIfNoVisionOverride skips the entire test (at parent level) when
// OLLAMA_TEST_MODEL is set to a non-vision model. This prevents the parent
// test from reporting PASS when all subtests are skipped.
func skipIfNoVisionOverride(t *testing.T) {
	t.Helper()
	if testModel == "" {
		return
	}
	// Check actual model capabilities via the API rather than a hardcoded list.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	resp, err := client.Show(ctx, &api.ShowRequest{Name: testModel})
	if err != nil {
		return // let the test proceed and fail naturally
	}
	if len(resp.Capabilities) > 0 && !slices.Contains(resp.Capabilities, model.CapabilityVision) {
		t.Skipf("model override %q does not have vision capability (has %v)", testModel, resp.Capabilities)
	}
}

// setupVisionModel pulls the model, preloads it, and skips if not GPU-loaded.
func setupVisionModel(ctx context.Context, t *testing.T, client *api.Client, model string) {
	t.Helper()
	if testModel == "" {
		pullOrSkip(ctx, t, client, model)
	}
	skipIfModelTooLargeForVRAM(ctx, t, client, model)
	requireCapability(ctx, t, client, model, "vision")
	err := client.Generate(ctx, &api.GenerateRequest{Model: model}, func(response api.GenerateResponse) error { return nil })
	if err != nil {
		t.Fatalf("failed to load model %s: %s", model, err)
	}
	skipIfNotGPULoaded(ctx, t, client, model, 80)
}

// TestVisionMultiTurn sends an image, gets a response, then asks follow-up
// questions about the same image. This verifies that the KV cache correctly
// handles cached image tokens across turns.
func TestVisionMultiTurn(t *testing.T) {
	skipUnderMinVRAM(t, 16)
	skipIfNoVisionOverride(t)

	// Models that fail on multi-turn detail questions (e.g. misidentifying objects).
	skipModels := map[string]string{
		"gemma3":          "misidentifies briefcase as smartphone on turn 3",
		"llama3.2-vision": "miscounts animals (says 3 instead of 4) on turn 2",
	}

	for _, model := range testModels(defaultVisionModels) {
		t.Run(model, func(t *testing.T) {
			if reason, ok := skipModels[model]; ok && testModel == "" {
				t.Skipf("skipping: %s", reason)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupVisionModel(ctx, t, client, model)
			abbeyRoad, _, _ := decodeTestImages(t)

			// Turn 1: describe the image
			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Describe this image briefly.",
						Images:  []api.ImageData{abbeyRoad},
					},
				},
				Stream: &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options:   map[string]any{"temperature": 0.0, "seed": 42},
			}
			resp1 := DoChat(ctx, t, client, req, []string{
				"llama", "cross", "walk", "road", "animal", "cartoon",
			}, 120*time.Second, 30*time.Second)
			if resp1 == nil {
				t.Fatal("no response from turn 1")
			}

			// Turn 2: follow-up about count
			req.Messages = append(req.Messages,
				*resp1,
				api.Message{Role: "user", Content: "How many animals are in the image?"},
			)
			resp2 := DoChat(ctx, t, client, req, []string{
				"four", "4", "three", "3",
			}, 60*time.Second, 30*time.Second)
			if resp2 == nil {
				t.Fatal("no response from turn 2")
			}

			// Turn 3: follow-up about specific detail
			req.Messages = append(req.Messages,
				*resp2,
				api.Message{Role: "user", Content: "Is any animal carrying something? What is it?"},
			)
			DoChat(ctx, t, client, req, []string{
				"briefcase", "suitcase", "bag", "case", "luggage",
			}, 60*time.Second, 30*time.Second)
		})
	}
}

// TestVisionObjectCounting asks the model to count objects in an image.
func TestVisionObjectCounting(t *testing.T) {
	skipUnderMinVRAM(t, 16)
	skipIfNoVisionOverride(t)

	skipModels := map[string]string{
		"llama3.2-vision": "consistently miscounts (says 3 instead of 4)",
	}

	for _, model := range testModels(defaultVisionModels) {
		t.Run(model, func(t *testing.T) {
			if reason, ok := skipModels[model]; ok && testModel == "" {
				t.Skipf("skipping: %s", reason)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupVisionModel(ctx, t, client, model)
			_, docs, _ := decodeTestImages(t)

			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "How many animals are shown in this image? Answer with just the number.",
						Images:  []api.ImageData{docs},
					},
				},
				Stream: &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options:   map[string]any{"temperature": 0.0, "seed": 42},
			}
			DoChat(ctx, t, client, req, []string{"4", "four"}, 120*time.Second, 30*time.Second)
		})
	}
}

// TestVisionSceneUnderstanding tests whether the model can identify
// cultural references and scene context from an image.
func TestVisionSceneUnderstanding(t *testing.T) {
	skipUnderMinVRAM(t, 16)
	skipIfNoVisionOverride(t)

	// Models known to be too small or not capable enough for cultural reference detection.
	skipModels := map[string]string{
		"llama3.2-vision": "3B model lacks cultural reference knowledge",
		"minicpm-v":       "too small for cultural reference detection",
	}

	for _, model := range testModels(defaultVisionModels) {
		t.Run(model, func(t *testing.T) {
			if reason, ok := skipModels[model]; ok && testModel == "" {
				t.Skipf("skipping: %s", reason)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupVisionModel(ctx, t, client, model)
			abbeyRoad, _, _ := decodeTestImages(t)

			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What famous image or album cover is this a parody of?",
						Images:  []api.ImageData{abbeyRoad},
					},
				},
				Stream: &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options:   map[string]any{"temperature": 0.0, "seed": 42},
			}
			DoChat(ctx, t, client, req, []string{
				"abbey road", "beatles", "abbey", "llama",
			}, 120*time.Second, 30*time.Second)
		})
	}
}

// TestVisionSpatialReasoning tests the model's ability to identify
// objects based on their spatial position in the image.
func TestVisionSpatialReasoning(t *testing.T) {
	skipUnderMinVRAM(t, 16)
	skipIfNoVisionOverride(t)

	for _, model := range testModels(defaultVisionModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupVisionModel(ctx, t, client, model)
			_, docs, _ := decodeTestImages(t)

			// The docs image has: leftmost llama on laptop with glasses,
			// rightmost llama sleeping.
			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "What is the animal on the far left doing in this image?",
						Images:  []api.ImageData{docs},
					},
				},
				Stream: &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options:   map[string]any{"temperature": 0.0, "seed": 42},
			}
			DoChat(ctx, t, client, req, []string{
				"laptop", "computer", "typing", "working", "desk", "writing", "pen", "glasses", "reading",
			}, 120*time.Second, 30*time.Second)
		})
	}
}

// TestVisionDetailRecognition tests whether the model can identify
// small details like accessories in an image.
func TestVisionDetailRecognition(t *testing.T) {
	skipUnderMinVRAM(t, 16)
	skipIfNoVisionOverride(t)

	for _, model := range testModels(defaultVisionModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupVisionModel(ctx, t, client, model)
			_, docs, _ := decodeTestImages(t)

			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Are any of the animals wearing glasses? Describe what you see.",
						Images:  []api.ImageData{docs},
					},
				},
				Stream: &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options:   map[string]any{"temperature": 0.0, "seed": 42},
			}
			DoChat(ctx, t, client, req, []string{
				"glasses", "spectacles", "eyeglasses",
			}, 120*time.Second, 30*time.Second)
		})
	}
}

// TestVisionMultiImage sends two images in a single message and asks
// the model to compare and contrast them. This exercises multi-image
// encoding and cross-image reasoning.
func TestVisionMultiImage(t *testing.T) {
	skipUnderMinVRAM(t, 16)
	skipIfNoVisionOverride(t)

	// Multi-image support varies across models.
	skipModels := map[string]string{
		"llama3.2-vision": "does not support multi-image input",
	}

	for _, model := range testModels(defaultVisionModels) {
		t.Run(model, func(t *testing.T) {
			if reason, ok := skipModels[model]; ok && testModel == "" {
				t.Skipf("skipping: %s", reason)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupVisionModel(ctx, t, client, model)
			abbeyRoad, docs, _ := decodeTestImages(t)

			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "I'm showing you two images. What do they have in common, and how are they different?",
						Images:  []api.ImageData{abbeyRoad, docs},
					},
				},
				Stream: &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options:   map[string]any{"temperature": 0.0, "seed": 42},
			}
			// Both images feature cartoon llamas/alpacas — the model should
			// note the common subject and the different settings.
			DoChat(ctx, t, client, req, []string{
				"llama", "alpaca", "animal", "cartoon",
			}, 120*time.Second, 30*time.Second)
		})
	}
}

// TestVisionImageDescription verifies that the model can describe the contents
// of the ollama homepage image (a cartoon llama with "Start building with
// open models" text). Basic sanity check that the vision pipeline works.
func TestVisionImageDescription(t *testing.T) {
	skipUnderMinVRAM(t, 16)
	skipIfNoVisionOverride(t)

	for _, model := range testModels(defaultVisionModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupVisionModel(ctx, t, client, model)
			_, _, ollamaHome := decodeTestImages(t)

			req := api.ChatRequest{
				Model: model,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Describe what you see in this image briefly.",
						Images:  []api.ImageData{ollamaHome},
					},
				},
				Stream: &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
				Options:   map[string]any{"temperature": 0.0, "seed": 42},
			}
			DoChat(ctx, t, client, req, []string{
				"llama", "animal", "build", "model", "open", "cartoon", "character",
			}, 120*time.Second, 30*time.Second)
		})
	}
}
