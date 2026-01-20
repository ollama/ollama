//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	imagegenapi "github.com/ollama/ollama/x/imagegen/api"
)

func TestImageGeneration(t *testing.T) {
	skipUnderMinVRAM(t, 8)

	type testCase struct {
		imageGenModel string
		visionModel   string
		prompt        string
		expectedWords []string
	}

	testCases := []testCase{
		{
			imageGenModel: "jmorgan/z-image-turbo",
			visionModel:   "llama3.2-vision",
			prompt:        "A cartoon style llama flying like a superhero through the air with clouds in the background",
			expectedWords: []string{"llama", "flying", "cartoon", "cloud", "sky", "superhero", "air", "animal", "camelid"},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s->%s", tc.imageGenModel, tc.visionModel), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()

			client, testEndpoint, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			// Pull both models
			if err := PullIfMissing(ctx, client, tc.imageGenModel); err != nil {
				t.Fatalf("failed to pull image gen model: %v", err)
			}
			if err := PullIfMissing(ctx, client, tc.visionModel); err != nil {
				t.Fatalf("failed to pull vision model: %v", err)
			}

			// Generate the image
			t.Logf("Generating image with prompt: %s", tc.prompt)
			imageBase64, err := generateImage(ctx, testEndpoint, tc.imageGenModel, tc.prompt)
			if err != nil {
				if strings.Contains(err.Error(), "image generation not available") {
					t.Skip("Target system does not support image generation")
				} else if strings.Contains(err.Error(), "executable file not found in") { // Windows pattern, not yet supported
					t.Skip("Windows does not support image generation yet")
				} else if strings.Contains(err.Error(), "CUDA driver version is insufficient") {
					t.Skip("Driver is too old")
				} else if strings.Contains(err.Error(), "insufficient memory for image generation") {
					t.Skip("insufficient memory for image generation")
				} else if strings.Contains(err.Error(), "error while loading shared libraries: libcuda.so.1") { // AMD GPU or CPU
					t.Skip("CUDA GPU is not available")
				} else if strings.Contains(err.Error(), "ollama-mlx: no such file or directory") {
					// most likely linux arm - not supported yet
					t.Skip("unsupported architecture")
				}
				t.Fatalf("failed to generate image: %v", err)
			}

			imageData, err := base64.StdEncoding.DecodeString(imageBase64)
			if err != nil {
				t.Fatalf("failed to decode image: %v", err)
			}
			t.Logf("Generated image: %d bytes", len(imageData))

			// Preload vision model and check GPU loading
			err = client.Generate(ctx, &api.GenerateRequest{Model: tc.visionModel}, func(response api.GenerateResponse) error { return nil })
			if err != nil {
				t.Fatalf("failed to load vision model: %v", err)
			}

			// Use vision model to describe the image
			chatReq := api.ChatRequest{
				Model: tc.visionModel,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "Describe this image in detail. What is shown? What style is it? What is the main subject doing?",
						Images:  []api.ImageData{imageData},
					},
				},
				Stream: &stream,
				Options: map[string]any{
					"seed":        42,
					"temperature": 0.0,
				},
			}

			// Verify the vision model's response contains expected keywords
			response := DoChat(ctx, t, client, chatReq, tc.expectedWords, 240*time.Second, 30*time.Second)
			if response != nil {
				t.Logf("Vision model response: %s", response.Content)

				// Additional detailed check for keywords
				content := strings.ToLower(response.Content)
				foundWords := []string{}
				missingWords := []string{}
				for _, word := range tc.expectedWords {
					if strings.Contains(content, word) {
						foundWords = append(foundWords, word)
					} else {
						missingWords = append(missingWords, word)
					}
				}
				t.Logf("Found keywords: %v", foundWords)
				if len(missingWords) > 0 {
					t.Logf("Missing keywords (at least one was found so test passed): %v", missingWords)
				}
			}
		})
	}
}

// generateImage calls the OpenAI-compatible image generation API and returns the base64 image data
func generateImage(ctx context.Context, endpoint, model, prompt string) (string, error) {
	reqBody := imagegenapi.ImageGenerationRequest{
		Model:          model,
		Prompt:         prompt,
		N:              1,
		Size:           "512x512",
		ResponseFormat: "b64_json",
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://%s/v1/images/generations", endpoint)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var buf bytes.Buffer
		buf.ReadFrom(resp.Body)
		return "", fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, buf.String())
	}

	var genResp imagegenapi.ImageGenerationResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if len(genResp.Data) == 0 {
		return "", fmt.Errorf("no image data in response")
	}

	return genResp.Data[0].B64JSON, nil
}
