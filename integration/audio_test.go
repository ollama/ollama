//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

var defaultAudioModels = []string{
	"nemotron3:33b",
	"gemma4:e2b",
	"gemma4:e4b",
}

// decodeTestAudio returns the test audio clip ("Why is the sky blue?", 16kHz mono WAV).
func decodeTestAudio(t *testing.T) api.ImageData {
	t.Helper()
	data, err := base64.StdEncoding.DecodeString(audioEncodingPrompt)
	if err != nil {
		t.Fatalf("failed to decode test audio: %v", err)
	}
	return data
}

// setupAudioModel pulls the model, preloads it, and skips if it doesn't support audio.
func setupAudioModel(ctx context.Context, t *testing.T, client *api.Client, model string) {
	t.Helper()
	if testModel == "" {
		pullOrSkip(ctx, t, client, model)
	}
	skipIfModelTooLargeForVRAM(ctx, t, client, model)
	requireCapability(ctx, t, client, model, "audio")
	err := client.Generate(ctx, &api.GenerateRequest{Model: model}, func(response api.GenerateResponse) error { return nil })
	if err != nil {
		t.Fatalf("failed to load model %s: %s", model, err)
	}
}

// TestAudioTranscription tests that the model can transcribe audio to text.
func TestAudioTranscription(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audio := decodeTestAudio(t)
			noThink := &api.ThinkValue{Value: false}

			req := api.ChatRequest{
				Model: model,
				Think: noThink,
				Messages: []api.Message{
					{
						Role:    "system",
						Content: "Transcribe the audio exactly as spoken. Output only the transcription.",
					},
					{
						Role:    "user",
						Content: "Transcribe this audio.",
						Images:  []api.ImageData{audio},
					},
				},
				Stream: &stream,
				Options: map[string]any{
					"temperature": 0,
					"seed":        123,
					"num_predict": 50,
				},
			}

			// The audio says "Why is the sky blue?" — expect key words in transcription.
			DoChat(ctx, t, client, req, []string{"sky", "blue"}, 60*time.Second, 10*time.Second)
		})
	}
}

// TestAudioResponse tests that the model can respond to a spoken question.
func TestAudioResponse(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, _, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audio := decodeTestAudio(t)
			noThink := &api.ThinkValue{Value: false}

			req := api.ChatRequest{
				Model: model,
				Think: noThink,
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "",
						Images:  []api.ImageData{audio},
					},
				},
				Stream: &stream,
				Options: map[string]any{
					"temperature": 0,
					"seed":        123,
					"num_predict": 200,
				},
			}

			// The audio asks "Why is the sky blue?" — expect an answer about light/scattering.
			DoChat(ctx, t, client, req, []string{
				"scatter", "light", "blue", "atmosphere", "wavelength", "rayleigh",
			}, 60*time.Second, 10*time.Second)
		})
	}
}

// TestOpenAIAudioTranscription tests the /v1/audio/transcriptions endpoint.
func TestOpenAIAudioTranscription(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, endpoint, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audioBytes := decodeTestAudio(t)

			// Build multipart form request.
			var body bytes.Buffer
			writer := multipart.NewWriter(&body)
			writer.WriteField("model", model)
			part, err := writer.CreateFormFile("file", "prompt.wav")
			if err != nil {
				t.Fatal(err)
			}
			part.Write(audioBytes)
			writer.Close()

			url := fmt.Sprintf("http://%s/v1/audio/transcriptions", endpoint)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &body)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", writer.FormDataContentType())

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				respBody, _ := io.ReadAll(resp.Body)
				t.Fatalf("expected 200, got %d: %s", resp.StatusCode, string(respBody))
			}

			respBody, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}

			text := strings.ToLower(string(respBody))
			if !strings.Contains(text, "sky") && !strings.Contains(text, "blue") {
				t.Errorf("transcription response missing expected words, got: %s", string(respBody))
			}
		})
	}
}

// TestOpenAIChatWithAudio tests /v1/chat/completions with input_audio content.
func TestOpenAIChatWithAudio(t *testing.T) {
	for _, model := range testModels(defaultAudioModels) {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			client, endpoint, cleanup := InitServerConnection(ctx, t)
			defer cleanup()

			setupAudioModel(ctx, t, client, model)
			audioB64 := audioEncodingPrompt

			reqBody := fmt.Sprintf(`{
				"model": %q,
				"messages": [{
					"role": "user",
					"content": [
						{"type": "input_audio", "input_audio": {"data": %q, "format": "wav"}}
					]
				}],
				"temperature": 0,
				"seed": 123,
				"max_tokens": 200,
				"think": false
			}`, model, strings.TrimSpace(audioB64))

			url := fmt.Sprintf("http://%s/v1/chat/completions", endpoint)
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(reqBody))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				respBody, _ := io.ReadAll(resp.Body)
				t.Fatalf("expected 200, got %d: %s", resp.StatusCode, string(respBody))
			}

			respBytes, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("failed to read response: %v", err)
			}

			var result struct {
				Choices []struct {
					Message struct {
						Content   string `json:"content"`
						Reasoning string `json:"reasoning"`
					} `json:"message"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(respBytes, &result); err != nil {
				t.Fatalf("failed to decode response: %v", err)
			}

			if len(result.Choices) == 0 {
				t.Fatal("no choices in response")
			}

			text := strings.ToLower(result.Choices[0].Message.Content + " " + result.Choices[0].Message.Reasoning)
			found := false
			for _, word := range []string{"sky", "blue", "scatter", "light", "atmosphere"} {
				if strings.Contains(text, word) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("response missing expected words about sky/blue/light, got: %s", result.Choices[0].Message.Content)
			}
		})
	}
}
