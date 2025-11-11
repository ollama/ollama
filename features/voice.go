package features

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"
)

// VoiceHandler handles voice input/output features
type VoiceHandler struct {
	apiKey string
	client *http.Client
}

// NewVoiceHandler creates a new voice handler
func NewVoiceHandler(apiKey string) *VoiceHandler {
	return &VoiceHandler{
		apiKey: apiKey,
		client: &http.Client{Timeout: 120 * time.Second},
	}
}

// Transcribe transcribes audio file to text (Whisper API)
func (vh *VoiceHandler) Transcribe(audioFile io.Reader, filename string) (string, error) {
	url := "https://api.openai.com/v1/audio/transcriptions"

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		return "", err
	}

	if _, err := io.Copy(part, audioFile); err != nil {
		return "", err
	}

	writer.WriteField("model", "whisper-1")
	writer.Close()

	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", "Bearer "+vh.apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := vh.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("transcription error: %s - %s", resp.Status, string(bodyBytes))
	}

	var result struct {
		Text string `json:"text"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	return result.Text, nil
}

// Synthesize synthesizes text to speech (TTS API)
func (vh *VoiceHandler) Synthesize(text string, voice string) ([]byte, error) {
	url := "https://api.openai.com/v1/audio/speech"

	if voice == "" {
		voice = "alloy" // default voice
	}

	payload := map[string]string{
		"model": "tts-1",
		"input": text,
		"voice": voice, // alloy, echo, fable, onyx, nova, shimmer
	}

	body, _ := json.Marshal(payload)

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+vh.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := vh.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("TTS error: %s - %s", resp.Status, string(bodyBytes))
	}

	return io.ReadAll(resp.Body)
}
