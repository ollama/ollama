package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type GoogleProvider struct {
	APIKey  string
	BaseURL string
	client  *http.Client
}

func NewGoogleProvider(apiKey string) *GoogleProvider {
	return &GoogleProvider{
		APIKey:  apiKey,
		BaseURL: "https://generativelanguage.googleapis.com/v1beta",
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *GoogleProvider) GetName() string {
	return "Google"
}

func (p *GoogleProvider) GetType() string {
	return "google"
}

func (p *GoogleProvider) ListModels(ctx context.Context) ([]Model, error) {
	return []Model{
		{
			ID:            "gemini-2.0-flash-exp",
			Name:          "gemini-2.0-flash-exp",
			DisplayName:   "Gemini 2.0 Flash Exp",
			ContextWindow: 1000000,
			Capabilities:  []string{"chat", "vision"},
		},
		{
			ID:            "gemini-1.5-pro",
			Name:          "gemini-1.5-pro",
			DisplayName:   "Gemini 1.5 Pro",
			ContextWindow: 2000000,
			Capabilities:  []string{"chat", "vision"},
		},
	}, nil
}

func (p *GoogleProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	startTime := time.Now()

	// Convert messages to Google format
	var contents []map[string]interface{}
	for _, msg := range req.Messages {
		role := "user"
		if msg.Role == "assistant" {
			role = "model"
		}
		contents = append(contents, map[string]interface{}{
			"role": role,
			"parts": []map[string]interface{}{
				{"text": msg.Content},
			},
		})
	}

	googleReq := map[string]interface{}{
		"contents": contents,
	}

	body, err := json.Marshal(googleReq)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", p.BaseURL, req.Model, p.APIKey)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Google API error: %s - %s", resp.Status, string(bodyBytes))
	}

	var googleResp struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&googleResp); err != nil {
		return nil, err
	}

	duration := time.Since(startTime)
	tokensPerSec := float64(0)
	if duration.Seconds() > 0 {
		tokensPerSec = float64(googleResp.UsageMetadata.CandidatesTokenCount) / duration.Seconds()
	}

	var textContent string
	if len(googleResp.Candidates) > 0 && len(googleResp.Candidates[0].Content.Parts) > 0 {
		textContent = googleResp.Candidates[0].Content.Parts[0].Text
	}

	response := &ChatResponse{
		ID:    fmt.Sprintf("google-%d", time.Now().Unix()),
		Model: req.Model,
		Message: Message{
			Role:    "assistant",
			Content: textContent,
		},
		Usage: UsageMetrics{
			InputTokens:  googleResp.UsageMetadata.PromptTokenCount,
			OutputTokens: googleResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  googleResp.UsageMetadata.TotalTokenCount,
		},
		Metrics: PerformanceMetrics{
			DurationMs:      duration.Milliseconds(),
			TokensPerSecond: tokensPerSec,
		},
	}

	return response, nil
}

func (p *GoogleProvider) ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("streaming not implemented yet")
}

func (p *GoogleProvider) ValidateCredentials(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	return err
}

func (p *GoogleProvider) GetPricing(modelName string) (*ModelPricing, error) {
	pricing := map[string]ModelPricing{
		"gemini-2.0-flash-exp": {
			InputPricePer1M:  0.0,
			OutputPricePer1M: 0.0,
			ContextWindow:    1000000,
		},
		"gemini-1.5-pro": {
			InputPricePer1M:  1.25,
			OutputPricePer1M: 5.0,
			ContextWindow:    2000000,
		},
	}

	if p, ok := pricing[modelName]; ok {
		return &p, nil
	}

	return nil, fmt.Errorf("pricing not found for model: %s", modelName)
}
