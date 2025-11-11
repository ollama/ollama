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

type AnthropicProvider struct {
	APIKey  string
	BaseURL string
	client  *http.Client
}

func NewAnthropicProvider(apiKey string) *AnthropicProvider {
	return &AnthropicProvider{
		APIKey:  apiKey,
		BaseURL: "https://api.anthropic.com/v1",
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *AnthropicProvider) GetName() string {
	return "Anthropic"
}

func (p *AnthropicProvider) GetType() string {
	return "anthropic"
}

func (p *AnthropicProvider) ListModels(ctx context.Context) ([]Model, error) {
	return []Model{
		{
			ID:            "claude-opus-4",
			Name:          "claude-opus-4",
			DisplayName:   "Claude Opus 4",
			ContextWindow: 200000,
			Capabilities:  []string{"chat", "tools", "vision", "thinking"},
		},
		{
			ID:            "claude-sonnet-4-5",
			Name:          "claude-sonnet-4-5",
			DisplayName:   "Claude Sonnet 4.5",
			ContextWindow: 200000,
			Capabilities:  []string{"chat", "tools", "vision"},
		},
		{
			ID:            "claude-haiku-4-5",
			Name:          "claude-haiku-4-5",
			DisplayName:   "Claude Haiku 4.5",
			ContextWindow: 200000,
			Capabilities:  []string{"chat", "tools", "vision"},
		},
	}, nil
}

func (p *AnthropicProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	startTime := time.Now()

	// Extract system message
	var systemPrompt string
	var messages []map[string]interface{}

	for _, msg := range req.Messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
			continue
		}

		messages = append(messages, map[string]interface{}{
			"role": msg.Role,
			"content": []map[string]interface{}{
				{"type": "text", "text": msg.Content},
			},
		})
	}

	anthropicReq := map[string]interface{}{
		"model":      req.Model,
		"messages":   messages,
		"max_tokens": 4096,
	}

	if systemPrompt != "" {
		anthropicReq["system"] = systemPrompt
	}

	if req.Temperature != nil {
		anthropicReq["temperature"] = *req.Temperature
	}

	if req.MaxTokens != nil {
		anthropicReq["max_tokens"] = *req.MaxTokens
	}

	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.BaseURL+"/messages",
		bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("x-api-key", p.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("content-type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Anthropic API error: %s - %s", resp.Status, string(bodyBytes))
	}

	var anthropicResp struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		} `json:"content"`
		Model string `json:"model"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&anthropicResp); err != nil {
		return nil, err
	}

	duration := time.Since(startTime)
	tokensPerSec := float64(0)
	if duration.Seconds() > 0 {
		tokensPerSec = float64(anthropicResp.Usage.OutputTokens) / duration.Seconds()
	}

	var textContent string
	for _, block := range anthropicResp.Content {
		if block.Type == "text" {
			textContent += block.Text
		}
	}

	response := &ChatResponse{
		ID:    anthropicResp.ID,
		Model: anthropicResp.Model,
		Message: Message{
			Role:    "assistant",
			Content: textContent,
		},
		Usage: UsageMetrics{
			InputTokens:  anthropicResp.Usage.InputTokens,
			OutputTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:  anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		},
		Metrics: PerformanceMetrics{
			DurationMs:      duration.Milliseconds(),
			TokensPerSecond: tokensPerSec,
		},
	}

	return response, nil
}

func (p *AnthropicProvider) ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("streaming not implemented yet")
}

func (p *AnthropicProvider) ValidateCredentials(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	return err
}

func (p *AnthropicProvider) GetPricing(modelName string) (*ModelPricing, error) {
	pricing := map[string]ModelPricing{
		"claude-opus-4": {
			InputPricePer1M:  15.0,
			OutputPricePer1M: 75.0,
			ContextWindow:    200000,
		},
		"claude-sonnet-4-5": {
			InputPricePer1M:  3.0,
			OutputPricePer1M: 15.0,
			ContextWindow:    200000,
		},
		"claude-haiku-4-5": {
			InputPricePer1M:  0.8,
			OutputPricePer1M: 4.0,
			ContextWindow:    200000,
		},
	}

	if p, ok := pricing[modelName]; ok {
		return &p, nil
	}

	return nil, fmt.Errorf("pricing not found for model: %s", modelName)
}
