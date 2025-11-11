package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type OpenAIProvider struct {
	APIKey  string
	BaseURL string
	client  *http.Client
}

func NewOpenAIProvider(apiKey string, baseURL string) *OpenAIProvider {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}
	return &OpenAIProvider{
		APIKey:  apiKey,
		BaseURL: baseURL,
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *OpenAIProvider) GetName() string {
	return "OpenAI"
}

func (p *OpenAIProvider) GetType() string {
	return "openai"
}

func (p *OpenAIProvider) ListModels(ctx context.Context) ([]Model, error) {
	// Return static list for now
	return []Model{
		{
			ID:            "gpt-4",
			Name:          "gpt-4",
			DisplayName:   "GPT-4",
			ContextWindow: 8192,
			Capabilities:  []string{"chat", "tools"},
		},
		{
			ID:            "gpt-3.5-turbo",
			Name:          "gpt-3.5-turbo",
			DisplayName:   "GPT-3.5 Turbo",
			ContextWindow: 16385,
			Capabilities:  []string{"chat", "tools"},
		},
	}, nil
}

func (p *OpenAIProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	startTime := time.Now()

	// Convert to OpenAI format
	openAIReq := map[string]interface{}{
		"model":    req.Model,
		"messages": req.Messages,
	}

	if req.Temperature != nil {
		openAIReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		openAIReq["max_tokens"] = *req.MaxTokens
	}

	body, err := json.Marshal(openAIReq)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.BaseURL+"/chat/completions",
		bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Authorization", "Bearer "+p.APIKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("OpenAI API error: %s - %s", resp.Status, string(bodyBytes))
	}

	var openAIResp struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		return nil, err
	}

	duration := time.Since(startTime)
	tokensPerSec := float64(0)
	if duration.Seconds() > 0 {
		tokensPerSec = float64(openAIResp.Usage.CompletionTokens) / duration.Seconds()
	}

	response := &ChatResponse{
		ID:    openAIResp.ID,
		Model: openAIResp.Model,
		Message: Message{
			Role:    "assistant",
			Content: "",
		},
		Usage: UsageMetrics{
			InputTokens:  openAIResp.Usage.PromptTokens,
			OutputTokens: openAIResp.Usage.CompletionTokens,
			TotalTokens:  openAIResp.Usage.TotalTokens,
		},
		Metrics: PerformanceMetrics{
			DurationMs:      duration.Milliseconds(),
			TokensPerSecond: tokensPerSec,
		},
	}

	if len(openAIResp.Choices) > 0 {
		response.Message.Content = openAIResp.Choices[0].Message.Content
	}

	return response, nil
}

func (p *OpenAIProvider) ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("streaming not implemented yet")
}

func (p *OpenAIProvider) ValidateCredentials(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	return err
}

func (p *OpenAIProvider) GetPricing(modelName string) (*ModelPricing, error) {
	pricing := map[string]ModelPricing{
		"gpt-4": {
			InputPricePer1M:  30.0,
			OutputPricePer1M: 60.0,
			ContextWindow:    8192,
		},
		"gpt-3.5-turbo": {
			InputPricePer1M:  0.5,
			OutputPricePer1M: 1.5,
			ContextWindow:    16385,
		},
	}

	if p, ok := pricing[modelName]; ok {
		return &p, nil
	}

	// Check with prefix
	for key, value := range pricing {
		if strings.HasPrefix(modelName, key) {
			return &value, nil
		}
	}

	return nil, fmt.Errorf("pricing not found for model: %s", modelName)
}
