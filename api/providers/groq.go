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

type GroqProvider struct {
	APIKey  string
	BaseURL string
	client  *http.Client
}

func NewGroqProvider(apiKey string) *GroqProvider {
	return &GroqProvider{
		APIKey:  apiKey,
		BaseURL: "https://api.groq.com/openai/v1",
		client:  &http.Client{Timeout: 120 * time.Second},
	}
}

func (p *GroqProvider) GetName() string {
	return "Groq"
}

func (p *GroqProvider) GetType() string {
	return "groq"
}

func (p *GroqProvider) ListModels(ctx context.Context) ([]Model, error) {
	return []Model{
		{
			ID:            "llama-3.3-70b-versatile",
			Name:          "llama-3.3-70b-versatile",
			DisplayName:   "Llama 3.3 70B",
			ContextWindow: 128000,
			Capabilities:  []string{"chat"},
		},
		{
			ID:            "mixtral-8x7b-32768",
			Name:          "mixtral-8x7b-32768",
			DisplayName:   "Mixtral 8x7B",
			ContextWindow: 32768,
			Capabilities:  []string{"chat"},
		},
	}, nil
}

func (p *GroqProvider) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	startTime := time.Now()

	groqReq := map[string]interface{}{
		"model":    req.Model,
		"messages": req.Messages,
	}

	if req.Temperature != nil {
		groqReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		groqReq["max_tokens"] = *req.MaxTokens
	}

	body, err := json.Marshal(groqReq)
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
		return nil, fmt.Errorf("Groq API error: %s - %s", resp.Status, string(bodyBytes))
	}

	var groqResp struct {
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
		XGroq struct {
			Usage struct {
				QueueTime        float64 `json:"queue_time"`
				PromptTime       float64 `json:"prompt_time"`
				CompletionTime   float64 `json:"completion_time"`
				TotalTime        float64 `json:"total_time"`
				PromptTokensPS   float64 `json:"prompt_tokens_per_second"`
				CompletionTokensPS float64 `json:"completion_tokens_per_second"`
			} `json:"usage"`
		} `json:"x_groq"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&groqResp); err != nil {
		return nil, err
	}

	duration := time.Since(startTime)
	tokensPerSec := groqResp.XGroq.Usage.CompletionTokensPS
	if tokensPerSec == 0 && duration.Seconds() > 0 {
		tokensPerSec = float64(groqResp.Usage.CompletionTokens) / duration.Seconds()
	}

	response := &ChatResponse{
		ID:    groqResp.ID,
		Model: groqResp.Model,
		Message: Message{
			Role:    "assistant",
			Content: "",
		},
		Usage: UsageMetrics{
			InputTokens:  groqResp.Usage.PromptTokens,
			OutputTokens: groqResp.Usage.CompletionTokens,
			TotalTokens:  groqResp.Usage.TotalTokens,
		},
		Metrics: PerformanceMetrics{
			DurationMs:      duration.Milliseconds(),
			TokensPerSecond: tokensPerSec,
		},
	}

	if len(groqResp.Choices) > 0 {
		response.Message.Content = groqResp.Choices[0].Message.Content
	}

	return response, nil
}

func (p *GroqProvider) ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("streaming not implemented yet")
}

func (p *GroqProvider) ValidateCredentials(ctx context.Context) error {
	_, err := p.ListModels(ctx)
	return err
}

func (p *GroqProvider) GetPricing(modelName string) (*ModelPricing, error) {
	// Groq is free for now
	pricing := map[string]ModelPricing{
		"llama-3.3-70b-versatile": {
			InputPricePer1M:  0.0,
			OutputPricePer1M: 0.0,
			ContextWindow:    128000,
		},
		"mixtral-8x7b-32768": {
			InputPricePer1M:  0.0,
			OutputPricePer1M: 0.0,
			ContextWindow:    32768,
		},
	}

	if p, ok := pricing[modelName]; ok {
		return &p, nil
	}

	return nil, fmt.Errorf("pricing not found for model: %s", modelName)
}
