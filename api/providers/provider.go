package providers

import (
	"context"
	"io"
)

// Provider interface - All API providers implement this
type Provider interface {
	GetName() string
	GetType() string
	ListModels(ctx context.Context) ([]Model, error)
	ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error)
	ChatCompletionStream(ctx context.Context, req ChatRequest) (io.ReadCloser, error)
	ValidateCredentials(ctx context.Context) error
	GetPricing(modelName string) (*ModelPricing, error)
}

// Model represents a model from any provider
type Model struct {
	ID            string   `json:"id"`
	Name          string   `json:"name"`
	DisplayName   string   `json:"display_name"`
	ContextWindow int      `json:"context_window"`
	Capabilities  []string `json:"capabilities"`
	Deprecated    bool     `json:"deprecated"`
}

// ChatRequest standardized request
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
}

// ChatResponse standardized response
type ChatResponse struct {
	ID      string             `json:"id"`
	Model   string             `json:"model"`
	Message Message            `json:"message"`
	Usage   UsageMetrics       `json:"usage"`
	Metrics PerformanceMetrics `json:"metrics"`
}

// Message structure
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// UsageMetrics token usage
type UsageMetrics struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// PerformanceMetrics performance tracking
type PerformanceMetrics struct {
	DurationMs       int64   `json:"duration_ms"`
	TokensPerSecond  float64 `json:"tokens_per_second"`
	TimeToFirstToken int64   `json:"time_to_first_token_ms,omitempty"`
}

// ModelPricing pricing information
type ModelPricing struct {
	InputPricePer1M  float64 `json:"input_price_per_1m"`
	OutputPricePer1M float64 `json:"output_price_per_1m"`
	ContextWindow    int     `json:"context_window"`
}
