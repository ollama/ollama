package pricing

import (
	"fmt"
	"github.com/ollama/ollama/api/providers"
)

// Calculator calculates API costs
type Calculator struct {
	provider providers.Provider
}

// NewCalculator creates a new pricing calculator
func NewCalculator(provider providers.Provider) *Calculator {
	return &Calculator{provider: provider}
}

// CalculateCost calculates cost for given usage
func (c *Calculator) CalculateCost(modelName string, inputTokens, outputTokens int) (*Cost, error) {
	pricing, err := c.provider.GetPricing(modelName)
	if err != nil {
		return nil, err
	}

	inputCost := float64(inputTokens) / 1000000.0 * pricing.InputPricePer1M
	outputCost := float64(outputTokens) / 1000000.0 * pricing.OutputPricePer1M
	totalCost := inputCost + outputCost

	return &Cost{
		InputCostUSD:  inputCost,
		OutputCostUSD: outputCost,
		TotalCostUSD:  totalCost,
		InputTokens:   inputTokens,
		OutputTokens:  outputTokens,
		TotalTokens:   inputTokens + outputTokens,
	}, nil
}

// Cost represents calculated cost
type Cost struct {
	InputCostUSD  float64 `json:"input_cost_usd"`
	OutputCostUSD float64 `json:"output_cost_usd"`
	TotalCostUSD  float64 `json:"total_cost_usd"`
	InputTokens   int     `json:"input_tokens"`
	OutputTokens  int     `json:"output_tokens"`
	TotalTokens   int     `json:"total_tokens"`
}

// FormatCost formats cost for display
func FormatCost(cost *Cost) string {
	if cost.TotalCostUSD < 0.01 {
		return fmt.Sprintf("$%.4f", cost.TotalCostUSD)
	}
	return fmt.Sprintf("$%.2f", cost.TotalCostUSD)
}
