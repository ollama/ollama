package qwen2

import (
	"context"
	"github.com/ollama/ollama/model"
)

// Template is the prompt template for Qwen2.5
func (q *Qwen2) Template() string {
	return DEFAULT_TEMPLATE
}

// HandleEmptySuffix ensures that empty string suffixes work correctly for FIM
// by converting them to a space character, which is known to work.
func (q *Qwen2) HandleEmptySuffix(suffix string) string {
	if suffix == "" {
		return " " // Replace empty suffix with a space for FIM to work properly
	}
	return suffix
}

// Generate implements the Model interface
func (q *Qwen2) Generate(ctx context.Context, prompt, suffix, format string, options map[string]interface{}) (string, error) {
	// Handle empty suffix for FIM
	suffix = q.HandleEmptySuffix(suffix)

	// Use the model package's Generate function
	return model.Generate(ctx, q, prompt, suffix, format, options)
}
