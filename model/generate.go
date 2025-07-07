package model

import (
	"context"
	"strings"
)

// Process a prompt along with optional suffix for FIM and format type.
func Generate(ctx context.Context, m Model, prompt, suffix, format string, options map[string]interface{}) (string, error) {
	if prompt == "" {
		return "", nil
	}

	// Handle empty suffix for models that require special handling (like qwen2.5-coder)
	if strings.Contains(m.ModelName(), "qwen2.5-coder") && suffix == "" {
		suffix = " " // Convert empty suffix to space for FIM functionality
	}

	// Apply bias adapters
	if err := m.SetBiasAdapters(options); err != nil {
		return "", err
	}

	// ...existing code...
	return "", nil // placeholder, replace with actual implementation
}
