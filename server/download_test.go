package server

import (
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsHuggingFaceURL(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		expected bool
	}{
		{
			name:     "nil url",
			url:      "",
			expected: false,
		},
		{
			name:     "huggingface.co main domain",
			url:      "https://huggingface.co/some/model",
			expected: true,
		},
		{
			name:     "cdn-lfs.huggingface.co subdomain",
			url:      "https://cdn-lfs.huggingface.co/repos/abc/123",
			expected: true,
		},
		{
			name:     "cdn-lfs3.hf.co CDN domain",
			url:      "https://cdn-lfs3.hf.co/repos/abc/123",
			expected: true,
		},
		{
			name:     "hf.co shortlink domain",
			url:      "https://hf.co/model",
			expected: true,
		},
		{
			name:     "uppercase HuggingFace domain",
			url:      "https://HUGGINGFACE.CO/model",
			expected: true,
		},
		{
			name:     "mixed case HF domain",
			url:      "https://Cdn-Lfs.HF.Co/repos",
			expected: true,
		},
		{
			name:     "ollama registry",
			url:      "https://registry.ollama.ai/v2/library/llama3",
			expected: false,
		},
		{
			name:     "github.com",
			url:      "https://github.com/ollama/ollama",
			expected: false,
		},
		{
			name:     "fake huggingface domain",
			url:      "https://nothuggingface.co/model",
			expected: false,
		},
		{
			name:     "fake hf domain",
			url:      "https://nothf.co/model",
			expected: false,
		},
		{
			name:     "huggingface in path not host",
			url:      "https://example.com/huggingface.co/model",
			expected: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var u *url.URL
			if tc.url != "" {
				var err error
				u, err = url.Parse(tc.url)
				if err != nil {
					t.Fatalf("failed to parse URL: %v", err)
				}
			}
			got := isHuggingFaceURL(u)
			assert.Equal(t, tc.expected, got)
		})
	}
}

func TestGetNumDownloadParts(t *testing.T) {
	tests := []struct {
		name        string
		url         string
		envValue    string
		expected    int
		description string
	}{
		{
			name:        "nil url returns default",
			url:         "",
			envValue:    "",
			expected:    numDownloadParts,
			description: "nil URL should return standard concurrency",
		},
		{
			name:        "ollama registry returns default",
			url:         "https://registry.ollama.ai/v2/library/llama3",
			envValue:    "",
			expected:    numDownloadParts,
			description: "Ollama registry should use standard concurrency",
		},
		{
			name:        "huggingface returns reduced default",
			url:         "https://huggingface.co/model/repo",
			envValue:    "",
			expected:    numHFDownloadParts,
			description: "HuggingFace should use reduced concurrency",
		},
		{
			name:        "hf.co CDN returns reduced default",
			url:         "https://cdn-lfs3.hf.co/repos/abc/123",
			envValue:    "",
			expected:    numHFDownloadParts,
			description: "HuggingFace CDN should use reduced concurrency",
		},
		{
			name:        "huggingface with env override",
			url:         "https://huggingface.co/model/repo",
			envValue:    "2",
			expected:    2,
			description: "OLLAMA_HF_CONCURRENCY should override default",
		},
		{
			name:        "huggingface with higher env override",
			url:         "https://huggingface.co/model/repo",
			envValue:    "8",
			expected:    8,
			description: "OLLAMA_HF_CONCURRENCY can be set higher than default",
		},
		{
			name:        "huggingface with invalid env (non-numeric)",
			url:         "https://huggingface.co/model/repo",
			envValue:    "invalid",
			expected:    numHFDownloadParts,
			description: "Invalid OLLAMA_HF_CONCURRENCY should fall back to default",
		},
		{
			name:        "huggingface with invalid env (zero)",
			url:         "https://huggingface.co/model/repo",
			envValue:    "0",
			expected:    numHFDownloadParts,
			description: "Zero OLLAMA_HF_CONCURRENCY should fall back to default",
		},
		{
			name:        "huggingface with invalid env (negative)",
			url:         "https://huggingface.co/model/repo",
			envValue:    "-1",
			expected:    numHFDownloadParts,
			description: "Negative OLLAMA_HF_CONCURRENCY should fall back to default",
		},
		{
			name:        "non-huggingface ignores env",
			url:         "https://registry.ollama.ai/v2/library/llama3",
			envValue:    "2",
			expected:    numDownloadParts,
			description: "OLLAMA_HF_CONCURRENCY should not affect non-HF URLs",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Set or clear the environment variable
			if tc.envValue != "" {
				t.Setenv("OLLAMA_HF_CONCURRENCY", tc.envValue)
			}

			var u *url.URL
			if tc.url != "" {
				var err error
				u, err = url.Parse(tc.url)
				if err != nil {
					t.Fatalf("failed to parse URL: %v", err)
				}
			}

			got := getNumDownloadParts(u)
			assert.Equal(t, tc.expected, got, tc.description)
		})
	}
}
