package cmd

import (
	"strings"
	"testing"
)

func TestIsLocalModel(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		expected  bool
	}{
		{
			name:      "local model without suffix",
			modelName: "llama3.2",
			expected:  true,
		},
		{
			name:      "local model with version",
			modelName: "qwen2.5:7b",
			expected:  true,
		},
		{
			name:      "cloud model",
			modelName: "gpt-oss:latest-cloud",
			expected:  false,
		},
		{
			name:      "cloud model with :cloud suffix",
			modelName: "gpt-oss:cloud",
			expected:  false,
		},
		{
			name:      "cloud model with version",
			modelName: "gpt-oss:20b-cloud",
			expected:  false,
		},
		{
			name:      "cloud model with version and :cloud suffix",
			modelName: "gpt-oss:20b:cloud",
			expected:  false,
		},
		{
			name:      "empty model name",
			modelName: "",
			expected:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isLocalModel(tt.modelName)
			if result != tt.expected {
				t.Errorf("isLocalModel(%q) = %v, expected %v", tt.modelName, result, tt.expected)
			}
		})
	}
}

func TestIsLocalServer(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		expected bool
	}{
		{
			name:     "empty host (default)",
			host:     "",
			expected: true,
		},
		{
			name:     "localhost",
			host:     "http://localhost:11434",
			expected: true,
		},
		{
			name:     "127.0.0.1",
			host:     "http://127.0.0.1:11434",
			expected: true,
		},
		{
			name:     "custom port on localhost",
			host:     "http://localhost:8080",
			expected: true, // localhost is always considered local
		},
		{
			name:     "remote host",
			host:     "http://ollama.example.com:11434",
			expected: true, // has :11434
		},
		{
			name:     "remote host different port",
			host:     "http://ollama.example.com:8080",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.host)
			result := isLocalServer()
			if result != tt.expected {
				t.Errorf("isLocalServer() with OLLAMA_HOST=%q = %v, expected %v", tt.host, result, tt.expected)
			}
		})
	}
}

func TestTruncateToolOutput(t *testing.T) {
	// Create outputs of different sizes
	localLimitOutput := make([]byte, 20000)   // > 4k tokens (16k chars)
	defaultLimitOutput := make([]byte, 50000) // > 10k tokens (40k chars)
	for i := range localLimitOutput {
		localLimitOutput[i] = 'a'
	}
	for i := range defaultLimitOutput {
		defaultLimitOutput[i] = 'b'
	}

	tests := []struct {
		name          string
		output        string
		modelName     string
		host          string
		shouldTrim    bool
		expectedLimit int
	}{
		{
			name:          "short output local model",
			output:        "hello world",
			modelName:     "llama3.2",
			host:          "",
			shouldTrim:    false,
			expectedLimit: localModelTokenLimit,
		},
		{
			name:          "long output local model - trimmed at 4k",
			output:        string(localLimitOutput),
			modelName:     "llama3.2",
			host:          "",
			shouldTrim:    true,
			expectedLimit: localModelTokenLimit,
		},
		{
			name:          "long output cloud model - uses 10k limit",
			output:        string(localLimitOutput), // 20k chars, under 10k token limit
			modelName:     "gpt-oss:latest-cloud",
			host:          "",
			shouldTrim:    false,
			expectedLimit: defaultTokenLimit,
		},
		{
			name:          "very long output cloud model - trimmed at 10k",
			output:        string(defaultLimitOutput),
			modelName:     "gpt-oss:latest-cloud",
			host:          "",
			shouldTrim:    true,
			expectedLimit: defaultTokenLimit,
		},
		{
			name:          "long output remote server - uses 10k limit",
			output:        string(localLimitOutput),
			modelName:     "llama3.2",
			host:          "http://remote.example.com:8080",
			shouldTrim:    false,
			expectedLimit: defaultTokenLimit,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.host)
			result := truncateToolOutput(tt.output, tt.modelName)

			if tt.shouldTrim {
				maxLen := tt.expectedLimit * charsPerToken
				if len(result) > maxLen+50 { // +50 for the truncation message
					t.Errorf("expected output to be truncated to ~%d chars, got %d", maxLen, len(result))
				}
				if result == tt.output {
					t.Error("expected output to be truncated but it wasn't")
				}
			} else {
				if result != tt.output {
					t.Error("expected output to not be truncated")
				}
			}
		})
	}
}

// Regression for https://github.com/ollama/ollama/issues/16648: the tool-call
// display must show the full command/query so the user can see exactly what is
// about to run. This is the only place a call is surfaced in yolo mode, where
// there is no approval prompt to inspect it first, so it must not be truncated.
func TestFormatToolCall(t *testing.T) {
	longCmd := "curl -sS -X POST https://api.example.com/v1/very/long/endpoint?token=abcdef0123456789 -H 'Content-Type: application/json' -d '{\"hello\":\"world\"}'"
	longQuery := strings.Repeat("how to write a very long search query that exceeds fifty characters ", 3)

	tests := []struct {
		name     string
		toolName string
		args     map[string]any
		want     string
	}{
		{
			name:     "bash command shown in full",
			toolName: "bash",
			args:     map[string]any{"command": longCmd},
			want:     longCmd,
		},
		{
			name:     "web_search query shown in full",
			toolName: "web_search",
			args:     map[string]any{"query": longQuery},
			want:     longQuery,
		},
		{
			name:     "unknown tool falls back to display name",
			toolName: "some_tool",
			args:     map[string]any{},
			want:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatToolCall(tt.toolName, tt.args)
			if tt.want != "" && !strings.Contains(got, tt.want) {
				t.Errorf("formatToolCall(%q) = %q, want it to contain the full value %q", tt.toolName, got, tt.want)
			}
			if strings.Contains(got, "...") {
				t.Errorf("formatToolCall(%q) = %q, want no truncation ellipsis", tt.toolName, got)
			}
		})
	}
}
