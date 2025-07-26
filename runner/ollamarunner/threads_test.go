package ollamarunner

import (
	"os"
	"runtime"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestOllamaRunnerThreadsEnvVar(t *testing.T) {
	tests := []struct {
		name        string
		envValue    string
		expected    uint
	}{
		{
			name:        "valid positive integer",
			envValue:    "8",
			expected:    8,
		},
		{
			name:        "valid positive integer large",
			envValue:    "32",
			expected:    32,
		},
		{
			name:        "zero threads",
			envValue:    "0",
			expected:    0, // envconfig.Uint returns 0 for invalid/zero values
		},
		{
			name:        "negative threads",
			envValue:    "-1",
			expected:    0, // envconfig.Uint returns default (0) for invalid values
		},
		{
			name:        "invalid string",
			envValue:    "abc",
			expected:    0, // envconfig.Uint returns default (0) for invalid values
		},
		{
			name:        "empty string",
			envValue:    "",
			expected:    0, // envconfig.Uint returns default (0) when env var not set
		},
		{
			name:        "float value",
			envValue:    "4.5",
			expected:    0, // envconfig.Uint returns default (0) for invalid values
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up environment variable
			if tt.envValue != "" {
				t.Setenv("OLLAMA_RUNNER_THREADS", tt.envValue)
			} else {
				os.Unsetenv("OLLAMA_RUNNER_THREADS")
			}

			// Test the environment variable parsing using envconfig
			threads := envconfig.RunnerThreads()

			if threads != tt.expected {
				t.Errorf("Expected threads=%d, got threads=%d", tt.expected, threads)
			}
		})
	}
}

func TestOllamaRunnerThreadsIntegration(t *testing.T) {
	// Test the full logic as it would be used in Execute function
	tests := []struct {
		name        string
		envValue    string
		expected    int
	}{
		{
			name:        "valid threads override",
			envValue:    "4",
			expected:    4,
		},
		{
			name:        "zero threads uses default",
			envValue:    "0",
			expected:    runtime.NumCPU(),
		},
		{
			name:        "invalid threads uses default",
			envValue:    "abc",
			expected:    runtime.NumCPU(),
		},
		{
			name:        "no env var uses default",
			envValue:    "",
			expected:    runtime.NumCPU(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up environment variable
			if tt.envValue != "" {
				t.Setenv("OLLAMA_RUNNER_THREADS", tt.envValue)
			} else {
				os.Unsetenv("OLLAMA_RUNNER_THREADS")
			}

			// Simulate the logic from Execute function
			threads := runtime.NumCPU() // default value from flag
			
			// Apply the same logic as in Execute function
			if runnerThreads := envconfig.RunnerThreads(); runnerThreads > 0 {
				threads = int(runnerThreads)
			}

			if threads != tt.expected {
				t.Errorf("Expected final threads=%d, got threads=%d", tt.expected, threads)
			}
		})
	}
}