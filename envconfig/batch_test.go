package envconfig

import (
	"os"
	"testing"
	"time"
)

func TestBatchEnabled(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected bool
	}{
		{"default", "", false},
		{"true", "true", true},
		{"false", "false", false},
		{"invalid", "invalid", true}, // Bool returns true for invalid values
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv("OLLAMA_BATCH_ENABLED", tt.envValue)
				defer os.Unsetenv("OLLAMA_BATCH_ENABLED")
			} else {
				os.Unsetenv("OLLAMA_BATCH_ENABLED")
			}

			result := BatchEnabled()
			if result != tt.expected {
				t.Errorf("BatchEnabled() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBatchTimeout(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected time.Duration
	}{
		{"default", "", 50 * time.Millisecond},
		{"valid_ms", "100ms", 100 * time.Millisecond},
		{"valid_s", "2s", 2 * time.Second},
		{"invalid", "invalid", 50 * time.Millisecond},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv("OLLAMA_BATCH_TIMEOUT", tt.envValue)
				defer os.Unsetenv("OLLAMA_BATCH_TIMEOUT")
			} else {
				os.Unsetenv("OLLAMA_BATCH_TIMEOUT")
			}

			result := BatchTimeout()
			if result != tt.expected {
				t.Errorf("BatchTimeout() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBatchSize(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected uint
	}{
		{"default", "", 8},
		{"valid", "16", 16},
		{"invalid", "invalid", 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv("OLLAMA_BATCH_SIZE", tt.envValue)
				defer os.Unsetenv("OLLAMA_BATCH_SIZE")
			} else {
				os.Unsetenv("OLLAMA_BATCH_SIZE")
			}

			result := BatchSize()
			if result != tt.expected {
				t.Errorf("BatchSize() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBatchMemoryFactor(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected float64
	}{
		{"default", "", 1.5},
		{"valid", "2.0", 2.0},
		{"invalid", "invalid", 1.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv("OLLAMA_BATCH_MEMORY_FACTOR", tt.envValue)
				defer os.Unsetenv("OLLAMA_BATCH_MEMORY_FACTOR")
			} else {
				os.Unsetenv("OLLAMA_BATCH_MEMORY_FACTOR")
			}

			result := BatchMemoryFactor()
			if result != tt.expected {
				t.Errorf("BatchMemoryFactor() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBatchMaxConcurrent(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected uint
	}{
		{"default", "", 2},
		{"valid", "4", 4},
		{"invalid", "invalid", 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv("OLLAMA_BATCH_MAX_CONCURRENT", tt.envValue)
				defer os.Unsetenv("OLLAMA_BATCH_MAX_CONCURRENT")
			} else {
				os.Unsetenv("OLLAMA_BATCH_MAX_CONCURRENT")
			}

			result := BatchMaxConcurrent()
			if result != tt.expected {
				t.Errorf("BatchMaxConcurrent() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBatchMinSize(t *testing.T) {
	tests := []struct {
		name     string
		envValue string
		expected uint
	}{
		{"default", "", 2},
		{"valid", "1", 1},
		{"invalid", "invalid", 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envValue != "" {
				os.Setenv("OLLAMA_BATCH_MIN_SIZE", tt.envValue)
				defer os.Unsetenv("OLLAMA_BATCH_MIN_SIZE")
			} else {
				os.Unsetenv("OLLAMA_BATCH_MIN_SIZE")
			}

			result := BatchMinSize()
			if result != tt.expected {
				t.Errorf("BatchMinSize() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// Test that the helper functions work correctly
func TestFloat64Helper(t *testing.T) {
	fn := Float64("TEST_FLOAT", 3.14)
	
	// Test default value
	os.Unsetenv("TEST_FLOAT")
	if result := fn(); result != 3.14 {
		t.Errorf("Float64 default = %v, want %v", result, 3.14)
	}
	
	// Test valid value
	os.Setenv("TEST_FLOAT", "2.5")
	defer os.Unsetenv("TEST_FLOAT")
	if result := fn(); result != 2.5 {
		t.Errorf("Float64 valid = %v, want %v", result, 2.5)
	}
}

func TestDurationHelper(t *testing.T) {
	fn := Duration("TEST_DURATION", time.Second)
	
	// Test default value
	os.Unsetenv("TEST_DURATION")
	if result := fn(); result != time.Second {
		t.Errorf("Duration default = %v, want %v", result, time.Second)
	}
	
	// Test valid value
	os.Setenv("TEST_DURATION", "2s")
	defer os.Unsetenv("TEST_DURATION")
	if result := fn(); result != 2*time.Second {
		t.Errorf("Duration valid = %v, want %v", result, 2*time.Second)
	}
}
