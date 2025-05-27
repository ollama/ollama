package envconfig

import (
	"log/slog"
	"strconv"
	"time"
)

func Float64(key string, defaultValue float64) func() float64 {
	return func() float64 {
		if s := Var(key); s != "" {
			if f, err := strconv.ParseFloat(s, 64); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return f
			}
		}
		return defaultValue
	}
}

func Duration(key string, defaultValue time.Duration) func() time.Duration {
	return func() time.Duration {
		if s := Var(key); s != "" {
			if d, err := time.ParseDuration(s); err != nil {
				slog.Warn("invalid environment variable, using default", "key", key, "value", s, "default", defaultValue)
			} else {
				return d
			}
		}
		return defaultValue
	}
}

var (
	// BatchEnabled controls whether batch processing is enabled
	BatchEnabled = Bool("OLLAMA_BATCH_ENABLED")
	
	// BatchTimeout sets the maximum time to wait for accumulating requests into a batch
	BatchTimeout = Duration("OLLAMA_BATCH_TIMEOUT", 500*time.Millisecond)
	
	// BatchSize sets the maximum number of requests to process in a single batch
	BatchSize = Uint("OLLAMA_BATCH_SIZE", 8)
	
	// BatchMemoryFactor sets the memory multiplier for batch processing overhead
	BatchMemoryFactor = Float64("OLLAMA_BATCH_MEMORY_FACTOR", 1.5)
	
	// BatchMaxConcurrent sets the maximum number of concurrent batches to process
	BatchMaxConcurrent = Uint("OLLAMA_BATCH_MAX_CONCURRENT", 2)
	
	// BatchMinSize sets the minimum number of requests required to form a batch
	BatchMinSize = Uint("OLLAMA_BATCH_MIN_SIZE", 2)
)
