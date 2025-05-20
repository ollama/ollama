package cerrors

import (
	"math/rand"
	"time"
)

// RetryPolicy and RetryConfig are the same thing - RetryPolicy is the older name
// that's being used in some places
type RetryPolicy = RetryConfig

// RetryConfig defines how to handle retries for temporary failures
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts
	MaxRetries int
	
	// BaseDelay is the initial delay duration before the first retry
	BaseDelay time.Duration
	
	// MaxDelay caps the maximum delay between retries
	MaxDelay time.Duration
	
	// Factor is the multiplier for each subsequent retry delay
	Factor float64
	
	// Jitter adds randomness to delays to prevent thundering herd problems
	Jitter float64
}

// NewDefaultRetryConfig creates a RetryConfig with sensible defaults
// - Exponential backoff starting at 100ms
// - Maximum delay of 30 seconds
// - Up to 10 retry attempts
// - 15% jitter to prevent thundering herd issues
func NewDefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries: 10,
		BaseDelay:  100 * time.Millisecond,
		MaxDelay:   30 * time.Second,
		Factor:     1.5,
		Jitter:     0.15,
	}
}

// NewDefaultRetryPolicy creates a RetryPolicy with sensible defaults
// Same as NewDefaultRetryConfig but returns the value instead of pointer
func NewDefaultRetryPolicy() RetryPolicy {
	return RetryConfig{
		MaxRetries: 10,
		BaseDelay:  100 * time.Millisecond,
		MaxDelay:   30 * time.Second,
		Factor:     1.5,
		Jitter:     0.15,
	}
}

// CalculateBackoff computes the next backoff duration with jitter
func (rc *RetryConfig) CalculateBackoff(attempt int) time.Duration {
	if attempt <= 0 {
		return rc.BaseDelay
	}
	
	// Calculate delay with exponential backoff
	delay := float64(rc.BaseDelay)
	for i := 0; i < attempt; i++ {
		delay *= rc.Factor
		if delay > float64(rc.MaxDelay) {
			delay = float64(rc.MaxDelay)
			break
		}
	}
	
	// Apply jitter if configured
	if rc.Jitter > 0 {
		jitter := rand.Float64() * rc.Jitter * delay
		delay = delay + jitter
	}
	
	return time.Duration(delay)
}

// ShouldRetry determines if another retry should be attempted based on retry count
func (rc *RetryConfig) ShouldRetry(attempt int) bool {
	return attempt < rc.MaxRetries
}

// WithMaxRetries returns a new RetryConfig with updated max retries
func (rc *RetryConfig) WithMaxRetries(maxRetries int) *RetryConfig {
	newConfig := *rc
	newConfig.MaxRetries = maxRetries
	return &newConfig
}

// WithBaseDelay returns a new RetryConfig with updated base delay
func (rc *RetryConfig) WithBaseDelay(baseDelay time.Duration) *RetryConfig {
	newConfig := *rc
	newConfig.BaseDelay = baseDelay
	return &newConfig
}

// WithMaxDelay returns a new RetryConfig with updated max delay
func (rc *RetryConfig) WithMaxDelay(maxDelay time.Duration) *RetryConfig {
	newConfig := *rc
	newConfig.MaxDelay = maxDelay
	return &newConfig
}