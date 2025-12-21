package backoff

import (
	"context"
	"math/rand/v2"
	"time"
)

// Retry calls fn repeatedly with exponential backoff until it returns nil,
// a non-retryable error (shouldRetry returns false), or the context is cancelled.
// The shouldRetry function determines if an error is retryable.
// Returns the last error encountered, or nil if fn succeeded.
func Retry(ctx context.Context, maxBackoff time.Duration, shouldRetry func(error) bool, fn func() error) error {
	var t *time.Timer
	for n := 0; ; n++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		err := fn()
		if err == nil {
			return nil
		}
		if !shouldRetry(err) {
			return err
		}

		// n^2 backoff timer is a little smoother than the
		// common choice of 2^n.
		d := min(time.Duration(n*n)*10*time.Millisecond, maxBackoff)
		// Randomize the delay between 0.5-1.5 x msec, in order
		// to prevent accidental "thundering herd" problems.
		d = time.Duration(float64(d) * (rand.Float64() + 0.5))

		if t == nil {
			t = time.NewTimer(d)
		} else {
			t.Reset(d)
		}
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
}
