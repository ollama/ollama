package backoff

import (
	"context"
	"errors"
	"iter"
	"math/rand"
	"time"
)

// Errors
var (
	// ErrMaxAttempts is not used by backoff but is available for use by
	// callers that want to signal that a maximum number of retries has
	// been exceeded. This should eliminate the need for callers to invent
	// their own error.
	ErrMaxAttempts = errors.New("max retries exceeded")
)

// Upto implements a backoff strategy that yields nil errors until the
// context is canceled, the maxRetries is exceeded, or yield returns false.
//
// The backoff strategy is a simple exponential backoff with a maximum
// backoff of maxBackoff. The backoff is randomized between 0.5-1.5 times
// the current backoff, in order to prevent accidental "thundering herd"
// problems.
func Upto(ctx context.Context, maxBackoff time.Duration) iter.Seq2[int, error] {
	var n int
	return func(yield func(int, error) bool) {
		for {
			if ctx.Err() != nil {
				yield(n, ctx.Err())
				return
			}

			n++

			// n^2 backoff timer is a little smoother than the
			// common choice of 2^n.
			d := time.Duration(n*n) * 10 * time.Millisecond
			if d > maxBackoff {
				d = maxBackoff
			}
			// Randomize the delay between 0.5-1.5 x msec, in order
			// to prevent accidental "thundering herd" problems.
			d = time.Duration(float64(d) * (rand.Float64() + 0.5))
			t := time.NewTimer(d)
			select {
			case <-ctx.Done():
				t.Stop()
			case <-t.C:
				if !yield(n, nil) {
					return
				}
			}
		}
	}
}
