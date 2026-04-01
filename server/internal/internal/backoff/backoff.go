package backoff

import (
	"context"
	"iter"
	"math/rand/v2"
	"time"
)

func Loop(ctx context.Context, maxBackoff time.Duration) iter.Seq2[int, error] {
	var n int
	return func(yield func(int, error) bool) {
		var t *time.Timer
		for {
			if ctx.Err() != nil {
				yield(n, ctx.Err())
				return
			}

			if !yield(n, nil) {
				return
			}

			n++

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
			case <-t.C:
			}
		}
	}
}
