//go:build goexperiment.synctest

package backoff

import (
	"errors"
	"testing"
	"testing/synctest"
	"time"
)

var errRetry = errors.New("retry")

func TestRetryAllocs(t *testing.T) {
	for i := range 3 {
		got := testing.AllocsPerRun(1000, func() {
			tick := 0
			Retry(t.Context(), 1, func(err error) bool { return true }, func() error {
				tick++
				if tick >= i {
					return nil
				}
				return errRetry
			})
		})
		want := float64(0)
		if i > 0 {
			want = 3 // due to time.NewTimer
		}
		if got > want {
			t.Errorf("[%d ticks]: allocs = %v, want <= %v", i, got, want)
		}
	}
}

func BenchmarkRetry(b *testing.B) {
	ctx := b.Context()
	synctest.Run(func() {
		n := 0
		Retry(ctx, 100*time.Millisecond, func(err error) bool { return true }, func() error {
			n++
			if n == b.N {
				return nil
			}
			return errRetry
		})
	})
}
