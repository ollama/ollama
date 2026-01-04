//go:build goexperiment.synctest

package backoff

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"
)

func TestLoop(t *testing.T) {
	synctest.Run(func() {
		last := -1

		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		for n, err := range Loop(ctx, 100*time.Millisecond) {
			if !errors.Is(err, ctx.Err()) {
				t.Errorf("err = %v, want nil", err)
			}
			if err != nil {
				break
			}
			if n != last+1 {
				t.Errorf("n = %d, want %d", n, last+1)
			}
			last = n
			if n > 5 {
				cancel()
			}
		}

		if last != 6 {
			t.Errorf("last = %d, want 6", last)
		}
	})
}
