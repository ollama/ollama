//go:build goexperiment.synctest

package backoff

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"
)

func TestRetry(t *testing.T) {
	synctest.Run(func() {
		n := 0

		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		err := Retry(ctx, 100*time.Millisecond, func(err error) bool { return true }, func() error {
			n++
			if n > 5 {
				cancel()
			}
			return errors.New("keep going")
		})

		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}

		if n != 6 {
			t.Errorf("n = %d, want 6", n)
		}
	})
}

func TestRetrySuccess(t *testing.T) {
	synctest.Run(func() {
		n := 0
		err := Retry(t.Context(), 100*time.Millisecond, func(err error) bool { return true }, func() error {
			n++
			if n >= 3 {
				return nil // success
			}
			return errors.New("retry")
		})

		if err != nil {
			t.Errorf("err = %v, want nil", err)
		}
		if n != 3 {
			t.Errorf("n = %d, want 3", n)
		}
	})
}

func TestRetryNonRetryable(t *testing.T) {
	synctest.Run(func() {
		permanent := errors.New("permanent error")
		n := 0
		err := Retry(t.Context(), 100*time.Millisecond, func(err error) bool {
			return !errors.Is(err, permanent)
		}, func() error {
			n++
			if n >= 2 {
				return permanent
			}
			return errors.New("retry")
		})

		if !errors.Is(err, permanent) {
			t.Errorf("err = %v, want permanent", err)
		}
		if n != 2 {
			t.Errorf("n = %d, want 2", n)
		}
	})
}
