package mlxrunner

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestStatusMemoryCacheWaitsForFastRefresh(t *testing.T) {
	var calls atomic.Int32
	cache := newStatusMemoryCache(context.Background(), 7, time.Now().Add(-time.Minute), time.Second, func() (uint64, error) {
		calls.Add(1)
		return 42, nil
	})

	if got := cache.Memory(); got != 42 {
		t.Fatalf("got memory %d, want 42", got)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}
}

func TestStatusMemoryCacheSupportsBlockingWait(t *testing.T) {
	cache := newStatusMemoryCache(context.Background(), 7, time.Now().Add(-time.Minute), 0, func() (uint64, error) {
		return 42, nil
	})

	if got := cache.Memory(); got != 42 {
		t.Fatalf("got memory %d, want 42", got)
	}
}

func TestStatusMemoryCacheReturnsCachedValueAndRefreshesLater(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	started := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int32

	cache := newStatusMemoryCache(ctx, 7, time.Now().Add(-time.Minute), time.Millisecond, func() (uint64, error) {
		if calls.Add(1) == 1 {
			close(started)
		}
		select {
		case <-release:
			return 42, nil
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	})

	start := time.Now()
	if got := cache.Memory(); got != 7 {
		t.Fatalf("got memory %d, want cached value 7", got)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("cached memory lookup took too long: %s", elapsed)
	}

	waitForRefreshStart(t, started)
	close(release)
	waitForCachedMemory(t, cache, 42)

	if got := calls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}
}

func TestStatusMemoryCacheReturnsCachedValueBeforeFirstRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	started := make(chan struct{})
	release := make(chan struct{})
	cache := newStatusMemoryCache(ctx, 7, time.Time{}, time.Millisecond, func() (uint64, error) {
		close(started)
		select {
		case <-release:
			return 42, nil
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	})

	if got := cache.Memory(); got != 7 {
		t.Fatalf("got memory %d, want cached value 7", got)
	}

	waitForRefreshStart(t, started)
	close(release)
	waitForCachedMemory(t, cache, 42)
}

func TestStatusMemoryCacheKeepsCachedValueWhenRefreshFails(t *testing.T) {
	var calls atomic.Int32
	cache := newStatusMemoryCache(context.Background(), 7, time.Now().Add(-time.Minute), time.Second, func() (uint64, error) {
		calls.Add(1)
		return 0, errors.New("refresh failed")
	})

	if got := cache.Memory(); got != 7 {
		t.Fatalf("got memory %d, want cached value 7", got)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}
}

func TestStatusMemoryCacheReturnsCachedValueWhenContextDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	started := make(chan struct{})
	release := make(chan struct{})
	cache := newStatusMemoryCache(ctx, 7, time.Now().Add(-time.Minute), time.Second, func() (uint64, error) {
		close(started)
		<-release
		return 0, ctx.Err()
	})

	cancel()
	if got := cache.Memory(); got != 7 {
		t.Fatalf("got memory %d, want cached value 7", got)
	}

	waitForRefreshStart(t, started)
	close(release)
	waitForInflightRefresh(t, cache)
}

func TestStatusMemoryCacheAllowsRefreshAfterFailure(t *testing.T) {
	var calls atomic.Int32
	cache := newStatusMemoryCache(context.Background(), 7, time.Now().Add(-time.Minute), time.Second, func() (uint64, error) {
		if calls.Add(1) == 1 {
			return 0, errors.New("refresh failed")
		}
		return 42, nil
	})

	if got := cache.Memory(); got != 7 {
		t.Fatalf("got memory %d, want cached value 7", got)
	}
	if got := cache.Memory(); got != 42 {
		t.Fatalf("got memory %d after retry, want 42", got)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("refresh calls = %d, want 2", got)
	}
}

func TestStatusMemoryCacheAllowsOneInflightRefresh(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	started := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int32

	cache := newStatusMemoryCache(ctx, 11, time.Now().Add(-time.Minute), time.Millisecond, func() (uint64, error) {
		if calls.Add(1) == 1 {
			close(started)
		}
		select {
		case <-release:
			return 99, nil
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	})

	const goroutines = 8
	var wg sync.WaitGroup
	errCh := make(chan string, goroutines)
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if got := cache.Memory(); got != 11 {
				errCh <- "got non-cached memory value"
			}
		}()
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	waitForRefreshStart(t, started)
	if got := calls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}

	close(release)
	waitForCachedMemory(t, cache, 99)
}

func waitForRefreshStart(t *testing.T, started <-chan struct{}) {
	t.Helper()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for refresh to start")
	}
}

func waitForCachedMemory(t *testing.T, cache *statusMemoryCache, want uint64) {
	t.Helper()
	deadline := time.After(time.Second)
	for {
		got, _ := cache.snapshot()
		if got == want {
			return
		}

		select {
		case <-deadline:
			t.Fatalf("cached memory = %d, want %d", got, want)
		case <-time.After(time.Millisecond):
		}
	}
}

func waitForInflightRefresh(t *testing.T, cache *statusMemoryCache) {
	t.Helper()
	deadline := time.After(time.Second)
	for {
		cache.mu.Lock()
		inFlight := cache.inFlight
		cache.mu.Unlock()
		if inFlight == nil {
			return
		}

		select {
		case <-deadline:
			t.Fatal("timeout waiting for refresh to finish")
		case <-time.After(time.Millisecond):
		}
	}
}
