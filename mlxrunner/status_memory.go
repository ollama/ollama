package mlxrunner

import (
	"context"
	"log/slog"
	"sync"
	"time"
)

const statusMemoryRefreshWait = 50 * time.Millisecond

type statusMemoryRefreshFunc func() (uint64, error)

// statusMemoryCache keeps health checks from depending synchronously on the
// serialized MLX worker while still refreshing memory telemetry opportunistically.
type statusMemoryCache struct {
	done    <-chan struct{}
	wait    time.Duration
	refresh statusMemoryRefreshFunc

	mu          sync.Mutex
	memory      uint64
	refreshedAt time.Time
	inFlight    chan struct{}
}

func newStatusMemoryCache(ctx context.Context, memory uint64, refreshedAt time.Time, wait time.Duration, refresh statusMemoryRefreshFunc) *statusMemoryCache {
	return &statusMemoryCache{
		done:        ctx.Done(),
		wait:        wait,
		refresh:     refresh,
		memory:      memory,
		refreshedAt: refreshedAt,
	}
}

func (c *statusMemoryCache) Memory() uint64 {
	done := c.startRefresh()
	if c.wait <= 0 {
		<-done
		memory, _ := c.snapshot()
		return memory
	}

	timer := time.NewTimer(c.wait)
	defer timer.Stop()

	select {
	case <-done:
	case <-timer.C:
		memory, refreshedAt := c.snapshot()
		if refreshedAt.IsZero() {
			slog.Debug("using cached MLX memory status before first refresh")
		} else {
			slog.Debug("using cached MLX memory status", "stale", time.Since(refreshedAt))
		}
		return memory
	case <-c.done:
	}

	memory, _ := c.snapshot()
	return memory
}

func (c *statusMemoryCache) startRefresh() chan struct{} {
	c.mu.Lock()
	if c.inFlight != nil {
		done := c.inFlight
		c.mu.Unlock()
		return done
	}

	refreshDone := make(chan struct{})
	c.inFlight = refreshDone
	refresh := c.refresh
	lifecycleDone := c.done
	c.mu.Unlock()

	go func() {
		memory, err := refresh()
		now := time.Now()

		c.mu.Lock()
		defer c.mu.Unlock()
		defer close(refreshDone)

		if err != nil {
			select {
			case <-lifecycleDone:
			default:
				slog.Debug("failed to refresh MLX memory status", "error", err)
			}
			c.inFlight = nil
			return
		}

		c.memory = memory
		c.refreshedAt = now
		c.inFlight = nil
	}()

	return refreshDone
}

func (c *statusMemoryCache) snapshot() (uint64, time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.memory, c.refreshedAt
}
