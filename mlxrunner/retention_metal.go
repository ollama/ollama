package mlxrunner

import (
	"context"
	"log/slog"
	"math"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/mlx"
)

type metalRetention struct {
	memory   *discover.MemoryMonitor
	parallel int
}

func (m *metalRetention) Changes() <-chan struct{} {
	if m.memory == nil {
		return nil
	}
	return m.memory.Changes()
}

func (m *metalRetention) Close() {
	if m.memory != nil {
		m.memory.Close()
	}
}

func (m *metalRetention) trim(caches, idle []*prefixCache, workspace, working int64, admitting bool) {
	var retained int64
	for _, c := range caches {
		retained += c.pagedOutBytes
	}
	var observation discover.MemoryHeadroom
	var err error
	if m.memory == nil {
		m.memory, err = discover.NewMemoryMonitor()
	}
	if err == nil {
		observation, err = m.memory.Read()
	}
	if err != nil {
		for _, c := range caches {
			c.limitBytes = maxPagedOutBytes / int64(max(1, m.parallel))
			c.enforceEvictionPolicy()
		}
		return
	}
	for _, c := range caches {
		c.limitBytes = math.MaxInt64
	}
	growth := observation.ReclaimBytes
	pressure := growth
	pool := int64(mlx.CacheMemory())
	storageBefore := int64(mlx.ActiveMemory()) + pool
	if (growth > 0 || (observation.FreeMemoryAboveTarget != nil && *observation.FreeMemoryAboveTarget < 0)) && pool > 0 {
		mlx.ClearCache()
		pressure = max(0, pressure-pool)
	}
	// Rising host compression asks each runner to yield cold owned history.
	// Other runners may also respond; this is bounded feedback, not a prediction
	// of which process caused pressure or a guarantee against an initial burst.
	reclaimed := reclaimColdHistory(caches, pressure)
	reserve := workspace + working
	available, reusable := observation.AvailableRAM, int64(mlx.CacheMemory())
	before := mlx.ActiveMemory()
	target := physicalHistoryLimit(observation, storageBefore, int64(before), reusable, reserve)
	trimRetainedHistory(caches, idle, target)
	slog.Log(context.Background(), logutil.LevelTrace, "MLX cache memory", "before_admission", admitting, "available", available, "immediate", *observation.FreeMemoryAboveTarget, "compression_growth", growth, "compression_reclaimed", reclaimed, "reusable", reusable, "workspace", workspace, "target", target, "active_before", before, "active_after", mlx.ActiveMemory(), "saved_before", retained, "working_copy", working)
}

// physicalHistoryLimit credits released native allocations as estimated headroom
// against the same host observation. Re-reading Mach counters can return a cached sample
// and charge a physical shortfall twice, unnecessarily discarding the frontier.
func physicalHistoryLimit(memory discover.MemoryHeadroom, before, active, reusable, reserve int64) int64 {
	released := max(0, before-active-reusable)
	target := max(1, active+memory.AvailableRAM+released+reusable-reserve)
	if memory.FreeMemoryAboveTarget != nil {
		if immediate := *memory.FreeMemoryAboveTarget + released; immediate < 0 {
			target = min(target, max(1, active+immediate))
		}
	}
	return target
}
