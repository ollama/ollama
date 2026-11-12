package mlxrunner

import (
	"context"
	"log/slog"
	"math"
	"time"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/mlx"
)

// Check memory at most once per interval during inference or idle time.
// Admission requests an immediate check. Reclamation waits for GPU completion;
// elapsed time never grants permission to free in-flight storage.
const snapshotCheckInterval = time.Second

// Keep reclamation local to this runner. Revisit cross-runner pruning after
// disk paging, with headroom for each model's active KV and workspace;
// fitting both sets of weights alone need not improve throughput.
func (s *batchScheduler) trimSnapshots(incoming queuedRequest) {
	if !s.hasPrefixCache() || !mlx.MetalIsAvailable() {
		return
	}
	if s.snapshotTimer == nil {
		s.snapshotTimer = time.NewTimer(snapshotCheckInterval)
	} else {
		s.snapshotTimer.Reset(snapshotCheckInterval)
	}
	var caches, idle []*prefixCache
	var working, next int64
	for _, slot := range s.slots {
		if slot == nil {
			continue
		}
		for _, c := range []*prefixCache{slot.cache, slot.scoreCache} {
			if c == nil {
				continue
			}
			caches = append(caches, c)
			live := int64(c.liveBytes())
			if c == slot.activeCache() {
				working += live
			} else {
				idle = append(idle, c)
				next = max(next, live)
			}
		}
	}
	var observation discover.MemoryHeadroom
	var err error
	if s.memory == nil {
		s.memory, err = discover.NewMemoryMonitor()
	}
	if err == nil {
		observation, err = s.memory.Read()
	}
	if err != nil {
		s.r.snapshotLimitBytes = maxPagedOutBytes / int64(max(1, s.r.parallel))
		overLimit := false
		for _, c := range caches {
			c.limitBytes = s.r.snapshotLimitBytes
			overLimit = overLimit || c.pagedOutBytes > c.limitBytes
		}
		if overLimit {
			mlx.DefaultStream().Synchronize()
			preferred := s.matchingHistory(incoming)
			for _, c := range caches {
				for c.pagedOutBytes > c.limitBytes && evictColdHistory([]*prefixCache{c}, nil, preferred) {
				}
			}
			mlx.ClearCache()
		}
		return
	}
	s.r.snapshotLimitBytes = math.MaxInt64
	for _, c := range caches {
		c.limitBytes = math.MaxInt64
	}
	// Allow one copy per active row, and one idle row which may be admitted.
	// Restoring history or writing a shared buffer can require a fresh copy.
	trimHistory(caches, idle, s.matchingHistory(incoming), observation, s.snapshotWorkspace+working+next)
}

// Prefer the arriving request's matching paths, including peer-slot donors.
// This is an eviction preference, not a reservation: severe pressure can still
// discard them. Longer matches take precedence over a shared system prefix.
// Match media identities and draft lookahead just as lookup does.
func (s *batchScheduler) matchingHistory(incoming queuedRequest) map[*trieNode]int {
	var tokens []int32
	var items []mediaItem
	score := false
	switch req := incoming.(type) {
	case request:
		tokens, items = req.Tokens, req.MediaItems
	case *evaluationRequest:
		tokens, items, score = req.tokens, req.items, true
	default:
		return nil
	}
	keys := effectiveKeyTokens(tokens, items)
	preferred := make(map[*trieNode]int)
	for _, slot := range s.slots {
		if slot == nil {
			continue
		}
		c := slot.cache
		if score {
			c = slot.scoreCache
		}
		if c == nil || c.root == nil {
			continue
		}
		path, matched := findBestMatch(c.root, c.key(keys))
		for _, n := range path {
			preferred[n] = matched
		}
	}
	return preferred
}

func trimHistory(caches, idle []*prefixCache, preferred map[*trieNode]int, memory discover.MemoryHeadroom, reserve int64) {
	active, pool := int64(mlx.ActiveMemory()), int64(mlx.CacheMemory())
	storageBefore := active + pool
	target := physicalHistoryLimit(memory, storageBefore, active, pool, reserve)
	shortfall := memory.FreeMemoryAboveTarget != nil && *memory.FreeMemoryAboveTarget < 0
	if memory.ReclaimBytes == 0 && !shortfall && active+pendingHistoryBytes(caches) <= target {
		return
	}
	// The scheduler serializes host operations, but GPU work can still be in
	// flight. Finish submitted work before dropping storage or reading allocator
	// counters for eviction. Lazy, unsubmitted graphs retain their own array refs.
	mlx.DefaultStream().Synchronize()
	pressure := memory.ReclaimBytes
	pool = int64(mlx.CacheMemory())
	if (pressure > 0 || shortfall) && pool > 0 {
		mlx.ClearCache()
		pressure = max(0, pressure-pool)
	}
	// Rising host compression asks each runner to yield cold owned history.
	// This is bounded feedback, not a prediction of which process caused pressure.
	reclaimed := reclaimColdHistory(caches, idle, pressure, preferred)
	active, pool = int64(mlx.ActiveMemory()), int64(mlx.CacheMemory())
	target = physicalHistoryLimit(memory, storageBefore, active, pool, reserve)
	trimRetainedHistory(caches, idle, target, preferred)
	slog.Log(context.Background(), logutil.LevelTrace, "MLX cache memory", "available", memory.AvailableRAM, "compression_growth", memory.ReclaimBytes, "compression_reclaimed", reclaimed, "reusable", pool, "reserve", reserve, "target", target, "active_before", active, "active_after", mlx.ActiveMemory())
}

// physicalHistoryLimit accounts for native storage changes against the same
// host observation, including allocations completed while synchronizing.
// Re-reading Mach counters can return a cached sample
// and charge a physical shortfall twice, unnecessarily discarding the frontier.
func physicalHistoryLimit(memory discover.MemoryHeadroom, before, active, reusable, reserve int64) int64 {
	released := before - active - reusable
	target := max(1, before+memory.AvailableRAM-reserve)
	if memory.FreeMemoryAboveTarget != nil {
		if immediate := *memory.FreeMemoryAboveTarget + released; immediate < 0 {
			target = min(target, max(1, active+immediate))
		}
	}
	return target
}
