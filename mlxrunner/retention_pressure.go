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

// retentionObservation describes the single memory pool backing MLX arrays.
// The observer selects pressure and reclamation signals that releasing those
// arrays can relieve. Host paging, for example, is relevant to shared RAM but
// does not by itself request eviction of arrays in discrete VRAM.
type retentionObservation struct {
	sampledAt    time.Time
	pressure     discover.PressureLevel
	reclaimBytes int64
	limit        int64
	available    retentionBounds // After OS/driver safety margins, before workspace.
}

func (m retentionObservation) headroom() int64 {
	return min(m.available.physical, m.available.commit, m.available.device)
}

// Keep independent constraints separate until deciding a target. A minimum
// alone can hide worsening commitment or GPU budgets behind unchanged RAM.
type retentionBounds struct {
	// physical is additional capacity in the backing pool (shared RAM today).
	// commit and device are independent allocation constraints on that pool.
	// freeTarget is its signed margin above an OS immediate-free-memory target.
	// Optional constraints use MaxInt64 when absent; zero is a real limit.
	physical, commit, device, freeTarget int64
}

func (m retentionObservation) bounds(reserve int64) retentionBounds {
	b := m.available
	b.physical -= reserve
	if b.commit != math.MaxInt64 {
		b.commit -= reserve
	}
	if b.device != math.MaxInt64 {
		b.device -= reserve
	}
	return b
}

func (m retentionObservation) capacity(reserve int64) int64 {
	b := m.bounds(reserve)
	capacity := min(b.physical, b.commit, b.device)
	if b.freeTarget < 0 {
		capacity = min(capacity, b.freeTarget)
	}
	return capacity
}

func (b retentionBounds) worsened(previous retentionBounds) bool {
	return b.physical < previous.physical || b.commit < previous.commit || b.device < previous.device || b.freeTarget < previous.freeTarget
}

type retentionObserver interface {
	Read() (retentionObservation, error)
	Changes() <-chan struct{}
	Close()
}

type retentionPolicy struct {
	target       int64
	observed     retentionBounds // Raw constraints, without credit for our own releases.
	healthySince time.Time
	settleUntil  time.Time
	lastReclaim  time.Time
}

func (p *retentionPolicy) update(m retentionObservation, before, active, reusable, reserve int64, now time.Time) (int64, bool) {
	observed := m.bounds(reserve)
	// Released storage can satisfy this decision before OS counters catch up.
	// Only raw observations are comparable when checking subsequent pressure.
	creditReleasedMemory(&m, before, active, reusable)
	headroom := m.headroom()
	capacity := m.capacity(reserve)
	healthy := m.pressure == discover.PressureNormal && headroom > reserve && !m.sampledAt.IsZero() && m.available.freeTarget >= 0
	if !healthy {
		p.healthySince = time.Time{}
	} else if p.healthySince.IsZero() || m.sampledAt.Before(p.healthySince) {
		p.healthySince = m.sampledAt
	}
	grow := healthy && m.sampledAt.Sub(p.healthySince) >= time.Second && !now.Before(p.settleUntil)
	// The native limit already includes its own safety margin.
	target := max(1, min(m.limit-reserve, active+capacity))
	if m.pressure == discover.PressureCritical {
		target = 1
	} else if p.target != 0 {
		if now.Before(p.settleUntil) && !observed.worsened(p.observed) {
			// OS accounting can lag eviction. Unchanged or improving constraints
			// need no second response. If any constraint worsens, use its current
			// absolute limit: we cannot distinguish fresh pressure from lag.
			target = max(1, min(m.limit-reserve, p.target))
		} else if !grow {
			target = min(target, p.target)
		}
	} else if !healthy {
		target = max(1, min(target, active))
	} else {
		// Keep the existing aggregate allowance while learning workspace and
		// waiting for a second healthy observation.
		target = min(target, active+maxPagedOutBytes)
	}
	p.target, p.observed = target, observed
	return target, grow
}

type cudaRetention struct {
	memory retentionObserver
	policy retentionPolicy
}

func (c *cudaRetention) Changes() <-chan struct{} { return c.memory.Changes() }
func (c *cudaRetention) Close()                   { c.memory.Close() }

func (c *cudaRetention) trim(caches, idle []*prefixCache, workspace, working int64, admitting bool) {
	m, err := c.memory.Read()
	quantified := m.reclaimBytes > 0
	storageBefore := int64(mlx.ActiveMemory()) + int64(mlx.CacheMemory())
	if err == nil && (m.pressure >= discover.PressureElevated || m.headroom() < workspace+working) {
		// CUDA's pool can reserve memory even when MLX's buffer cache is empty.
		// Release it before deciding which reusable history must be discarded.
		mlx.ClearCache()
		released := max(0, storageBefore-int64(mlx.ActiveMemory())-int64(mlx.CacheMemory()))
		if released == 0 {
			// CUDA may return pool allocations that CacheMemory does not count.
			// Re-sample those; retain feedback already received in this cycle.
			feedback, pressure := m.reclaimBytes, m.pressure
			m, err = c.memory.Read()
			// Compression and paging estimates can describe the same pages.
			// Keep the strongest request within this maintenance pass.
			m.reclaimBytes = max(feedback, m.reclaimBytes)
			m.pressure = max(m.pressure, pressure)
			quantified = m.reclaimBytes > 0
		}
		m.reclaimBytes = max(0, m.reclaimBytes-released)
	}
	if err != nil {
		c.policy = retentionPolicy{}
		// Unknown headroom must not trigger allocation-heavy checkpoint merges.
		// Keep executing state, but discard optional history without copying.
		for _, c := range caches {
			c.limitBytes = math.MaxInt64
		}
		trimRetainedHistory(caches, idle, 1)
		mlx.ClearCache()
		slog.Debug("adaptive history observation failed", "error", err)
		return
	}
	before := int64(mlx.ActiveMemory())
	now := time.Now()
	if m.reclaimBytes > 0 {
		reclaimColdHistory(caches, m.reclaimBytes)
	} else if !quantified && m.pressure == discover.PressureElevated && m.headroom() >= workspace+working && m.available.freeTarget >= 0 && m.sampledAt.Sub(c.policy.lastReclaim) >= time.Second {
		// A bounded fallback for real pressure without a byte estimate. The
		// maintenance interval prevents admissions from accelerating eviction.
		if evictColdHistory(caches, idle) {
			mlx.ClearCache()
		}
		c.policy.lastReclaim = m.sampledAt
	}
	active, reusable := int64(mlx.ActiveMemory()), int64(mlx.CacheMemory())
	target, grow := c.policy.update(m, storageBefore, active, reusable, workspace+working, now)
	for _, c := range caches {
		// The global target owns eviction while observations are available.
		// Per-cache eviction may materialize copies when merging checkpoints.
		c.limitBytes = math.MaxInt64
	}
	trimRetainedHistory(caches, idle, target)
	after := int64(mlx.ActiveMemory())
	if after < before || storageBefore > after+int64(mlx.CacheMemory()) {
		c.policy.settleUntil = now.Add(2 * time.Second)
		c.policy.lastReclaim = m.sampledAt
	}
	slog.Log(context.Background(), logutil.LevelTrace, "MLX cache memory",
		"before_admission", admitting, "physical_headroom", m.available.physical,
		"commit_headroom", m.available.commit, "device_headroom", m.available.device,
		"pressure", m.pressure, "reclaim_bytes", m.reclaimBytes,
		"headroom", m.headroom(), "limit", m.limit, "workspace", workspace,
		"target", target, "active_before", before, "active_after", after, "growth", grow)
}

// Apply releases to the observation that preceded them. OS counters can lag
// on an immediate re-read; charging that shortfall twice would discard
// the active frontier unnecessarily. Reusable buffers can satisfy allocations
// but only returned storage relieves the OS's immediate-free-memory deficit.
func creditReleasedMemory(m *retentionObservation, before, active, reusable int64) {
	released := max(0, before-active-reusable)
	m.available.physical += min(max(reusable, before-active), math.MaxInt64-max(0, m.available.physical))
	m.available.freeTarget += min(released, math.MaxInt64-max(0, m.available.freeTarget))
}
