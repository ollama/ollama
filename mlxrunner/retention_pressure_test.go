package mlxrunner

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestSharedRetentionObservation(t *testing.T) {
	const gib = int64(1 << 30)
	for _, tc := range []struct {
		name string
		m    retentionObservation
		want int64
	}{
		{"shared host capacity", sharedRetentionObservation(discover.MemoryHeadroom{AvailableRAM: 16 * gib, RAMLimit: 64 * gib}, nil, 0), 16*gib - 64*gib/20},
		{"shared pools overlap", sharedRetentionObservation(discover.MemoryHeadroom{AvailableRAM: 16 * gib, AvailableCommit: new(24 * gib), RAMLimit: 64 * gib}, &mlx.MemoryBudget{Budget: 20 * gib, Usage: 8 * gib}, 0), 11 * gib},
		{"commit constrained", sharedRetentionObservation(discover.MemoryHeadroom{AvailableRAM: 16 * gib, AvailableCommit: new(2 * gib), RAMLimit: 64 * gib}, &mlx.MemoryBudget{Budget: 20 * gib}, 0), gib},
		{"over GPU budget", sharedRetentionObservation(discover.MemoryHeadroom{AvailableRAM: 16 * gib, AvailableCommit: new(24 * gib)}, &mlx.MemoryBudget{Budget: 20 * gib, Usage: 22 * gib}, 0), -3 * gib},
		{"zero budget", sharedRetentionObservation(discover.MemoryHeadroom{AvailableRAM: 16 * gib, AvailableCommit: new(24 * gib)}, &mlx.MemoryBudget{}, 0), -256 << 20},
		{"host reserve", sharedRetentionObservation(discover.MemoryHeadroom{AvailableRAM: gib, AvailableCommit: new(24 * gib), RAMLimit: 64 * gib}, &mlx.MemoryBudget{Budget: 20 * gib}, 0), gib - 64*gib/20},
		{"native free target owns reserve", sharedRetentionObservation(discover.MemoryHeadroom{AvailableRAM: gib, RAMLimit: 64 * gib, FreeMemoryAboveTarget: new(gib)}, nil, 0), gib},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.m.headroom(); got != tc.want {
				t.Fatalf("headroom = %d, want %d", got, tc.want)
			}
		})
	}
}

type recordedRetentionObserver struct {
	observation retentionObservation
	err         error
	following   []retentionObservation
}

func (o *recordedRetentionObserver) Read() (retentionObservation, error) {
	current := o.observation
	if len(o.following) > 0 {
		o.observation, o.following = o.following[0], o.following[1:]
	}
	return current, o.err
}
func (o *recordedRetentionObserver) Changes() <-chan struct{} { return nil }
func (o *recordedRetentionObserver) Close()                   {}

func TestPressurePolicyPreservesExecutingHistory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		observer := &recordedRetentionObserver{observation: sharedRetentionObservation(discover.MemoryHeadroom{Pressure: discover.PressureCritical}, nil, 0)}
		s := &cudaRetention{memory: observer}
		before := mlx.ActiveMemory()
		mlx.ResetPeakMemory()
		s.trim([]*prefixCache{c}, nil, 0, int64(c.liveBytes()), false)
		if mlx.PeakMemory() > before || c.limitBytes != math.MaxInt64 {
			t.Fatal("pressure eviction allocated or enabled copying per-cache merges")
		}
		if c.activePath[len(c.activePath)-1].endOffset != 12 || c.minCacheOffset() != 12 {
			t.Fatal("low-memory policy discarded executing history")
		}
		input := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		if session := c.beginScore(input, nil); len(session.remaining) != 0 {
			t.Fatal("surviving executing history cannot be reused")
		}
		values := c.caches[0].State()[0].Floats()
		for i, token := range input {
			if values[i*8] != float32(token) {
				t.Fatalf("token %d changed during pressure eviction", i)
			}
		}
		s.trim([]*prefixCache{c}, []*prefixCache{c}, 0, int64(c.liveBytes()), false)
		if len(c.root.children) != 0 || c.minCacheOffset() != 0 || mlx.CacheMemory() != 0 {
			t.Fatal("idle history or allocator cache survived sustained low memory")
		}
	})
}

func TestPressureObservationFailureBoundsGrowth(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		c.limitBytes = math.MaxInt64
		s := &cudaRetention{memory: &recordedRetentionObserver{err: errors.New("budget unavailable")}}
		before := mlx.ActiveMemory()
		mlx.ResetPeakMemory()
		s.trim([]*prefixCache{c}, nil, 0, 0, false)
		if c.limitBytes != math.MaxInt64 || mlx.CacheMemory() != 0 || mlx.PeakMemory() > before {
			t.Fatal("observation failure allocated or retained unused allocator buffers")
		}
		if c.minCacheOffset() != 12 {
			t.Fatal("observation failure discarded executing state")
		}
		s.trim([]*prefixCache{c}, []*prefixCache{c}, 0, 0, false)
		if len(c.root.children) != 0 || c.minCacheOffset() != 0 {
			t.Fatal("observation failure retained idle history")
		}
	})
}

func TestRetentionPressureRecovery(t *testing.T) {
	const gib = int64(1 << 30)
	now := time.Unix(100, 0)
	healthy := sharedRetentionObservation(discover.MemoryHeadroom{SampledAt: now, AvailableRAM: 16 * gib, AvailableCommit: new(20 * gib), Pressure: discover.PressureNormal}, &mlx.MemoryBudget{Budget: 20 * gib, Usage: 8 * gib}, 19*gib)
	var p retentionPolicy
	target, grow := p.update(healthy, 8*gib, 8*gib, 0, gib, now)
	if grow || target != 16*gib {
		t.Fatalf("initial observation: target=%d grow=%v", target, grow)
	}
	healthy.sampledAt = now.Add(100 * time.Millisecond)
	if _, grow := p.update(healthy, 8*gib, 8*gib, 0, gib, healthy.sampledAt); grow {
		t.Fatal("admissions bypassed recovery interval")
	}
	healthy.sampledAt = now.Add(time.Second)
	if _, grow := p.update(healthy, 8*gib, 8*gib, 0, gib, healthy.sampledAt); !grow {
		t.Fatal("sustained healthy memory did not permit growth")
	}
	low := healthy
	low.pressure, low.sampledAt = discover.PressureCritical, now.Add(2*time.Second)
	if target, grow := p.update(low, 8*gib, 8*gib, 0, gib, now.Add(2*time.Second)); target != 1 || grow {
		t.Fatal("low memory must reclaim optional history")
	}
	neutral := healthy
	neutral.pressure, neutral.sampledAt = discover.PressureUnknown, now.Add(3*time.Second)
	if target, grow := p.update(neutral, 8*gib, 8*gib, 0, gib, now.Add(3*time.Second)); target != 1 || grow {
		t.Fatal("intermediate memory state allowed growth")
	}
	healthy.sampledAt = now.Add(4 * time.Second)
	if _, grow := p.update(healthy, 8*gib, 8*gib, 0, gib, healthy.sampledAt); grow {
		t.Fatal("one recovery sample permitted growth")
	}
	healthy.sampledAt = now.Add(5 * time.Second)
	if _, grow := p.update(healthy, 8*gib, 8*gib, 0, gib, healthy.sampledAt); !grow {
		t.Fatal("recovery did not restore growth")
	}
}

func TestRetentionWaitsForReleaseAccounting(t *testing.T) {
	const gib = int64(1 << 30)
	now := time.Unix(100, 0)
	m := sharedRetentionObservation(discover.MemoryHeadroom{SampledAt: now, AvailableRAM: 20 * gib, AvailableCommit: new(20 * gib)}, &mlx.MemoryBudget{Budget: 20 * gib, Usage: 20 * gib}, 19*gib)
	var p retentionPolicy
	target, _ := p.update(m, 8*gib, 8*gib, 0, gib, now)
	if target != 6*gib {
		t.Fatalf("initial target = %d, want %d", target, 6*gib)
	}
	p.settleUntil = now.Add(2 * time.Second)
	if got, _ := p.update(m, target, target, 0, gib, now.Add(time.Second)); got != target {
		t.Fatalf("unchanged observation charged release twice: %d", got)
	}
	m.available.device -= gib
	if got, _ := p.update(m, target, target, 0, gib, now.Add(time.Second)); got != 3*gib {
		t.Fatalf("worsening pressure must use current absolute capacity: got %d, want %d", got, 3*gib)
	}
	if got, _ := p.update(m, 5*gib, 5*gib, 0, gib, now.Add(3*time.Second)); got != 2*gib {
		t.Fatal("fresh sustained pressure did not contract after settling")
	}
}

func TestRetentionIndependentConstraintsAfterRelease(t *testing.T) {
	const gib = int64(1 << 30)
	now := time.Unix(100, 0)
	for _, constraint := range []string{"commit", "device", "workspace"} {
		t.Run(constraint, func(t *testing.T) {
			m := sharedRetentionObservation(discover.MemoryHeadroom{
				SampledAt: now, RAMLimit: 20 * gib, AvailableRAM: 2 * gib,
				Pressure: discover.PressureElevated,
			}, nil, math.MaxInt64)
			reserve := gib
			switch constraint {
			case "commit":
				m.available.commit = 2 * gib
			case "device":
				m.available.device = 2 * gib
			case "workspace":
				m.available.commit = 2 * gib
			}
			p := retentionPolicy{target: 10 * gib}
			if target, _ := p.update(m, 7*gib, 5*gib, 0, reserve, now); target != 6*gib {
				t.Fatalf("initial target = %d, want %d", target, 6*gib)
			}
			p.settleUntil = now.Add(2 * time.Second)
			m.sampledAt = now.Add(time.Second)
			want := 5 * gib
			switch constraint {
			case "commit":
				m.available.commit = gib
			case "device":
				m.available.device = gib
			case "workspace":
				reserve += gib
				want = 4 * gib
			}
			if target, _ := p.update(m, 5*gib, 5*gib, 0, reserve, m.sampledAt); target != want {
				t.Fatalf("worsening %s hidden by unchanged RAM: target = %d, want %d", constraint, target, want)
			}
		})
	}
}

func TestRetentionBackingPoolCredit(t *testing.T) {
	const gib = int64(1 << 30)
	for _, tc := range []struct {
		name     string
		reusable int64
		wantFree int64
	}{
		{"returned to OS", 0, gib},
		{"still in allocator", 2 * gib, -gib},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := retentionObservation{available: retentionBounds{
				physical: -gib, commit: 0, device: gib, freeTarget: -gib,
			}}
			creditReleasedMemory(&m, 4*gib, 2*gib, tc.reusable)
			if m.available.physical != gib || m.available.freeTarget != tc.wantFree {
				t.Fatalf("release credited the wrong storage: %+v", m.available)
			}
			if m.available.commit != 0 || m.available.device != gib || m.headroom() != 0 {
				t.Fatalf("release concealed an independent constraint: %+v", m.available)
			}
		})
	}
}

func TestRetentionWithoutHostConstraints(t *testing.T) {
	const gib = int64(1 << 30)
	now := time.Unix(100, 0)
	// A backing-pool observation need not contain host RAM or host pressure.
	// The observer has already applied the safety margin appropriate to it.
	m := retentionObservation{
		sampledAt: now, pressure: discover.PressureNormal, limit: 8 * gib,
		available: retentionBounds{physical: 2 * gib, commit: math.MaxInt64, device: math.MaxInt64, freeTarget: math.MaxInt64},
	}
	var p retentionPolicy
	if target, _ := p.update(m, 3*gib, 2*gib, 0, gib, now); target != 4*gib {
		t.Fatalf("pool-only target = %d, want %d", target, 4*gib)
	}
	// Neither a decision nor its release credit may mutate the raw sample.
	if m.available.physical != 2*gib || p.observed.physical != gib || p.observed.freeTarget != math.MaxInt64 {
		t.Fatalf("release changed raw constraints: sample=%+v observed=%+v", m.available, p.observed)
	}
	p.settleUntil = now.Add(2 * time.Second)
	if target, _ := p.update(m, 2*gib, 2*gib, 0, gib, now.Add(time.Second)); target != 4*gib {
		t.Fatalf("disappearing credit became fresh pressure: target=%d", target)
	}
}

func TestRetentionCreditSaturates(t *testing.T) {
	for _, freeTarget := range []int64{math.MaxInt64 - 5, math.MaxInt64} {
		m := retentionObservation{available: retentionBounds{
			physical: math.MaxInt64 - 10, commit: math.MaxInt64,
			device: math.MaxInt64, freeTarget: freeTarget,
		}}
		creditReleasedMemory(&m, 20, 0, 0)
		if m.available.physical != math.MaxInt64 || m.available.freeTarget != math.MaxInt64 {
			t.Fatalf("release overflowed headroom: %+v", m.available)
		}
		b := m.bounds(100)
		if b.physical != math.MaxInt64-100 || b.commit != math.MaxInt64 || b.device != math.MaxInt64 || b.freeTarget != math.MaxInt64 {
			t.Fatalf("reserve changed an absent constraint: %+v", b)
		}
	}
}

func TestSharedRetentionObservationOwnsValues(t *testing.T) {
	const gib = int64(1 << 30)
	host := discover.MemoryHeadroom{AvailableRAM: 8 * gib, AvailableCommit: new(6 * gib), FreeMemoryAboveTarget: new(-gib)}
	device := &mlx.MemoryBudget{Budget: 20 * gib, Usage: 10 * gib}
	m := sharedRetentionObservation(host, device, 19*gib)
	*host.AvailableCommit, *host.FreeMemoryAboveTarget, device.Usage = 0, 0, 20*gib
	if m.available.physical != 8*gib || m.available.commit != 5*gib || m.available.device != 9*gib || m.available.freeTarget != -gib {
		t.Fatalf("source mutation changed a sampled constraint: %+v", m.available)
	}
}

func TestRetentionReleaseCreditExpiresWhenCountersCatchUp(t *testing.T) {
	const gib = int64(1 << 30)
	now := time.Unix(100, 0)
	m := sharedRetentionObservation(discover.MemoryHeadroom{
		SampledAt: now, AvailableRAM: 6 * gib, FreeMemoryAboveTarget: new(-gib),
		Pressure: discover.PressureElevated,
	}, nil, math.MaxInt64)
	p := retentionPolicy{target: 10 * gib}
	if got, _ := p.update(m, 6*gib, 5*gib, 0, 3*gib, now); got != 9*gib {
		t.Fatalf("target after release = %d, want %d", got, 9*gib)
	}
	p.settleUntil = now.Add(2 * time.Second)
	// Both counters now include the release. A later deficit must not reuse
	// the credit from that earlier decision, even inside the settling window.
	m.available.physical, m.available.freeTarget = 7*gib, 0
	m.sampledAt = now.Add(time.Millisecond)
	if got, _ := p.update(m, 5*gib, 5*gib, 0, 3*gib, m.sampledAt); got != 9*gib {
		t.Fatalf("improving counters changed target to %d", got)
	}
	m.available.physical, m.available.freeTarget = 6*gib, -gib
	m.sampledAt = now.Add(2 * time.Millisecond)
	if got, _ := p.update(m, 5*gib, 5*gib, 0, 3*gib, m.sampledAt); got != 4*gib {
		t.Fatalf("old release concealed a fresh deficit: target = %d, want %d", got, 4*gib)
	}
}

func TestRetentionReleaseCreditDoesNotBecomePressure(t *testing.T) {
	const gib = int64(1 << 30)
	now := time.Unix(100, 0)
	for _, tc := range []struct {
		name        string
		observation retentionObservation
		before      int64
		active      int64
		reserve     int64
		want        int64
	}{
		{
			name: "Mac free target catches up after release",
			observation: sharedRetentionObservation(discover.MemoryHeadroom{
				AvailableRAM: 6 * gib, FreeMemoryAboveTarget: new(-gib),
			}, nil, 0),
			before: 6 * gib, active: 5 * gib, reserve: 3 * gib, want: 9 * gib,
		},
		{
			name: "Linux available RAM catches up after release",
			observation: sharedRetentionObservation(discover.MemoryHeadroom{
				RAMLimit: 20 * gib, AvailableRAM: 2 * gib,
			}, nil, 0),
			before: 9 * gib, active: 8 * gib, reserve: gib, want: 9 * gib,
		},
		{
			name: "Windows commitment overlaps available RAM",
			observation: sharedRetentionObservation(discover.MemoryHeadroom{
				RAMLimit: 20 * gib, AvailableRAM: 2 * gib, AvailableCommit: new(3 * gib),
			}, nil, 0),
			before: 9 * gib, active: 8 * gib, reserve: gib, want: 9 * gib,
		},
		{
			name: "Windows GPU budget overlaps available RAM",
			observation: sharedRetentionObservation(discover.MemoryHeadroom{
				RAMLimit: 20 * gib, AvailableRAM: 2 * gib,
			}, &mlx.MemoryBudget{Budget: 20 * gib, Usage: 17 * gib}, 0),
			before: 9 * gib, active: 8 * gib, reserve: gib, want: 9 * gib,
		},
		{
			name: "RAM credit cannot cover exhausted commitment",
			observation: sharedRetentionObservation(discover.MemoryHeadroom{
				RAMLimit: 20 * gib, AvailableRAM: 2 * gib, AvailableCommit: new(gib),
			}, nil, 0),
			before: 9 * gib, active: 8 * gib, reserve: gib, want: 7 * gib,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := tc.observation
			raw.sampledAt, raw.pressure, raw.limit = now, discover.PressureElevated, math.MaxInt64
			policy := retentionPolicy{target: 10 * gib}
			if got, _ := policy.update(raw, tc.before, tc.active, 0, tc.reserve, now); got != tc.want {
				t.Fatalf("target after release = %d, want %d", got, tc.want)
			}
			policy.settleUntil = now.Add(2 * time.Second)
			for _, elapsed := range []time.Duration{time.Millisecond, time.Second} {
				raw.sampledAt = now.Add(elapsed)
				if got, _ := policy.update(raw, tc.active, tc.active, 0, tc.reserve, raw.sampledAt); got != tc.want {
					t.Fatalf("unchanged OS observation after %s: target = %d, want %d", elapsed, got, tc.want)
				}
			}
		})
	}
}

func TestQualitativePressureSurvivesPoolResample(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		mlx.ClearCache()
		now := time.Now()
		observation := sharedRetentionObservation(discover.MemoryHeadroom{SampledAt: now, AvailableRAM: 8 << 30, RAMLimit: 16 << 30, FreeMemoryAboveTarget: new(int64(1 << 30)), Pressure: discover.PressureElevated}, nil, math.MaxInt64)
		quiet := observation
		quiet.pressure = discover.PressureNormal
		observer := &recordedRetentionObserver{observation: observation, following: []retentionObservation{quiet}}
		s := &cudaRetention{memory: observer, policy: retentionPolicy{target: 8 << 30}}
		before := len(c.activePath)
		s.trim([]*prefixCache{c}, nil, 0, 0, false)
		if len(observer.following) != 0 || len(c.activePath) != before-1 {
			t.Fatalf("pressure was lost during the pool re-read: path=%d, before=%d", len(c.activePath), before)
		}
		if mlx.CacheMemory() != 0 {
			t.Fatal("qualitative eviction left the released memory in the allocator cache")
		}
		// Another admission at the same sample time must not discard another
		// checkpoint just because the qualitative state remains elevated.
		observer.observation = observation
		s.trim([]*prefixCache{c}, nil, 0, 0, true)
		if len(c.activePath) != before-1 {
			t.Fatal("admission repeated qualitative eviction without a fresh maintenance interval")
		}
	})
}

func TestPressureRereadDoesNotDoubleReclaim(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		mlx.ClearCache()
		checkpoint := c.activePath[1].snapshotBytes()
		if checkpoint == 0 {
			t.Fatal("fixture needs independently owned cold history")
		}
		observation := sharedRetentionObservation(discover.MemoryHeadroom{
			SampledAt: time.Now(), AvailableRAM: 8 << 30, RAMLimit: 16 << 30,
			FreeMemoryAboveTarget: new(int64(1 << 30)),
			ReclaimBytes:          checkpoint, Pressure: discover.PressureElevated,
		}, nil, math.MaxInt64)
		// A compression estimate and then a paging estimate can describe the
		// same pages during the allocator-clear re-read in one maintenance pass.
		observer := &recordedRetentionObserver{observation: observation, following: []retentionObservation{observation}}
		s := &cudaRetention{memory: observer, policy: retentionPolicy{target: 8 << 30}}
		before := len(c.activePath)
		s.trim([]*prefixCache{c}, nil, 0, 0, false)
		if len(observer.following) != 0 {
			t.Fatal("fixture did not exercise the allocator-clear re-read")
		}
		if got, want := len(c.activePath), before-1; got != want {
			t.Fatalf("retained path length = %d, want %d after one checkpoint of pressure", got, want)
		}
	})
}

func TestPressureAllocatorReleaseWaitsForOSAccounting(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		previous := mlx.SetCacheLimit(8 << 20)
		c := nativeRetentionHistory()
		defer func() {
			c.close()
			mlx.ClearCache()
			mlx.SetCacheLimit(previous)
		}()
		mlx.ClearCache()
		mlx.Scoped(func() { mlx.Eval(mlx.Zeros(mlx.DTypeFloat32, 1<<18)) })
		mlx.DefaultStream().Synchronize()
		pool := int64(mlx.CacheMemory())
		if pool == 0 {
			t.Fatal("fixture did not retain unused allocator storage")
		}
		observer := &recordedRetentionObserver{observation: sharedRetentionObservation(discover.MemoryHeadroom{
			SampledAt: time.Now(), AvailableRAM: 1 << 30, RAMLimit: 16 << 30,
			ReclaimBytes: pool, Pressure: discover.PressureElevated,
		}, nil, math.MaxInt64)}
		s := &cudaRetention{memory: observer, policy: retentionPolicy{target: 8 << 30}}
		before := len(c.activePath)
		s.trim([]*prefixCache{c}, nil, pool/2, 0, false)
		if mlx.CacheMemory() != 0 || len(c.activePath) != before {
			t.Fatal("unused allocator storage should cover pressure without evicting history")
		}
		if s.policy.settleUntil.IsZero() {
			t.Fatal("allocator-only release did not wait for OS accounting")
		}
		observer.observation.reclaimBytes = 0
		observer.observation.sampledAt = observer.observation.sampledAt.Add(time.Millisecond)
		s.trim([]*prefixCache{c}, nil, pool/2, 0, true)
		if len(c.activePath) != before {
			t.Fatal("unchanged OS counters discarded history after the allocator covered the shortfall")
		}
	})
}

func TestPressureFeedbackThenQualitativeReadDoesNotReclaimAgain(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		mlx.ClearCache()
		now := time.Now()
		observation := sharedRetentionObservation(discover.MemoryHeadroom{
			SampledAt: now, AvailableRAM: 8 << 30, RAMLimit: 16 << 30,
			ReclaimBytes: c.activePath[1].snapshotBytes(), Pressure: discover.PressureElevated,
		}, nil, math.MaxInt64)
		following := observation
		following.reclaimBytes = 0
		observer := &recordedRetentionObserver{observation: observation, following: []retentionObservation{following}}
		s := &cudaRetention{memory: observer, policy: retentionPolicy{target: 8 << 30}}
		s.trim([]*prefixCache{c}, nil, 0, 0, false)
		before := len(c.activePath)
		observer.observation.sampledAt = now.Add(time.Millisecond)
		s.trim([]*prefixCache{c}, nil, 0, 0, true)
		if got := len(c.activePath); got != before {
			t.Fatalf("qualitative follow-up repeated recent quantified reclamation: path = %d, want %d", got, before)
		}
		observer.observation.sampledAt = now.Add(time.Second)
		s.trim([]*prefixCache{c}, nil, 0, 0, false)
		if got := len(c.activePath); got != before-1 {
			t.Fatalf("sustained qualitative pressure no longer reclaims history: path = %d, want %d", got, before-1)
		}
	})
}
