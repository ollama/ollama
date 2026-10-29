package mlxrunner

import (
	"testing"
	"time"
)

// driveDepthController runs the controller for n steps against a regime mirroring
// the real loop: next() picks a draft depth D; the regime reports the forward time
// at fused width D+1, fed to the cost model keyed by draft depth D, and how many of
// the D drafts are accepted as a prefix (fed to the controller's acceptance
// estimator). It returns the depth the controller selects after the run. draw(pos,
// step) is the per-position acceptance outcome; a draft of length D accepts the
// leading run of positions whose draws succeed.
func driveDepthController(c *depthController, n int, fwdMS func(width int) float64, draw func(pos, step int) bool) int {
	for step := range n {
		d := c.next()
		c.cost.observe(d, time.Duration(fwdMS(d+1)*float64(time.Millisecond)))
		accepted := 0
		for i := 1; i <= d; i++ {
			if draw(i, step) {
				accepted++
			} else {
				break
			}
		}
		c.acc.observe(d, accepted)
	}
	return c.selected()
}

// deterministicDraw turns a per-position acceptance rate into a reproducible,
// independent Bernoulli accept/reject per (pos, step). It uses a splitmix64-style
// avalanche so adjacent positions and steps decorrelate — a plain linear hash leaves
// the prefix acceptance correlated across positions and biases the recovered rates.
func deterministicDraw(acceptance func(pos int) float64) func(pos, step int) bool {
	return func(pos, step int) bool {
		x := uint64(step)*0x9E3779B97F4A7C15 + uint64(pos)*0xD1B54A32D192ED03
		x ^= x >> 30
		x *= 0xBF58476D1CE4E5B9
		x ^= x >> 27
		x *= 0x94D049BB133111EB
		x ^= x >> 31
		frac := float64(x%1_000_000) / 1_000_000.0
		return frac < acceptance(pos)
	}
}

// Per-position acceptance rates (probability draft position i is accepted given the
// prefix survived), defined directly as valid probabilities that decay with depth —
// the model property both quants share. bf16 and nvfp4 use the same acceptance curve;
// only their cost curves differ, which is what shifts the optimum. The expected
// accepted prefix at depth N is Σ_{k<=N} ∏_{i<=k} a_i, which decays as the product
// shrinks, matching the measured diminishing returns.
func decayAcceptance(pos int) float64 {
	rates := []float64{0, 0.90, 0.82, 0.72, 0.60, 0.48, 0.40, 0.34, 0.30, 0.27, 0.25, 0.23, 0.22, 0.21, 0.20, 0.20, 0.20}
	if pos < 1 || pos >= len(rates) {
		return 0.18
	}
	return rates[pos]
}

func bf16Acceptance(pos int) float64  { return decayAcceptance(pos) }
func nvfp4Acceptance(pos int) float64 { return decayAcceptance(pos) }

// bandwidthBoundCost models a near-flat forward (deep optimum); computeBoundCost a
// steep one (shallow optimum). width = 1 + drafts.
func bandwidthBoundCost(width int) float64 { return 210.0 + 2.6*float64(width) }
func computeBoundCost(width int) float64   { return 30.0 + 19.0*float64(width) }

func TestDepthControllerSelectsDeepWhenForwardFlat(t *testing.T) {
	c := newDepthController()
	selected := driveDepthController(c, 600, bandwidthBoundCost, deterministicDraw(bf16Acceptance))
	// A near-flat forward makes extra width nearly free, so committed/cost keeps
	// rising until acceptance saturates: the EV peak sits deep (~6 for this curve).
	if selected < 5 {
		t.Errorf("expected a deep selection on a near-flat-cost forward, got %d", selected)
	}
}

func TestDepthControllerSelectsShallowWhenForwardSteep(t *testing.T) {
	c := newDepthController()
	selected := driveDepthController(c, 600, computeBoundCost, deterministicDraw(nvfp4Acceptance))
	// A steep forward makes each extra validated token expensive, so committed/cost
	// peaks shallow.
	if selected > 4 {
		t.Errorf("expected a shallow selection on a steep-cost forward, got %d", selected)
	}
}

func TestDepthControllerSelectsZeroWhenDraftNotWorthIt(t *testing.T) {
	c := newDepthController()
	cost := c.cost
	// A steep forward (each extra column ~19ms over a 30ms base) paired with poor
	// acceptance: even the first draft is accepted under half the time. One draft
	// costs cost(1)/cost(0) = 49/30 = 1.63x the plain step but commits only
	// 1 + 0.4 = 1.4 tokens, so plain decode (depth 0) wins. The controller must be able
	// to stop speculating, not be floored at drafting one token.
	poor := func(int) float64 { return 0.4 }
	selected := driveDepthController(c, 600, computeBoundCost, deterministicDraw(poor))
	if selected != 0 {
		t.Errorf("expected depth 0 (plain decode) when no draft pays for itself, got %d", selected)
	}
	// The depth-0 (plain-decode) cost must actually have been measured, or depth 0 was
	// chosen on a phantom cost rather than a real comparison.
	if !cost.sampled(0) {
		t.Error("depth-0 cost was never sampled, so depth 0 was not a real EV comparison")
	}
}

func TestDepthControllerExploresBeforeCostReady(t *testing.T) {
	c := newDepthController()
	cost := c.cost
	// Before any cost data, warmup walks the floor upward from depth 0 to seed depths
	// from the shallowest end — never jumping deep. The first draft is 0 (depth-0,
	// plain-decode cost); once that depth is recorded the next is 1. Those are the two
	// depths the controller first compares at.
	first := c.next()
	if cost.ready() {
		t.Fatal("cost model should not be ready before any observation")
	}
	if first != 0 {
		t.Errorf("first warmup draft = %d, want 0 (depth-0 cost first, never deep)", first)
	}
	cost.observe(first, 24*time.Millisecond)
	second := c.next()
	if second != 1 {
		t.Errorf("second warmup draft = %d, want 1 (depth-1 cost next)", second)
	}
	// After two distinct depths are observed the model becomes usable.
	cost.observe(second, 31*time.Millisecond)
	if !cost.ready() {
		t.Error("cost model should be ready after two distinct depths")
	}
}

func TestDepthControllerClimbsOutwardWithinFrontier(t *testing.T) {
	// The bug this guards: on a near-flat-cost forward the inherited optimistic
	// acceptance (1 for unmeasured positions) makes deep depths look best, so an
	// unbounded argmax jumps straight to a deep depth and drafts a dozen tokens that are
	// almost all rejected. The frontier bound forbids drafting beyond one past the
	// deepest position with trusted data, so the depth can only climb outward as
	// fast as acceptance is actually measured. Assert that invariant holds at every
	// step, even with always-accept (the most deep-favoring case) and flat cost.
	c := newDepthController()
	cost := c.cost
	always := func(int) float64 { return 1.0 }
	draw := deterministicDraw(always)
	for step := range 400 {
		d := c.next()
		if d > c.frontier()+1 {
			t.Fatalf("step %d drafted depth %d, want <= frontier+1 = %d", step, d, c.frontier()+1)
		}
		cost.observe(d, time.Duration(bandwidthBoundCost(d+1)*float64(time.Millisecond)))
		accepted := 0
		for i := 1; i <= d; i++ {
			if draw(i, step) {
				accepted++
			} else {
				break
			}
		}
		c.acc.observe(d, accepted)
	}
}

func TestCostModelBoundsSingleSampleInfluence(t *testing.T) {
	m := newCostModel()
	for range 20 {
		m.observe(0, 16*time.Millisecond)
	}
	base := m.cost(0)
	// One stall-inflated sample may move the estimate by at most alpha times
	// the clamp fraction.
	m.observe(0, 49*time.Millisecond)
	if bound := base * (1 + costEWMAAlpha*costClampFraction) * 1.001; m.cost(0) > bound {
		t.Errorf("one outlier moved cost(0) from %.2fms to %.2fms, beyond the clamp bound %.2fms", base, m.cost(0), bound)
	}
	// A genuine cost change is rate-limited, not rejected: repeated samples at
	// the new level must converge to it.
	for range 60 {
		m.observe(0, 32*time.Millisecond)
	}
	if got := m.cost(0); got < 31 || got > 33 {
		t.Errorf("estimate did not converge to a persistent cost change: got %.2fms, want ~32ms", got)
	}
}

func TestDepthControllerStaysParkedThroughHostStall(t *testing.T) {
	c := newDepthController()
	cost := c.cost
	// A regime where plain decode wins (steep width cost, mediocre acceptance):
	// the controller settles parked at depth 0.
	steep := func(width int) float64 { return 16.0 + 15.0*float64(width-1) }
	mediocre := func(int) float64 { return 0.6 }
	if selected := driveDepthController(c, 600, steep, deterministicDraw(mediocre)); selected != 0 {
		t.Fatalf("expected the controller to park at depth 0, got %d", selected)
	}
	// A host stall between two parked rounds lands in a single depth-0 cost
	// sample at ~3x the real round time. Its influence must stay bounded so the
	// parked baseline remains credible and the controller does not start
	// drafting against a phantom plain-decode cost.
	cost.observe(0, 49*time.Millisecond)
	if selected := c.selected(); selected != 0 {
		t.Errorf("a single stall-inflated depth-0 sample un-parked the controller: selected %d with cost(0)=%.1fms", selected, cost.cost(0))
	}
}

// driveCountingProbes is driveDepthController with a count of the rounds that
// drafted (the parked regime's probes).
func driveCountingProbes(c *depthController, n int, fwdMS func(width int) float64, draw func(pos, step int) bool) int {
	probes := 0
	for step := range n {
		d := c.next()
		if d > 0 {
			probes++
		}
		c.cost.observe(d, time.Duration(fwdMS(d+1)*float64(time.Millisecond)))
		accepted := 0
		for i := 1; i <= d; i++ {
			if draw(i, step) {
				accepted++
			} else {
				break
			}
		}
		c.acc.observe(d, accepted)
	}
	return probes
}

func TestDepthControllerProbeBackoff(t *testing.T) {
	c := newDepthController()
	// A regime where plain decode wins: the controller parks and its probes
	// keep changing nothing, so the cadence must back off to the cap and the
	// probe count over a long stretch must follow the doubling series, not the
	// base cadence.
	steep := func(width int) float64 { return 16.0 + 15.0*float64(width-1) }
	mediocre := deterministicDraw(func(int) float64 { return 0.6 })
	driveDepthController(c, 600, steep, mediocre)
	if c.selected() != 0 {
		t.Fatalf("expected the controller to park at depth 0, got %d", c.selected())
	}
	probes := driveCountingProbes(c, 4096, steep, mediocre)
	if c.probeInterval != depthProbeIntervalMax {
		t.Errorf("probe interval = %d, want backed off to %d", c.probeInterval, depthProbeIntervalMax)
	}
	if probes > 10 {
		t.Errorf("%d probes in 4096 parked rounds, want the backed-off handful", probes)
	}

	// The cadence is the controller's own persistent state, so a fresh stretch
	// starts at the backed-off interval: a stretch shorter than it never probes.
	c.probeSince = 0
	if probes := driveCountingProbes(c, 400, steep, mediocre); probes != 0 {
		t.Errorf("%d probes within the first backed-off interval, want 0", probes)
	}

	// When drafting turns worthwhile — cheap wide forwards and high acceptance —
	// the probes eventually pay off, and the selection change must snap the
	// cadence back to the base interval.
	flat := func(width int) float64 { return 16.0 + 0.5*float64(width-1) }
	good := deterministicDraw(func(int) float64 { return 0.95 })
	if selected := driveDepthController(c, 8192, flat, good); selected == 0 {
		t.Fatal("controller never re-engaged after the regime turned draft-favorable")
	}
	if c.probeInterval != depthProbeInterval {
		t.Errorf("probe interval = %d after the selection changed, want reset to %d", c.probeInterval, depthProbeInterval)
	}
}

func TestDepthControllerTracksAcceptanceDrift(t *testing.T) {
	c := newDepthController()
	// Flat cost throughout, so depth is governed by acceptance alone. First phase:
	// deep acceptance is poor, so the optimum is shallow. Second phase: acceptance
	// stays high deep, so the optimum moves deeper. The EWMA acceptance rate must follow the
	// shift — a cumulative all-time mean would stay anchored to the first phase.
	shallowFavoring := func(pos int) float64 {
		if pos <= 2 {
			return 0.9
		}
		return 0.2 // deep rarely accepted
	}
	deepFavoring := func(int) float64 { return 0.95 } // deep readily accepted
	driveDepthController(c, 400, bandwidthBoundCost, deterministicDraw(shallowFavoring))
	shallow := c.selected()
	if shallow > 4 {
		t.Fatalf("expected shallow selection while deep acceptance is poor, got %d", shallow)
	}
	// The probe cadence backed off while phase-one probes changed nothing, so
	// recovery runs at backed-off probe spacing until the first selection
	// change snaps it back — the drive must span several backed-off intervals.
	driveDepthController(c, 4096, bandwidthBoundCost, deterministicDraw(deepFavoring))
	deep := c.selected()
	if deep <= shallow {
		t.Errorf("expected selection to deepen after acceptance rose (was %d, now %d)", shallow, deep)
	}
}
