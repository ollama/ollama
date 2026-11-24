package mlxrunner

import (
	"fmt"
	"slices"
	"strings"
	"time"
)

// depthProbeInterval is the base cadence at which the controller drafts one past its
// selection to refresh the next position up; without it a shallow controller never
// notices a deeper draft becoming worthwhile.
const depthProbeInterval = 4

// depthProbeIntervalMax caps the probe backoff. Probes that keep changing nothing
// double the interval up to this cap, trading slower re-engagement — worst case
// plain decode speed, never below — for fewer probes.
const depthProbeIntervalMax = 512

// depthController drafts argmax_N EV(N) where EV(N) = committed(N) / cost(N), over
// depths from plain decode up to one past the measured acceptance frontier,
// bounded by the drafter's capacity. A probe can observe a complete block.
//
// Drafters carry the learned state across requests. Adaptive block selectors
// also restart the probe schedule with a full block for each request.
type depthController struct {
	cost   *costModel
	acc    *acceptanceModel
	probed bool // the previous round drafted a probe
	block  bool // sample a complete draft block when resuming from plain decode

	// plainTokens is the number of plain-row tokens per drafting row in a
	// mixed batch. Include them when comparing whole-batch throughput.
	plainTokens float64

	// drafterLimit is the deepest draft this model's drafter can produce, 0 when
	// nothing bounds it. A depth the drafter never produces is never measured,
	// and an unmeasured depth would always look best, pinning the search there.
	drafterLimit int

	// scheduled is the draft depth next() chose for the upcoming round, carried
	// across requests so a new request's first round consumes it instead of
	// recomputing at the boundary. Its zero value is the depth-0 warmup start.
	scheduled int

	// Probe cadence, persisted so a backed-off request need not restart at the base.
	probeInterval int // rounds between probes; backs off while probes change nothing
	probeSince    int // rounds since the cadence was last calibrated
	lastSelected  int // the selection the cadence was calibrated against
}

func newDepthController() *depthController {
	return &depthController{cost: newCostModel(), acc: newAcceptanceModel(acceptanceMinSamples), probeInterval: depthProbeInterval}
}

// newBlockDepthController is the controller of a drafter whose block selector
// adapts its depth: each request is a forBlock session sharing its costs and
// pooled acceptance.
func newBlockDepthController() *depthController {
	c := newDepthController()
	c.block = true
	c.acc = newBlockAcceptanceModel()
	return c
}

// forBlock starts a request's block session. Costs and acceptance carry across
// requests, and the acceptance EWMA follows a new text within a few rounds. The
// probe schedule restarts with a full block, which gives evidence for every
// position in the new text.
func (c *depthController) forBlock() *depthController {
	return &depthController{
		cost:          c.cost,
		acc:           c.acc,
		block:         true,
		drafterLimit:  c.drafterLimit,
		scheduled:     c.drafterLimit,
		probeInterval: depthProbeInterval,
	}
}

func (c *depthController) frontier() int { return c.acc.frontier() }

// limit bounds the search at one beyond the measured acceptance frontier.
func (c *depthController) limit() int {
	limit := c.frontier() + 1
	if c.drafterLimit > 0 {
		limit = min(limit, c.drafterLimit)
	}
	return limit
}

// next selects the best measured throughput and periodically probes a deeper
// draft. A parked block selector can probe all positions with one draft forward.
func (c *depthController) next() int {
	return c.nextFor(c.limit(), c.expectedCommitted)
}

func (c *depthController) expectedCommitted(n int) float64 {
	return c.acc.expectedCommitted(n) + c.plainTokens
}

// nextFor also serves a batch, where the expected output sums the contribution
// of each conversation and the measured cost covers the whole forward.
func (c *depthController) nextFor(limit int, expected func(int) float64) (depth int) {
	defer func() { c.scheduled = depth }()
	sel := c.selected(limit, expected)
	if sel != c.lastSelected {
		c.probeInterval = depthProbeInterval
		c.probeSince = 0
		c.lastSelected = sel
	} else if c.probed {
		maxInterval := depthProbeIntervalMax
		if c.block {
			// A transition from reasoning to predictable code can make a
			// whole block profitable before the first position has been relearnt.
			maxInterval = 2 * depthProbeInterval
		}
		c.probeInterval = min(c.probeInterval*2, maxInterval)
	}
	c.probed = false

	c.probeSince++
	if c.probeSince >= c.probeInterval {
		c.probeSince = 0
		probe := min(sel+1, limit)
		if c.block && sel == 0 {
			probe = c.drafterLimit
		}
		if probe != sel {
			c.probed = true
			return probe
		}
	}
	// Seed a clean cost sample for every depth in [0, frontier+1] before judging by
	// EV. Cost is only recorded when a round's draft depth matches the prior round's
	// (the same-depth gate dropping batch-shape transitions), so a depth never dwelt
	// at has no clean sample; draft the shallowest such depth to dwell there. Without
	// this the controller stays at the one depth it can sit at without a transition
	// (depth 0) and never learns that drafting pays on a deep-optimum model.
	if seed := c.costSeedDepth(limit); seed >= 0 {
		return seed
	}
	return sel
}

// costSeedDepth is the shallowest depth within the search limit with no clean
// cost sample, or -1 if all are sampled; the bound keeps cost-seeding from
// outrunning the acceptance frontier or the drafter.
func (c *depthController) costSeedDepth(limit int) int {
	for n := 0; n <= limit; n++ {
		if !c.cost.sampled(n) {
			return n
		}
	}
	return -1
}

// selected returns the EV-optimal draft depth without mutating probe state, the
// argmax within the search limit. The frontier bound keeps the inherited optimistic
// rate from making ever-deeper depths look best; the depth-0 floor lets it stop
// speculating. Returns 0 until the cost model can compare depths.
func (c *depthController) selected(limit int, expected func(int) float64) int {
	if !c.cost.ready() {
		return 0
	}
	best, bestEV := 0, expected(0)/c.cost.cost(0)
	for n := 1; n <= limit; n++ {
		if ev := expected(n) / c.cost.cost(n); ev > bestEV {
			best, bestEV = n, ev
		}
	}
	return best
}

// costEWMAAlpha weights the newest cost sample. Fixed-depth cost is low-variance,
// so a responsive alpha converges in a few visits while smoothing scheduler jitter.
const costEWMAAlpha = 0.3

// costModel is the target-forward cost for validating N drafts — a fused batch of
// 1 current token plus N drafts — as an EWMA per visited draft depth read by
// piecewise-linear interpolation between samples (flat beyond the extremes).
// Interpolation assumes no curve shape, so a steep compute-bound or flat
// bandwidth-bound forward is represented as measured. Cost is static within a run,
// learned from decode steps that already sync the forward, so there is no startup
// probe.
type costModel struct {
	ewma   map[int]float64 // EWMA of measured ms by draft depth
	depths []int           // sampled draft depths, sorted; updated as new depths arrive
}

func newCostModel() *costModel {
	return &costModel{ewma: map[int]float64{}}
}

// costClampFraction bounds one sample's EWMA innovation. A host stall (cache trim,
// backpressure) can inflate a sample severalfold; unclamped it can flip the EV
// comparison against plain decode, and once the controller stops parking it stops
// resampling depth 0, so the error never heals. A genuine cost change still
// converges because every sample keeps pushing toward it.
const costClampFraction = 0.25

// observe folds one forward's wall time into the draft depth's EWMA, clamping the
// innovation so one stall-inflated sample cannot move it far.
func (m *costModel) observe(drafts int, dt time.Duration) {
	if drafts < 0 || dt <= 0 {
		return
	}
	ms := float64(dt) / float64(time.Millisecond)
	if v, ok := m.ewma[drafts]; ok {
		limit := costClampFraction * v
		m.ewma[drafts] = v + costEWMAAlpha*max(-limit, min(limit, ms-v))
	} else {
		m.ewma[drafts] = ms
		i, _ := slices.BinarySearch(m.depths, drafts)
		m.depths = slices.Insert(m.depths, i, drafts)
	}
}

// ready reports whether two distinct depths have been sampled, so a slope exists.
func (m *costModel) ready() bool { return len(m.ewma) >= 2 }

func (m *costModel) sampled(drafts int) bool {
	_, ok := m.ewma[drafts]
	return ok
}

// cost returns the estimated forward wall time (ms) for validating drafts tokens:
// a piecewise-linear interpolation of the per-depth EWMAs, clamping to the nearest
// sample outside the sampled range (the curve beyond is unknown).
func (m *costModel) cost(drafts int) float64 {
	ds := m.depths
	if len(ds) == 0 {
		return 0
	}
	if drafts <= ds[0] {
		return m.ewma[ds[0]]
	}
	if drafts >= ds[len(ds)-1] {
		return m.ewma[ds[len(ds)-1]]
	}
	for i := 1; i < len(ds); i++ {
		if drafts <= ds[i] {
			lo, hi := ds[i-1], ds[i]
			t := float64(drafts-lo) / float64(hi-lo)
			return m.ewma[lo] + t*(m.ewma[hi]-m.ewma[lo])
		}
	}
	return m.ewma[ds[len(ds)-1]]
}

// sampleString reports the EWMA ms per visited draft depth for diagnostics.
func (m *costModel) sampleString() string {
	ds := m.depths
	parts := make([]string, 0, len(ds))
	for _, d := range ds {
		parts = append(parts, fmt.Sprintf("%d:%.0fms", d, m.ewma[d]))
	}
	return strings.Join(parts, " ")
}

// acceptanceEWMAAlpha weights the newest acceptance outcome. Acceptance drifts with
// content, so an EWMA follows the drift instead of being anchored by early tokens.
const acceptanceEWMAAlpha = 0.1

// acceptanceMinSamples is how many reaches a position needs before its rate is
// trusted; since the search reaches one past the frontier, it also gates how fast
// the frontier advances. Set near the EWMA's memory (~1/alpha); larger only slows
// the ramp, since each depth is re-measured as it is drafted.
const acceptanceMinSamples = 10

// acceptanceModel estimates the chance that position i is accepted given that
// every preceding draft was accepted. Its EWMA follows changes in the text.
type acceptanceModel struct {
	minSamples int
	rate       []float64 // EWMA acceptance rate given the prefix survived; index i is position i
	seen       []int     // times position i was reached (warmup gate for rate)

	// pooled shares one rate among the positions past the first. A block
	// draft's later positions accept at similar rates, and pooling lets every
	// reached position inform the ones beyond the current depth, which only
	// probes reach. Their per-position rates still update but are not read.
	pooled   bool
	deep     float64
	deepSeen int
}

func newAcceptanceModel(minSamples int) *acceptanceModel {
	return &acceptanceModel{minSamples: minSamples, rate: []float64{0}, seen: []int{0}}
}

// newBlockAcceptanceModel trusts the first observation of each position, so a
// full first probe covers the whole block, and pools the positions past the first.
func newBlockAcceptanceModel() *acceptanceModel {
	a := newAcceptanceModel(1)
	a.pooled = true
	return a
}

func (a *acceptanceModel) grow(i int) {
	for len(a.seen) <= i {
		a.rate = append(a.rate, 0)
		a.seen = append(a.seen, 0)
	}
}

// observe folds a step's outcome into each reached position's EWMA. A position is
// reached only when the prefix before it survived (accepted >= i-1), and is accepted
// iff accepted >= i; updating only the surviving prefix avoids diluting deeper
// positions the step never reached.
func (a *acceptanceModel) observe(drafted, accepted int) {
	for i := 1; i <= drafted; i++ {
		if accepted < i-1 {
			break // prefix did not survive to position i; deeper positions unreached
		}
		a.grow(i)
		outcome := 0.0
		if accepted >= i {
			outcome = 1.0
		}
		if a.seen[i] == 0 {
			a.rate[i] = outcome
		} else {
			a.rate[i] += acceptanceEWMAAlpha * (outcome - a.rate[i])
		}
		a.seen[i]++
		if a.pooled && i >= 2 {
			if a.deepSeen == 0 {
				a.deep = outcome
			} else {
				a.deep += acceptanceEWMAAlpha * (outcome - a.deep)
			}
			a.deepSeen++
		}
	}
}

// acceptance returns the rate position i is accepted given its prefix survived. An under-sampled
// position inherits the deepest trusted rate rather than zero, so the controller
// keeps exploring deeper instead of locking shallow on noise. A pooled model
// answers positions past the first with their shared rate once it is trusted.
func (a *acceptanceModel) acceptance(i int) float64 {
	if a.pooled && i >= 2 && a.deepSeen >= a.minSamples {
		return a.deep
	}
	if i >= 1 && i < len(a.seen) && a.seen[i] >= a.minSamples {
		return a.rate[i]
	}
	// Inherit the deepest trusted rate; fall back to optimistic 1 if none yet.
	for j := i - 1; j >= 1; j-- {
		if j < len(a.seen) && a.seen[j] >= a.minSamples {
			return a.rate[j]
		}
	}
	return 1
}

// expectedCommitted returns expected committed tokens at depth N: the current token,
// which always commits, plus the expected number of accepted drafts — each draft
// position contributes the probability its whole prefix was accepted, the running
// product of the per-position acceptance rates summed over positions.
func (a *acceptanceModel) expectedCommitted(n int) float64 {
	total, prod := 1.0, 1.0
	for k := 1; k <= n; k++ {
		prod *= a.acceptance(k)
		total += prod
	}
	return total
}

// frontier is the deepest position with a trusted acceptance rate. The controller
// never selects beyond frontier+1, so the selection grows outward one position at a
// time instead of jumping deep on the inherited optimistic rate.
func (a *acceptanceModel) frontier() int {
	f := 0
	for i := 1; i < len(a.seen); i++ {
		if a.seen[i] >= a.minSamples {
			f = i
		} else {
			break
		}
	}
	return f
}
