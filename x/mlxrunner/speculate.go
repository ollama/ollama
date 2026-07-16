package mlxrunner

import (
	"fmt"
	"slices"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

// drafter proposes speculative tokens for the engine to validate, learning
// the conversation through the committed-stream reports.
type drafter interface {
	// propose returns up to maxTokens draft tokens with their proposal
	// distributions, or nil to decode this round plainly.
	propose(current *mlx.Array, maxTokens int) *draftCandidates

	// committed reports a run of tokens committed to the target caches:
	// tokens[i] sits at slot position+i and hiddens row i is the target
	// hidden state at that slot. Runs arrive in slot order — prefill
	// chunks, the decode seed, then each round's validated tokens.
	committed(tokens, hiddens *mlx.Array, position int)

	// settle completes any open frontier pair with next — the token after
	// the last committed slot — and writes buffered reports through,
	// leveling the draft caches with the target caches.
	settle(next *mlx.Array)

	close()
}

// speculation is the persistent speculative-decoding subsystem of a Runner —
// one per loaded model, holding everything that outlives a request: the draft
// model, the target-cache partition, and the depth state learned across
// requests so each request starts at the proven-out depth. Each request opens a
// short-lived speculationSession cursor over it; nothing here is rebuilt per request. A
// nil *speculation means the checkpoint ships no draft head, so every request
// decodes plainly.
type speculation struct {
	r     *Runner
	draft base.DraftModel

	// caches is the whole persistent slice, passed to every forward; draftKV
	// are the draft head's own caches and targets are the rest — the caches the
	// target forward writes, which speculation snapshots and rollback cover.
	// Bound the first time the caches exist (the Runner reuses one cache slice
	// for its life) and stable thereafter.
	caches  []cache.Cache
	draftKV []cache.Cache
	targets []cache.Cache

	// drafter is the persistent half of the MTP drafting machinery; each
	// request's session comes from drafter.open.
	drafter *mtpDrafter

	// depth selects each request's draft length and owns the cost/acceptance
	// models and probe cadence it learns across requests.
	depth *depthController
}

// newSpeculation builds the speculative-decoding subsystem for a loaded model,
// or nil when the checkpoint ships no draft head.
func newSpeculation(r *Runner, draft base.DraftModel) *speculation {
	if draft == nil {
		return nil
	}
	s := &speculation{r: r, draft: draft, depth: newDepthController()}
	s.bind(r.cache.caches)
	s.drafter = newMTPDrafter(s)
	return s
}

// bind computes the draft/target cache partition the first time the persistent
// caches exist; later requests reuse the same slice, so it runs once.
func (s *speculation) bind(caches []cache.Cache) {
	if s.caches != nil {
		if !slices.Equal(s.caches, caches) {
			panic("speculation: cache slice changed between requests")
		}
		return
	}
	draftKV := s.draft.DraftCaches(caches)

	// Partition caches into target slots (everything not in draftKV) in one
	// pass. The count check rejects a draft slot that isn't a member of caches.
	targets := make([]cache.Cache, 0, len(caches))
	for _, c := range caches {
		if !slices.Contains(draftKV, c) {
			targets = append(targets, c)
		}
	}
	if len(caches)-len(targets) != len(draftKV) {
		panic("speculation: DraftCaches must select slots of the cache slice")
	}

	s.caches = caches
	s.draftKV = draftKV
	s.targets = targets
}

// speculationSession is the per-request cursor over the persistent speculation:
// it owns the drafter and runs the validate rounds. A nil session is a plain
// decode.
type speculationSession struct {
	spec    *speculation
	drafter drafter
	enabled bool // whether this request drafts; false parks (maintain-only)
	limit   int  // current draft length
	stats   specStats

	// Cost sampling: each round's wall time (start to next start, spanning the
	// next emit's sync) is attributed to its draft depth only when the depth
	// matches the previous round's, since batch-shape transitions inflate it.
	lastRoundStart time.Time
	prevDrafts     int
	roundDrafts    int
}

// open returns the speculation cursor for this request or nil when the model ships
// no draft head (a nil receiver), which decodes plainly.
func (s *speculation) open(request Request, caches []cache.Cache) *speculationSession {
	if s == nil {
		return nil
	}
	s.bind(caches)
	d := s.drafter.open()

	// Logprobs are not yet supported, so a logprobs request keeps a speculationSession
	// only to maintain a draft cache in lockstep (permanently parked).
	opts := request.SamplerOpts
	enabled := !opts.Logprobs && opts.TopLogprobs == 0

	spec := &speculationSession{spec: s, drafter: d, enabled: enabled, prevDrafts: -1, roundDrafts: -1}
	if enabled {
		spec.limit = s.depth.scheduled
		spec.stats.maxDraft = spec.limit
	}
	return spec
}

// beginRound records the previous round's cost sample (its wall time runs to
// this round's start) and starts timing the new one.
func (s *speculationSession) beginRound() {
	now := time.Now()
	if !s.lastRoundStart.IsZero() && s.roundDrafts >= 0 {
		s.stats.recordRound(s.roundDrafts)
		if s.roundDrafts == s.prevDrafts {
			s.spec.depth.cost.observe(s.roundDrafts, now.Sub(s.lastRoundStart))
		}
		s.prevDrafts = s.roundDrafts
	}
	s.lastRoundStart = now
}

// endRound records a completed round's draft depth, proposal outcome, and the
// controller's next draft length. observed is the leading draft positions the
// acceptance model learns from: the full round, except an accepted EOS holds out
// positions past it (a terminator, not a target rejection).
func (s *speculationSession) endRound(drafted, accepted, observed int) {
	s.roundDrafts = drafted
	s.stats.iterations++
	s.stats.drafted += drafted
	s.stats.accepted += accepted
	if s.enabled {
		if observed > 0 {
			s.spec.depth.acc.observe(observed, accepted)
		}
		s.limit = s.spec.depth.next()
		s.stats.maxDraft = max(s.stats.maxDraft, s.limit)
	}
}

func (s *speculationSession) committed(tokens, hiddens *mlx.Array, position int) {
	if s == nil {
		return
	}
	s.drafter.committed(tokens, hiddens, position)
}

// settle completes the drafter's open frontier pair with next and writes
// buffered reports through to the draft caches.
func (s *speculationSession) settle(next *mlx.Array) {
	if s == nil {
		return
	}
	s.drafter.settle(next)
}

func (s *speculationSession) close() {
	if s == nil {
		return
	}
	s.drafter.close()
}

// speculativeDecoder decodes one speculative round per call: the engine
// forwards the current token (emitted by the previous call, so a token that
// ends generation is never forwarded) fused with the drafter's proposals,
// returning the round's accepted tokens followed by the next token. The seed
// primes current and is never returned. While the engine cannot draft (parked
// at depth zero, or nothing committed to propose from) calls delegate to an
// inner pipelined decoder at plain decode speed.
type speculativeDecoder struct {
	s        *speculationSession
	position int
	current  sampler.Result    // emitted (or the seed), not yet forwarded
	inner    *pipelinedDecoder // pipelines plain tokens while parked; nil while drafting
}

// decoder returns the decoder for this engine's session. A speculationSession that
// cannot draft (logprobs) has no depth controller and permanently parks,
// running the inner pipelined decoder whose reports keep the draft KV level.
func (s *speculationSession) decoder(seed *mlx.Array, position int) decoder {
	current := sampler.Result{Token: seed}
	mlx.Pin(current.Arrays()...)
	return &speculativeDecoder{s: s, position: position, current: current}
}

func (st *speculativeDecoder) next(remaining int) ([]sampler.Result, error) {
	// Route: end a parked stretch by emitting the inner sample, draft on a
	// positive length and a primed drafter, else decode parked.
	var results []sampler.Result
	if s := st.s; st.inner != nil && s.limit > 0 {
		results = st.resume()
	} else {
		s.beginRound()
		var candidates *draftCandidates
		if s.limit > 0 {
			// A round emits the accepted drafts plus one more token (the bonus
			// or residual), so cap the draft one below the remaining budget to
			// land that extra token within it rather than overshooting. At
			// remaining 1 the cap is 0 and the last token decodes plainly.
			candidates = s.drafter.propose(st.current.Token, min(s.limit, remaining-1))
		}
		var accepted, observed int
		var err error
		if candidates == nil {
			results, err = st.park(remaining)
		} else {
			// candidates stays pinned across accept's internal sweep and the
			// draft-count read below; accept pins only its own intermediates.
			mlx.Pin(candidates.tokens)
			defer mlx.Unpin(candidates.tokens)
			results, accepted, observed, err = st.s.accept(&st.position, st.current, candidates)
		}
		if err != nil {
			return nil, err
		}
		drafted := 0
		if candidates != nil {
			drafted = candidates.tokens.Dim(1)
		}
		s.endRound(drafted, accepted, observed)
	}

	st.advance(results[len(results)-1])
	return results, nil
}

// advance retires the last returned token as the next call's current, pinned
// across the sweeps the next call runs before reading it. Nothing is forced here.
func (st *speculativeDecoder) advance(next sampler.Result) {
	mlx.Pin(next.Arrays()...)
	mlx.Unpin(st.current.Arrays()...)
	st.current = next
}

// resume ends a parked stretch: the inner decoder's in-flight sample (sampled
// but never forwarded) is exactly the current token a drafting round expects,
// so emit it and let the next call draft from it.
func (st *speculativeDecoder) resume() []sampler.Result {
	next, position := st.inner.drain()
	st.position = position
	st.inner.close()
	st.inner = nil
	// No round spans this call, so the next beginRound attributes no cost.
	st.s.roundDrafts = -1
	return next
}

// park decodes one pipelined plain token while the engine cannot draft. Each
// is a depth-0 round in the controller's accounting, and the inner decoder's
// reports keep the drafter primed and maintained.
func (st *speculativeDecoder) park(remaining int) ([]sampler.Result, error) {
	s := st.s
	if st.inner == nil {
		st.inner = s.spec.r.pipelinedDecoder(s, s.spec.caches, st.current.Token.ExpandDims(-1), st.position)
	}
	return st.inner.next(remaining)
}

// drain surrenders the inner decoder's undelivered sample while parked; a
// drafting decoder has already delivered everything it sampled.
func (st *speculativeDecoder) drain() ([]sampler.Result, int) {
	if st.inner != nil {
		return st.inner.drain()
	}
	return nil, st.position
}

func (st *speculativeDecoder) close() {
	if st.inner != nil {
		// Ended while parked: the inner decoder's close settles the drafter
		// with its in-flight sample.
		st.inner.close()
	} else {
		// The final token was emitted but never forwarded; its report settles
		// the drafter level with the caches' resting offset.
		st.s.settle(st.current.Token)
	}
	mlx.Unpin(st.current.Arrays()...)
	st.s.logStats()
}

// draftCandidates is one round's draft tokens and the proposal distribution
// each was sampled from, weighed against the target during acceptance.
type draftCandidates struct {
	tokens *mlx.Array
	dist   sampler.Distribution
}

func (c *draftCandidates) Arrays() []*mlx.Array {
	if c == nil {
		return nil
	}
	return append([]*mlx.Array{c.tokens}, c.dist.Arrays()...)
}

// scheduleSpeculation schedules per-token snapshots at offsets
// [before, before+draftCount) on every cache, so the speculative forward
// captures a rollback point before each draft token's write.
func scheduleSpeculation(caches []cache.Cache, before, draftCount int) {
	offsets := make([]int, draftCount)
	for i := range offsets {
		offsets[i] = before + i
	}
	for _, c := range caches {
		if c != nil {
			c.PrepareSnapshots(offsets)
		}
	}
}

// commitSpeculation rolls every cache back to before+accepted, keeping only
// the accepted prefix; full acceptance needs no restore. Rollback tries a
// live rewind first (Restore(nil)) and falls back to the captured snapshot.
func commitSpeculation(caches []cache.Cache, accepted, draftCount, before int) {
	target := before + accepted
	for _, c := range caches {
		if c == nil {
			continue
		}
		snaps := c.TakeSnapshots()
		if accepted < draftCount {
			// Close the snapshots we won't restore from before restoring: a
			// snapshot restore on a wrapped RotatingKVCache copies out every
			// outstanding lazy snapshot before it rebuilds the buffer, so
			// dropping the unused ones first stops that copy-out from
			// materializing snapshots we are about to discard anyway.
			for i, s := range snaps {
				if s != nil && i != accepted {
					s.Close()
					snaps[i] = nil
				}
			}
			if !c.Restore(nil, target) && !c.Restore(snaps[accepted], target) {
				panic(fmt.Sprintf("speculation: cache restore to %d failed", target))
			}
		}
		for _, s := range snaps {
			if s != nil {
				s.Close()
			}
		}
	}
}

// accept accepts the longest draft prefix that survives rejection sampling,
// returning the accepted drafts followed by the target's own next token
// (residual at a rejection, bonus past a full run), except an accepted EOS
// ends the run with no continuation. observed is the leading positions the
// acceptance model learns from, capped at the EOS (a terminator, not a target
// rejection). NumPredict is the decode loop's to enforce, so a token past the
// budget is left for decode to drop, not cut here.
//
// The caller keeps current and the candidate tokens pinned across the call,
// since accept sweeps before its eval and reads both afterward; accept pins
// only the intermediates it produces.
func (s *speculationSession) accept(position *int, current sampler.Result, candidates *draftCandidates) (results []sampler.Result, accepted, observed int, err error) {
	r := s.spec.r
	before := *position
	draftCount := candidates.tokens.Dim(1)
	scheduleSpeculation(s.spec.targets, before+1, draftCount)

	// Every exit between schedule and commit must drain the snapshot
	// schedule and roll the speculative writes back out of the live caches:
	// an undrained schedule panics the next PrepareSnapshots, and
	// uncommitted speculative tokens reach the trie through session.close.
	committed := false
	commit := func(keep int) {
		if committed {
			return
		}
		committed = true
		commitSpeculation(s.spec.targets, keep, draftCount, before+1)
	}
	defer commit(0)

	hiddenSeq := r.Model.Forward(&batch.Batch{
		InputIDs:     current.Token.ExpandDims(-1).Concatenate(1, candidates.tokens),
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(draftCount + 1)},
	}, s.spec.caches)

	// Row i of the fused hidden is the state after the token at before+i, so
	// the rows already line up with the drafts: row 0 (current's state)
	// predicts draft 0, and the row after the last accepted draft is the
	// bonus row. No separate base-logits forward exists on this path.
	targetDist := r.Sampler.Distribution(pipelineSlot, r.Model.Unembed(hiddenSeq), candidates.tokens)
	draftDist := candidates.dist
	acceptedMask := r.sampleAcceptedMask(targetDist.SliceRows(0, draftCount), draftDist, candidates.tokens)

	// The next token is sampled for every possible outcome before anything
	// is evaluated — the residual at each rejection point in one batched
	// draw, plus the bonus row — so a single Eval covers acceptance and the
	// next token instead of a second host round trip after the rejection
	// point is known.
	residualTokens := r.Sampler.SampleDistribution(pipelineSlot, targetDist.SliceRows(0, draftCount).ResidualAgainst(draftDist))
	bonusToken := r.sampleTokenAt(targetDist, draftCount)

	// Pin the arrays read after the eval, then sweep so the draft proposal
	// chain and this validation forward's intermediates are freed as the eval
	// consumes them, the way the plain decode dispatch sweeps before its eval.
	// current and the candidate tokens stay pinned by the caller across the call.
	live := []*mlx.Array{hiddenSeq, acceptedMask, residualTokens, bonusToken}
	mlx.Pin(live...)
	defer mlx.Unpin(live...)
	mlx.Sweep()
	mlx.Eval(candidates.tokens, acceptedMask, residualTokens, bonusToken)

	draftIDs := candidates.tokens.Ints()
	acceptedFlags := acceptedMask.Ints()
	for _, ok := range acceptedFlags {
		if ok == 0 {
			break
		}
		accepted++
	}
	if accepted > draftCount {
		return nil, 0, 0, fmt.Errorf("speculation validation accepted %d tokens for %d draft tokens", accepted, draftCount)
	}
	observed = draftCount

	// Find where an accepted EOS ends the run, before committing, so the cut
	// is known while the per-token rollback snapshots still exist. The EOS also
	// caps observed: positions past it are held out, not logged as rejections.
	commitIDs := make([]int32, 0, accepted+1)
	keep := accepted
	done := false
	for i, id := range draftIDs[:accepted] {
		commitIDs = append(commitIDs, int32(id))
		if r.Tokenizer.IsEOS(int32(id)) {
			done = true
			accepted = i + 1
			observed = accepted
			// Leave the EOS's own state uncommitted: the next sequence won't
			// contain this EOS, and recurrent state can't drop a folded-in
			// token, so committing it would carry the caches past the
			// reusable prefix and force the next sequence to recompute.
			keep = i
			break
		}
	}

	commit(keep)
	*position = before + 1 + keep

	// Report the validated run (current plus kept drafts) to the drafter before
	// returning, so a cancelled emission still leaves it matching the caches. A
	// done generation's final token is uncommitted; it reaches finish instead.
	runIDs := append([]int32{int32(current.Token.Int())}, commitIDs[:keep]...)
	s.drafter.committed(
		mlx.FromValues(runIDs, 1, len(runIDs)),
		hiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, len(runIDs)), mlx.Slice()),
		before)

	results = draftResults(draftIDs[:accepted])
	if done {
		r.Sampler.Commit(pipelineSlot, commitIDs)
		return results, accepted, observed, nil
	}

	var nextID int32
	if accepted < draftCount {
		nextID = int32(residualTokens.Ints()[accepted])
	} else {
		nextID = int32(bonusToken.Int())
	}
	commitIDs = append(commitIDs, nextID)
	r.Sampler.Commit(pipelineSlot, commitIDs)

	results = append(results, sampler.Result{Token: mlx.FromValues([]int32{nextID}, 1)})
	return results, accepted, observed, nil
}

func (r *Runner) sampleAcceptedMask(targetDist, draftDist sampler.Distribution, draftTokens *mlx.Array) *mlx.Array {
	p := targetDist.Prob(draftTokens)
	q := draftDist.Prob(draftTokens)
	acceptP := mlx.Minimum(p.Divide(q), mlx.FromValue(float32(1)))
	return r.Sampler.Bernoulli(pipelineSlot, acceptP).AsType(mlx.DTypeInt32)
}

func (r *Runner) sampleTokenAt(dist sampler.Distribution, index int) *mlx.Array {
	return r.Sampler.SampleDistribution(pipelineSlot, dist.SliceRows(index, index+1))
}

// draftResults wraps accepted draft ids as sampler results; drafts carry no
// logprobs, so only the token id is set.
func draftResults(ids []int) []sampler.Result {
	results := make([]sampler.Result, len(ids))
	for i, id := range ids {
		results[i] = sampler.Result{Token: mlx.FromValues([]int32{int32(id)}, 1)}
	}
	return results
}
