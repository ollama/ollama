package mlxrunner

import (
	"errors"
	"fmt"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	sampler "github.com/ollama/ollama/mlxrunner/sample"
)

// draftSession proposes speculative tokens for one request, learning the
// conversation through the committed-stream reports.
type draftSession interface {
	// propose returns up to maxTokens draft tokens with their proposal
	// distributions, or nil to decode this round plainly.
	propose(current *mlx.Array, maxTokens int) *draftCandidates

	// committed reports a run of tokens committed to the target caches:
	// tokens[i] sits at slot position+i and hiddens row i is the target
	// hidden state at that slot. Runs arrive in slot order — prefill
	// chunks, the decode seed, then each round's validated tokens. media is
	// the run's manifest (feature-bearing for items the run overlaps), valid
	// only for the call; a session that defers its forward holds what it
	// keeps. Nil outside prefill.
	committed(tokens, hiddens *mlx.Array, position int, media []batch.MediaItem)

	// settle completes any open frontier pair with next — the token after
	// the last committed slot — and writes buffered reports through,
	// leveling the draft caches with the target caches.
	settle(next *mlx.Array)

	close()
}

// speculation is the persistent speculative-decoding subsystem of a Runner —
// one per cache slot, holding everything that outlives a request: the draft
// model, the target-cache partition, and the depth state learned across
// requests so each request starts at the proven-out depth. Each request opens a
// short-lived speculationSession cursor over it; nothing here is rebuilt per request. A
// nil *speculation means the checkpoint ships no draft head, so every request
// decodes plainly.
type speculation struct {
	r      *Runner
	slotID int
	draft  model.DraftModel

	// targets are the model's cache slots, which speculation snapshots and
	// rolls back; draftKV are the drafter's. Both are built at load and
	// never change.
	targets []cache.Cache
	draftKV []cache.Cache

	// drafter is the persistent half of the drafting machinery; each
	// request's session comes from drafter.open.
	drafter drafter

	// depth selects each request's draft length and owns the cost/acceptance
	// models and probe cadence it learns across requests.
	depth *depthController
}

// drafter is the persistent half of a drafting implementation, opening each
// request's drafting session with the request's per-row layout state.
type drafter interface {
	open(layout []any) draftSession

	// draftLimit is the deepest draft this drafter can produce, 0 when nothing
	// bounds it. A depth past it is never measured, so the depth search must
	// not schedule one.
	draftLimit() int
}

// newSpeculation builds the speculative-decoding subsystem for a loaded model,
// or nil when the checkpoint ships no draft head.
func newSpeculation(r *Runner, slotID int, draft model.DraftModel, targets, draftKV []cache.Cache) *speculation {
	if draft == nil {
		return nil
	}
	s := &speculation{r: r, slotID: slotID, draft: draft, targets: targets, draftKV: draftKV, depth: newDepthController()}
	if bd, ok := draft.(model.BlockDraft); ok {
		s.drafter = newDFlashDrafter(s, bd)
	} else {
		s.drafter = newMTPDrafter(s)
	}
	s.depth.drafterLimit = s.drafter.draftLimit()
	return s
}

// speculationSession is the per-request cursor over the persistent speculation:
// it owns the drafter and runs the validate rounds. A nil session is a plain
// decode.
type speculationSession struct {
	spec    *speculation
	drafter draftSession
	enabled bool  // whether this request drafts; false parks (maintain-only)
	limit   int   // current draft length
	layout  []any // the request's per-row layout state, stamped on every target forward
	stats   specStats

	// fixedDepth holds limit for the whole request, bypassing the depth controller.
	fixedDepth bool

	// Cost sampling: each round's wall time (start to next start, spanning the
	// next emit's sync) is attributed to its draft depth only when the depth
	// matches the previous round's, since batch-shape transitions inflate it.
	lastRoundStart time.Time
	prevDrafts     int
	roundDrafts    int
}

// open returns the speculation cursor for this request or nil when the model ships
// no draft head (a nil receiver), which decodes plainly.
func (s *speculation) open(request request, layout []any) *speculationSession {
	if s == nil {
		return nil
	}
	d := s.drafter.open(layout)

	// Logprobs are not yet supported, so a logprobs request keeps a speculationSession
	// only to maintain a draft cache in lockstep (permanently parked).
	opts := request.SamplerOpts
	enabled := !opts.Logprobs && opts.TopLogprobs == 0

	spec := &speculationSession{spec: s, drafter: d, layout: layout, enabled: enabled, prevDrafts: -1, roundDrafts: -1}
	if enabled {
		spec.limit = s.depth.scheduled
		if _, ok := s.draft.(model.BlockSelector); ok {
			spec.limit, spec.fixedDepth = dflash2Depth, true
		}
	}
	return spec
}

// beginRound records the previous round's cost sample (its wall time runs to
// this round's start) and starts timing the new one. A session that cannot
// draft records nothing: its parked rounds carry logprobs work that would
// skew the shared depth-0 cost.
func (s *speculationSession) beginRound() {
	if !s.enabled || s.fixedDepth {
		return
	}
	now := time.Now()
	if !s.lastRoundStart.IsZero() && s.roundDrafts >= 0 {
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
	s.recordRound(drafted, accepted)
	// A positive depth can produce no drafts when the request has no draft budget
	// left or its drafter is not primed yet. Keep that choice for the next
	// eligible round, including across requests, instead of consuming a probe.
	if !s.enabled || s.fixedDepth || (s.limit > 0 && drafted == 0) {
		return
	}
	if observed > 0 {
		s.spec.depth.acc.observe(observed, accepted)
	}
	s.limit = s.spec.depth.next()
}

func (s *speculationSession) recordRound(drafted, accepted int) {
	s.stats.maxDraft = max(s.stats.maxDraft, drafted)
	s.stats.recordRound(drafted)
	s.stats.iterations++
	s.stats.drafted += drafted
	s.stats.accepted += accepted
}

func (s *speculationSession) committed(tokens, hiddens *mlx.Array, position int, media []batch.MediaItem) {
	if s == nil {
		return
	}
	s.drafter.committed(tokens, hiddens, position, media)
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
	grammar  *grammar
	scope    *mlx.Scope // holds current across rounds
}

// decoder returns the decoder for this engine's session. A speculationSession that
// cannot draft (logprobs) has no depth controller and permanently parks,
// running the inner pipelined decoder whose reports keep the draft KV level.
func (s *speculationSession) decoder(seed *mlx.Array, position int, grammar *grammar) *speculativeDecoder {
	current := sampler.Result{Token: seed}
	scope := mlx.NewScope()
	scope.Attach(current.Arrays()...)
	return &speculativeDecoder{s: s, position: position, current: current, grammar: grammar, scope: scope}
}

func (st *speculativeDecoder) next(remaining int) ([]sampler.Result, error) {
	// Route: end a parked stretch by emitting the inner sample, draft on a
	// positive length and a primed drafter, else decode parked.
	var results []sampler.Result
	if s := st.s; st.inner != nil && s.limit > 0 {
		var err error
		results, err = st.resume()
		if err != nil {
			return nil, err
		}
	} else {
		s.beginRound()
		var candidates *draftCandidates
		if s.limit > 0 {
			// A round emits the accepted drafts plus one more token (the bonus
			// or residual), so cap the draft one below the remaining budget to
			// land that extra token within it rather than overshooting. At
			// remaining 1 the cap is 0 and the last token decodes plainly.
			candidates = s.drafter.propose(st.current.Token, min(s.limit, remaining-1))
			mlx.AsyncEval(candidates.Arrays()...)
		}
		var accepted, observed int
		var err error
		if candidates == nil {
			results, err = st.park(remaining)
		} else {
			results, accepted, observed, err = st.s.accept(&st.position, st.current, candidates, st.grammar)
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

// advance retires the last returned token as the next call's current, held
// until the next call reads it. Nothing is forced here.
func (st *speculativeDecoder) advance(next sampler.Result) {
	st.scope.Attach(next.Arrays()...)
	st.scope.Discard(st.current.Arrays()...)
	st.current = next
}

// resume ends a parked stretch: the inner decoder's in-flight sample (sampled
// but never forwarded) is exactly the current token a drafting round expects,
// so emit it and let the next call draft from it.
func (st *speculativeDecoder) resume() ([]sampler.Result, error) {
	next, position, err := st.inner.drain()
	if err != nil {
		return nil, err
	}
	st.position = position
	st.inner.close()
	st.inner = nil
	// The drained emission counts as one more plain round, but no round spans
	// this call, so the next beginRound attributes no cost.
	st.s.stats.recordRound(0)
	st.s.stats.iterations++
	st.s.roundDrafts = -1
	return next, nil
}

// park decodes one pipelined plain token while the engine cannot draft. Each
// is a depth-0 round in the controller's accounting, and the inner decoder's
// reports keep the drafter primed and maintained.
func (st *speculativeDecoder) park(remaining int) ([]sampler.Result, error) {
	return st.pipeline().next(remaining)
}

// pipeline parks at the same boundary used by plain single-request decoding.
// A scheduler may advance this decoder together with other requests; next then
// resumes speculative decoding through the existing drain path.
func (st *speculativeDecoder) pipeline() *pipelinedDecoder {
	if st.inner == nil {
		s := st.s
		st.inner = s.spec.r.pipelinedDecoder(s.spec.slotID, s, s.spec.targets, st.current.Token.ExpandDims(-1), st.position, s.layout, st.grammar)
	}
	return st.inner
}

func (st *speculativeDecoder) batched(result sampler.Result) {
	st.advance(result)
	// Contention is not a sample of this request's isolated drafting cost.
	st.s.lastRoundStart = time.Time{}
	st.s.prevDrafts, st.s.roundDrafts = -1, -1
	st.s.stats.recordRound(0)
	st.s.stats.iterations++
}

// drain surrenders the inner decoder's undelivered sample while parked; a
// drafting decoder has already delivered everything it sampled.
func (st *speculativeDecoder) drain() ([]sampler.Result, int, error) {
	if st.inner != nil {
		return st.inner.drain()
	}
	return nil, st.position, nil
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
	st.scope.Close()
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
func (s *speculationSession) accept(position *int, current sampler.Result, candidates *draftCandidates, g *grammar) (results []sampler.Result, accepted, observed int, err error) {
	out := s.spec.r.acceptDrafts([]draftRound{{s, position, current, candidates, g}})[0]
	return out.results, out.accepted, out.observed, out.err
}

// draftRound keeps verification state attached to the request that owns it.
// Equal-length rounds can share the target forward without sharing acceptance,
// sampling history, or rollback positions.
type draftRound struct {
	s          *speculationSession
	position   *int
	current    sampler.Result
	candidates *draftCandidates
	grammar    *grammar
}

type draftOutcome struct {
	results            []sampler.Result
	accepted, observed int
	err                error
}

func (r *Runner) acceptDrafts(rounds []draftRound) []draftOutcome {
	outcomes := make([]draftOutcome, len(rounds))
	if len(rounds) == 0 {
		return outcomes
	}
	draftCount := rounds[0].candidates.tokens.Dim(1)
	owners := make([][]cache.Cache, len(rounds))
	ids := make([]*mlx.Array, len(rounds))
	offsets, lengths := make([]int32, len(rounds)), make([]int32, len(rounds))
	var layout []any
	for i, round := range rounds {
		if round.candidates.tokens.Dim(1) != draftCount {
			for j := range outcomes {
				outcomes[j].err = fmt.Errorf("speculation: verification rows have different lengths")
			}
			return outcomes
		}
		owners[i] = round.s.spec.targets
		ids[i] = round.current.Token.ExpandDims(-1).Concatenate(1, round.candidates.tokens)
		offsets[i], lengths[i] = int32(*round.position), int32(draftCount+1)
		if len(round.s.layout) > 0 {
			if layout == nil {
				layout = make([]any, len(rounds))
			}
			layout[i] = round.s.layout[0]
		}
	}
	layers, err := cache.Batch(owners)
	if err != nil {
		for i := range outcomes {
			outcomes[i].err = err
		}
		return outcomes
	}

	committed := make([]bool, len(rounds))
	for i, round := range rounds {
		scheduleSpeculation(round.s.spec.targets, int(offsets[i])+1, draftCount)
	}
	defer func() {
		for i, round := range rounds {
			if !committed[i] {
				commitSpeculation(round.s.spec.targets, 0, draftCount, int(offsets[i])+1)
			}
		}
	}()

	type verification struct {
		acceptedMask, residualTokens, bonusToken *mlx.Array
		draftIDs                                 []int32
		constrained                              bool
	}
	verified := make([]verification, len(rounds))
	var auxHiddenSeq *mlx.Array
	var retainedFeatures []*mlx.Array
	mlx.ScopedEval(func() []*mlx.Array {
		hidden, aux := r.Model.Forward(&batch.Batch{
			InputIDs: mlx.Concatenate(ids, 0), SeqOffsets: offsets, SeqQueryLens: lengths, Layout: layout,
		}, layers)
		auxHiddenSeq = aux
		logits := r.Model.Unembed(hidden)
		arrays := []*mlx.Array{aux}
		for i, round := range rounds {
			v := &verified[i]
			v.draftIDs = round.candidates.tokens.Ints()
			v.constrained = round.grammar.constraining()
			rowLogits := logits.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice())
			if v.constrained {
				var errs []error
				rowLogits, errs = r.grammarEngine.mask([]*grammar{round.grammar}, rowLogits, [][]int32{v.draftIDs})
				if outcomes[i].err = errors.Join(errs...); outcomes[i].err != nil {
					continue
				}
			}
			slot := round.s.spec.slotID
			targetDist := r.Sampler.Distribution(slot, rowLogits, round.candidates.tokens)
			draftDist := round.candidates.dist
			v.acceptedMask = r.sampleAcceptedMask(slot, targetDist.SliceRows(0, draftCount), draftDist, round.candidates.tokens)
			v.residualTokens = r.Sampler.SampleDistribution(slot, targetDist.SliceRows(0, draftCount).ResidualAgainst(draftDist))
			v.bonusToken = r.sampleTokenAt(slot, targetDist, draftCount)
			arrays = append(arrays, v.acceptedMask, v.residualTokens, v.bonusToken)
		}
		return arrays
	})

	for i, round := range rounds {
		out, v := &outcomes[i], &verified[i]
		if out.err != nil {
			continue
		}
		for _, ok := range v.acceptedMask.Ints() {
			if ok == 0 {
				break
			}
			out.accepted++
		}
		if out.accepted > draftCount {
			out.err = fmt.Errorf("speculation validation accepted %d tokens for %d draft tokens", out.accepted, draftCount)
			continue
		}
		out.observed = draftCount
		commitIDs := make([]int32, 0, out.accepted+1)
		keep := out.accepted
		done := false
		for j, id := range v.draftIDs[:out.accepted] {
			commitIDs = append(commitIDs, id)
			if r.Tokenizer.IsEOS(id) {
				done = true
				out.accepted, out.observed = j+1, j+1
				// The terminating token is emitted but not part of the reusable cache.
				keep = j
				break
			}
		}
		var nextID int32
		if !done {
			if out.accepted < draftCount {
				nextID = v.residualTokens.Ints()[out.accepted]
			} else {
				nextID = v.bonusToken.Int()
			}
			commitIDs = append(commitIDs, nextID)
		}
		if v.constrained {
			for _, id := range commitIDs {
				if out.err = errors.Join(r.grammarEngine.accept([]*grammar{round.grammar}, []int32{id})...); out.err != nil {
					break
				}
			}
			if out.err != nil {
				continue
			}
		}
		before := int(offsets[i])
		committed[i] = true
		commitSpeculation(round.s.spec.targets, keep, draftCount, before+1)
		*round.position = before + 1 + keep
		runIDs := append([]int32{round.current.Token.Int()}, commitIDs[:keep]...)
		features := auxHiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, len(runIDs)), mlx.Slice())
		if len(rounds) > 1 {
			// Draft sessions retain these features between rounds. Own the row
			// so a paused request cannot keep its peers' feature buffer alive.
			features = features.TakeAxis(mlx.FromValues([]int32{int32(i)}, 1), 0)
			retainedFeatures = append(retainedFeatures, features)
		}
		round.s.drafter.committed(mlx.FromValues(runIDs, 1, len(runIDs)), features, before, nil)
		out.results = draftResults(v.draftIDs[:out.accepted])
		if !done {
			out.results = append(out.results, sampler.Result{Token: mlx.FromValues([]int32{nextID}, 1)})
		}
		r.Sampler.Commit(round.s.spec.slotID, commitIDs)
	}
	if len(rounds) > 1 {
		// Materialize owned features and cache state before a reader can pause.
		state := retainedFeatures
		for _, layer := range layers {
			if layer != nil {
				state = append(state, layer.State()...)
			}
		}
		mlx.AsyncEval(state...)
	}
	return outcomes
}

func (r *Runner) sampleAcceptedMask(slotID int, targetDist, draftDist sampler.Distribution, draftTokens *mlx.Array) *mlx.Array {
	p := targetDist.Prob(draftTokens)
	q := draftDist.Prob(draftTokens)
	acceptP := mlx.Minimum(p.Divide(q), mlx.FromValue(float32(1)))
	return r.Sampler.Bernoulli(slotID, acceptP).AsType(mlx.DTypeInt32)
}

func (r *Runner) sampleTokenAt(slotID int, dist sampler.Distribution, index int) *mlx.Array {
	return r.Sampler.SampleDistribution(slotID, dist.SliceRows(index, index+1))
}

// draftResults wraps accepted draft ids as sampler results; drafts carry no
// logprobs, so only the token id is set.
func draftResults(ids []int32) []sampler.Result {
	results := make([]sampler.Result, len(ids))
	for i, id := range ids {
		results[i] = sampler.Result{Token: mlx.FromValues([]int32{id}, 1)}
	}
	return results
}
