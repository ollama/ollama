package mlxrunner

import (
	"context"
	"log/slog"
	"slices"
	"time"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/sample"
)

// snapshotAlignedPrefill keeps prefill batch widths consistent across restores.
type snapshotAlignedPrefill interface {
	AlignPrefillToSnapshots()
}

type batchSlot struct {
	id          int
	cache       *prefixCache
	targets     []cache.Cache
	speculation *speculation
	request     *batchRequest
	evaluation  *activeEvaluation
	speech      *speechRequest
	scoreCache  *prefixCache
	scoreHidden *cache.HiddenCache
}

func (slot *batchSlot) busy() bool {
	return slot.request != nil || slot.evaluation != nil || slot.speech != nil
}

// activeCache identifies state still owned by an executing session. A response
// waiting for a slow reader no longer needs the completed session's cache.
func (slot *batchSlot) activeCache() *prefixCache {
	if slot.request != nil && !slot.request.finished {
		return slot.cache
	}
	if slot.evaluation != nil {
		return slot.scoreCache
	}
	return nil
}

type batchRequest struct {
	request   request
	session   *cacheSession
	media     *requestMedia
	spec      *speculationSession
	prefill   *prefillCursor
	decoder   decoder
	grammar   *grammar
	output    *generationOutput
	responses []CompletionResponse
	finished  bool

	// Captured before waiting for grammar compilation.
	promptEval time.Duration
}

type batchScheduler struct {
	r                 *Runner
	slots             []*batchSlot
	nextPrefill       int
	keepScratch       bool
	drafting          batchDrafting
	snapshotWorkspace int64
	snapshotTimer     *time.Timer
	memory            *discover.MemoryMonitor
}

type draftBatchSize struct {
	rows, drafting int
}

// batchDrafting learns shared-verification costs for each batch width and mix
// of drafting and plain rows. A round spans consecutive decode calls, including
// pipelined GPU work. Prefill, backpressure and shape transitions break samples.
type batchDrafting struct {
	depths         map[draftBatchSize]*depthController
	requests       []*batchRequest
	started        time.Time
	drafts         int
	previousDrafts int
}

func (b *batchDrafting) reset() {
	b.requests = nil
	b.interrupt()
}

// Interleaved work invalidates the timing sample. The same ready requests
// can still finish their scheduled probe on the next complete decode round.
func (b *batchDrafting) interrupt() {
	b.started = time.Time{}
}

func (b *batchDrafting) begin(slots []*batchSlot) *depthController {
	requests := make([]*batchRequest, len(slots))
	size := draftBatchSize{rows: len(slots)}
	block := false
	for i, slot := range slots {
		req := slot.request
		if req.spec == nil || req.spec.fixedDepth {
			b.reset()
			return nil
		}
		requests[i] = req
		if req.spec.enabled {
			size.drafting++
			block = req.spec.depth.block
		}
	}
	if size.drafting == 0 {
		b.reset()
		return nil
	}
	if b.depths == nil {
		b.depths = make(map[draftBatchSize]*depthController)
	}
	c := b.depths[size]
	if c == nil {
		c = newDepthController()
		c.plainTokens = float64(size.rows-size.drafting) / float64(size.drafting)
		c.drafterLimit = slots[0].speculation.drafter.draftLimit()
		b.depths[size] = c
	}
	same := slices.Equal(requests, b.requests)
	if block && !same {
		// The batch shape keeps its measured costs. A different group starts
		// a fresh probe schedule; each request's acceptance stays with its slot.
		c = c.forBlock()
		b.depths[size] = c
	}
	now := time.Now()
	if !b.started.IsZero() && same {
		if b.drafts == b.previousDrafts {
			c.cost.observe(b.drafts, now.Sub(b.started))
		}
		b.previousDrafts = b.drafts
	} else {
		b.previousDrafts = -1
	}
	b.requests, b.started, b.drafts = requests, now, c.scheduled
	if slog.Default().Enabled(context.Background(), logutil.LevelTrace) {
		logutil.Trace("mlx batch draft depth", "rows", len(slots), "drafting", size.drafting, "depth", b.drafts, "cost", c.cost.sampleString())
	}
	return c
}

// A block verifier uses one width for all drafting rows. Its throughput is the
// sum of their expected committed tokens plus one for each plain row, divided
// by the measured cost of the whole batch at that width.
func (b *batchDrafting) next(c *depthController) {
	if !c.block {
		c.next()
		return
	}
	frontier := 0
	for _, req := range b.requests {
		if req.spec.enabled {
			frontier = max(frontier, req.spec.depth.frontier())
		}
	}
	c.nextFor(min(frontier+1, c.drafterLimit), func(n int) float64 {
		var tokens float64
		for _, req := range b.requests {
			if req.spec.enabled {
				tokens += req.spec.depth.acc.expectedCommitted(n)
			} else {
				tokens++
			}
		}
		return tokens
	})
}

func (r *Runner) onMLX(ctx context.Context, f func() error) error {
	if r.mlxThread != nil {
		return r.mlxThread.Do(ctx, f)
	}
	return f()
}

func (r *Runner) runBatchRequests(ctx context.Context) error {
	s := &batchScheduler{r: r, slots: make([]*batchSlot, max(1, r.parallel))}
	defer func() {
		if s.snapshotTimer != nil {
			s.snapshotTimer.Stop()
		}
		if s.memory != nil {
			s.memory.Close()
		}
		r.snapshotLimitBytes = 0
	}()
	defer skipCleanupOnPanic(func() {
		_ = r.onMLX(context.Background(), func() error {
			mlx.Scoped(func() {
				for _, slot := range s.slots {
					if slot == nil {
						continue
					}
					if slot.request != nil {
						s.finish(slot, context.Canceled)
						close(slot.request.request.Responses)
					}
					if slot.evaluation != nil {
						s.finishEvaluation(slot, context.Canceled)
					}
					if slot.speech != nil {
						s.finishSpeech(slot, context.Canceled)
					}
					if slot.cache != nil {
						slot.cache.close()
					}
					if slot.scoreCache != nil {
						slot.scoreCache.close()
					}
				}
			})
			return nil
		})
	})
	slog.Info("MLX request scheduler", "parallel", r.parallel, "drafting", "shared-verification", "attention", "per-request")
	for {
		snapshotChanged := false
		var pending queuedRequest
		var incoming chan queuedRequest
		active := false
		for _, slot := range s.slots {
			if slot == nil || !slot.busy() {
				incoming = r.Requests
			} else {
				active = true
			}
		}
		var maintenance <-chan time.Time
		if s.snapshotTimer != nil {
			maintenance = s.snapshotTimer.C
		}
		if !active {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case pending = <-incoming:
			case <-maintenance:
				snapshotChanged = true
			}
		} else {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case pending = <-incoming:
			case <-maintenance:
				snapshotChanged = true
			default:
			}
		}
		progressed := false
		err := r.onMLX(ctx, func() error {
			measure := false
			var before, peak int
			mlx.Scoped(func() {
				admitted := false
				if pending != nil {
					if _, generation := pending.(request); generation && !active {
						mlx.ResetPeakMemory()
					}
					admitted = s.admit(pending)
					progressed = true
				}
				// Coalesce already waiting callers without delaying the first one.
				// Admission trims with the incoming prefix in view. Do not let a
				// simultaneous timer wake discard it before lookup.
				admitted = s.admitPending() || admitted
				if snapshotChanged && !admitted {
					s.trimSnapshots(nil, -1)
				}
				step := func() {
					worked := s.step()
					progressed = progressed || worked
				}
				measure = s.hasPrefixCache() && s.memory != nil
				if measure {
					before = mlx.ActiveMemory()
					peak = mlx.MeasurePeakMemory(step)
				} else {
					step()
				}
			})
			if measure {
				// Reserve transient workspace and growth of live cache buffers.
				// Prefill buffers can be larger than their decode-time state.
				s.snapshotWorkspace = max(s.snapshotWorkspace, int64(peak-before))
			}
			// Release unused allocator buffers after the round's temporaries
			// are freed, once all responses have drained.
			for _, slot := range s.slots {
				if slot != nil && slot.busy() {
					return nil
				}
			}
			if s.keepScratch && mlx.MetalIsAvailable() {
				mlx.DefaultStream().Synchronize()
				if mlx.CacheMemory() <= scoreScratchLimit {
					return nil
				}
			}
			mlx.ClearCache()
			return nil
		})
		if err != nil {
			// Dispatch failed before admission; no slot owns this request yet.
			if pending != nil {
				pending.reject(err)
			}
			return err
		}
		if !progressed {
			// Only backpressure or a grammar compile can leave an admitted request
			// idle. This bounded wait also notices cancellation without GPU work.
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(2 * time.Millisecond):
			}
		}
	}
}

func (s *batchScheduler) admit(pending queuedRequest) bool {
	if err := pending.context().Err(); err != nil {
		pending.reject(err)
		return false
	}
	// Choose the destination before trimming so retention uses the same local
	// cache and peer-checkpoint rules as the request that follows.
	index := s.selectSlot(pending)
	s.trimSnapshots(pending, index)
	switch req := pending.(type) {
	case request:
		s.admitGeneration(req, index)
	case *evaluationRequest:
		s.admitEvaluation(req, index)
	case *speechRequest:
		s.admitSpeech(req)
	default:
		panic("unknown scheduler request type")
	}
	// The first prefix cache is created during admission. Seed the observation
	// before its first forward so that any resulting compression remains visible.
	if s.snapshotTimer == nil {
		s.trimSnapshots(pending, index)
	}
	return true
}

// Generation prefers the longest local match; other requests use the first
// available slot. A completed response still draining to its reader is busy.
func (s *batchScheduler) selectSlot(pending queuedRequest) int {
	var keys []uint32
	req, generation := pending.(request)
	if generation {
		keys = effectiveKeyTokens(req.Tokens, req.MediaItems)
	}
	best, bestMatch := -1, -1
	for i, slot := range s.slots {
		if slot != nil && slot.busy() {
			continue
		}
		matched := 0
		if len(keys) > 0 && slot != nil && slot.cache != nil && slot.cache.root != nil {
			plan := slot.cache.prefillPlan(len(req.Tokens), req.MediaItems)
			_, matched = findBestMatch(slot.cache.root, plan.tag(slot.cache.key(keys)))
			matched = plan.restoreOffset(matched)
		}
		if matched > bestMatch {
			best, bestMatch = i, matched
		}
	}
	return best
}

func (s *batchScheduler) admitGeneration(request request, best int) {
	s.keepScratch = false
	if err := request.Ctx.Err(); err != nil {
		request.reject(err)
		return
	}
	slot := s.slots[best]
	if slot == nil {
		slot = &batchSlot{id: best}
		s.slots[best] = slot
	}
	if slot.cache == nil {
		targets := s.r.Model.NewCaches()
		var draft []cache.Cache
		if s.r.spec != nil {
			draft = newDraftCaches(s.r.spec.draft)
		}
		slot.targets = targets
		slot.cache = newPrefixCache(slices.Concat(targets, draft))
		_, slot.cache.alignPrefill = s.r.Model.(snapshotAlignedPrefill)
		slot.cache.limitBytes = s.r.snapshotLimit()
		if s.r.spec != nil {
			slot.speculation = newSpeculation(s.r, slot.id, s.r.spec.draft, targets, draft)
			slot.cache.draftLookahead = s.r.cache.draftLookahead
		}
		s.slots[best] = slot
	}
	req := &batchRequest{request: request}
	var peers []*prefixCache
	for _, other := range s.slots {
		if other != nil && other != slot && other.cache != nil {
			peers = append(peers, other.cache)
		}
	}
	req.session = slot.cache.begin(request.Tokens, request.MediaItems, peers...)
	req.media = s.r.openMedia(request)
	req.spec = slot.speculation.open(request, req.media.rowLayout())
	req.prefill = s.r.newPrefill(req.session, req.spec, req.media)
	slot.request = req
}

func (s *batchScheduler) finish(slot *batchSlot, err error) {
	req := slot.request
	if req.finished {
		return
	}
	if req.decoder != nil {
		results, _, _ := req.decoder.drain()
		for _, result := range results {
			req.session.outputs = append(req.session.outputs, result.Token.Int())
		}
		req.decoder.close()
	} else if req.prefill != nil {
		p := req.prefill
		req.spec.settle(mlx.FromValues(p.session.remaining[p.processed:p.processed+1], 1))
	}
	if req.output != nil {
		s.r.Sampler.Remove(slot.id)
	}
	req.spec.close()
	if req.spec != nil && req.spec.failedVerification {
		// Discard uncommitted writes. Previously saved prefixes remain available on retry.
		slot.cache.switchToPath([]*trieNode{slot.cache.root}, 0)
	} else {
		req.session.close()
	}
	if slog.Default().Enabled(req.request.Ctx, logutil.LevelTrace) {
		slot.cache.dumpTree()
	}
	req.media.close()
	req.request.Grammar.close()
	req.finished = true
	if err != nil {
		req.responses = append(req.responses, CompletionResponse{Error: completionStatusError(err)})
	} else {
		req.responses = append(req.responses, req.output.finish())
	}
	slog.Debug("MLX batch request finished", "slot", slot.id, "memory", mlx.Memory{})
}

func (s *batchScheduler) step() bool {
	progressed := false
	for _, slot := range s.slots {
		if slot == nil || slot.request == nil {
			continue
		}
		req := slot.request
		if err := req.request.Ctx.Err(); err != nil {
			s.finish(slot, err)
			req.responses = nil
		}
		for len(req.responses) > 0 {
			select {
			case req.request.Responses <- req.responses[0]:
				req.responses = req.responses[1:]
				progressed = true
			default:
				goto blocked
			}
		}
	blocked:
		if req.finished && len(req.responses) == 0 {
			close(req.request.Responses)
			slot.request = nil
			progressed = true
		}
	}
	for _, slot := range s.slots {
		if slot != nil && slot.evaluation != nil {
			if err := slot.evaluation.request.ctx.Err(); err != nil {
				s.finishEvaluation(slot, err)
				progressed = true
			}
		}
	}
	// At most one generation prompt chunk or scoring/embedding forward per decode round.
	// Rotate across both kinds so neither can starve the other.
	for n := range len(s.slots) {
		i := (s.nextPrefill + n) % len(s.slots)
		slot := s.slots[i]
		if slot != nil && slot.evaluation != nil {
			s.nextPrefill = (i + 1) % len(s.slots)
			s.drafting.interrupt()
			s.stepEvaluation(slot)
			progressed = true
			break
		}
		if slot == nil || slot.request == nil || slot.request.finished || slot.request.prefill == nil {
			continue
		}
		req := slot.request
		s.drafting.interrupt()
		before := req.prefill.processed
		ready, err := req.prefill.step(req.request.Ctx)
		progressed = progressed || req.prefill.processed != before
		s.nextPrefill = (i + 1) % len(s.slots)
		if err != nil {
			s.finish(slot, err)
			break
		}
		if ready {
			if req.promptEval == 0 {
				req.promptEval = time.Since(req.prefill.started)
			}
			// Compilation runs concurrently; do not block the MLX worker waiting on it.
			compiled := true
			if c := req.request.Grammar; c != nil {
				select {
				case <-c.done:
				default:
					compiled = false
				}
			}
			if !compiled {
				break
			}
			progressed = true
			req.grammar, err = req.request.Grammar.resolve(req.request.Ctx)
			if err != nil {
				s.finish(slot, err)
				break
			}
			// Include final cache settlement, but exclude the grammar wait.
			started := time.Now()
			seed := req.prefill.seed()
			req.promptEval += time.Since(started)
			s.r.Sampler.Add(slot.id, req.request.SamplerOpts, req.request.Tokens)
			if req.spec != nil {
				req.decoder = req.spec.decoder(seed, req.prefill.position, req.grammar)
			} else {
				req.decoder = s.r.pipelinedDecoder(slot.id, nil, slot.targets, seed.ExpandDims(-1), req.prefill.position, req.media.rowLayout(), req.grammar)
			}
			req.output = newGenerationOutput(s.r, req.request, req.session, req.promptEval)
			req.prefill = nil
		}
		break
	}
	if s.decodeSpeech() {
		s.drafting.interrupt()
		progressed = true
	}
	var ready []*batchSlot
	active := 0
	for _, slot := range s.slots {
		if slot == nil || slot.request == nil || slot.request.finished {
			continue
		}
		active++
		if slot.request.prefill == nil && len(slot.request.responses) == 0 {
			ready = append(ready, slot)
		}
	}
	if len(ready) == 0 {
		s.drafting.reset()
		return progressed
	}
	if err := s.decode(ready, active); err != nil {
		// Cache compatibility is checked before a shared forward writes state.
		// Fail this cohort without leaving an idle HTTP server waiting forever.
		for _, slot := range ready {
			s.finish(slot, err)
		}
	}
	return true
}

func (s *batchScheduler) emit(slot *batchSlot, results []sample.Result, err error) {
	req := slot.request
	if err != nil {
		s.finish(slot, err)
		return
	}
	before := req.output.generated
	responses, done := req.output.consume(results)
	req.responses = append(req.responses, responses...)
	if done {
		s.finish(slot, nil)
	} else if req.output.generated/256 != before/256 {
		mlx.ClearCache()
	}
}

// decode advances the ready requests, grouping equal-length speculative rounds
// for a shared target forward when multiple requests are active.
// Proposals, sampling and rollback still belong to each request. Plain rows keep
// their pipelined forward; no request waits for a peer to choose the same depth.
func (s *batchScheduler) decode(ready []*batchSlot, active int) error {
	// A prefilling or backpressured peer still affects decode costs. Only a
	// request running alone may use and train its isolated depth controller.
	if active == 1 {
		s.drafting.reset()
		slot := ready[0]
		req := slot.request
		results, err := req.decoder.next(req.output.limit - req.output.generated)
		s.emit(slot, results, err)
		return nil
	}
	if s.r.spec == nil {
		return s.decodePlain(ready)
	}
	type group struct {
		slots  []*batchSlot
		rounds []draftRound
	}
	controller := s.drafting.begin(ready)
	var groups []group
	var plain []*batchSlot
	for _, slot := range ready {
		req := slot.request
		d, ok := req.decoder.(*speculativeDecoder)
		limit := 0
		if ok {
			limit = d.s.limit
			if controller != nil && d.s.enabled {
				limit = controller.scheduled
			}
		}
		if !ok || limit == 0 {
			plain = append(plain, slot)
			continue
		}
		// Keep the isolated depth controller's timings free of peer work.
		// Batch depth selection never changes the isolated controller's limit.
		d.s.lastRoundStart = time.Time{}
		d.s.prevDrafts, d.s.roundDrafts = -1, -1
		if d.inner != nil {
			s.drafting.interrupt()
			results, err := d.resume()
			if err == nil {
				d.advance(results[len(results)-1])
			}
			s.emit(slot, results, err)
			continue
		}
		remaining := req.output.limit - req.output.generated
		candidates := d.s.drafter.propose(d.current.Token, min(limit, remaining-1))
		if candidates == nil {
			s.drafting.interrupt()
			plain = append(plain, slot)
			continue
		}
		mlx.AsyncEval(candidates.Arrays()...)
		n := candidates.tokens.Dim(1)
		if n != limit {
			s.drafting.interrupt()
		}
		i := slices.IndexFunc(groups, func(g group) bool { return g.rounds[0].candidates.tokens.Dim(1) == n })
		if i < 0 {
			i = len(groups)
			groups = append(groups, group{})
		}
		groups[i].slots = append(groups[i].slots, slot)
		groups[i].rounds = append(groups[i].rounds, draftRound{d.s, &d.position, d.current, candidates, d.grammar})
	}
	for _, g := range groups {
		outcomes := s.r.acceptDrafts(g.rounds)
		for i, out := range outcomes {
			d := g.slots[i].request.decoder.(*speculativeDecoder)
			if out.err == nil {
				d.s.recordRound(g.rounds[i].candidates.tokens.Dim(1), out.accepted)
				if controller != nil {
					evidence := controller.acc
					if controller.block {
						evidence = d.s.depth.acc
					}
					evidence.observe(out.observed, out.accepted)
				}
				d.advance(out.results[len(out.results)-1])
			}
			s.emit(g.slots[i], out.results, out.err)
		}
	}
	err := s.decodePlain(plain)
	if err != nil {
		s.drafting.reset()
	}
	if controller != nil && !s.drafting.started.IsZero() {
		// Draining a parked pipeline does not execute the scheduled probe.
		// Hold its depth until a complete round consumes it.
		s.drafting.next(controller)
	}
	return err
}

// decodePlain advances pipelined rows together, including requests whose draft
// head is parked or disabled by their sampling options.
func (s *batchScheduler) decodePlain(ready []*batchSlot) error {
	var slots []*batchSlot
	var decoders []*pipelinedDecoder
	for _, slot := range ready {
		var d *pipelinedDecoder
		switch decoder := slot.request.decoder.(type) {
		case *speculativeDecoder:
			d = decoder.pipeline()
		case *pipelinedDecoder:
			d = decoder
		}
		if d.err != nil {
			s.finish(slot, d.err)
			continue
		}
		slots = append(slots, slot)
		decoders = append(decoders, d)
	}
	if len(slots) == 0 {
		return nil
	}
	ids := make([]*mlx.Array, len(slots))
	offsets, lengths := make([]int32, len(slots)), make([]int32, len(slots))
	owners := make([][]cache.Cache, len(slots))
	slotIDs := make([]int, len(slots))
	var layout []any
	for i, d := range decoders {
		ids[i] = d.pending.Token
		offsets[i], lengths[i] = int32(d.position), 1
		owners[i], slotIDs[i] = d.caches, d.slotID
		if len(d.layout) > 0 {
			if layout == nil {
				layout = make([]any, len(slots))
			}
			layout[i] = d.layout[0]
		}
	}
	layers, err := cache.Batch(owners)
	if err != nil {
		return err
	}
	tokens := mlx.Concatenate(ids, 0).Reshape(len(slots), 1)
	b := &batch.Batch{InputIDs: tokens, SeqOffsets: offsets, SeqQueryLens: lengths, Layout: layout}
	hidden, aux := s.r.Model.Forward(b, layers)
	logits := s.r.Model.Unembed(hidden)
	constrained := false
	for i, d := range decoders {
		if d.spec != nil {
			d.spec.committed(tokens.Slice(mlx.Slice(i, i+1), mlx.Slice()), aux.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()), d.position, nil)
		}
		d.position++
		constrained = constrained || s.r.grammarEngine.hasGrammar(d.grammars)
	}
	if constrained {
		rows := make([]*mlx.Array, len(decoders))
		for i, d := range decoders {
			rows[i] = d.constrain(logits.Slice(mlx.Slice(i, i+1), mlx.Slice(), mlx.Slice()), d.pending.Token)
		}
		logits = mlx.Concatenate(rows, 0)
	}
	next := s.r.Sampler.Sample(slotIDs, logits.Squeeze(1))
	arrays := next.Arrays()
	for _, layer := range layers {
		if layer != nil {
			arrays = append(arrays, layer.State()...)
		}
	}
	// Finish compact state copies even if a reader stalls before its next round.
	mlx.AsyncEval(arrays...)
	logutil.Trace("mlx batch decode", "rows", len(slots), "slots", slotIDs, "offsets", offsets)
	for i, slot := range slots {
		d := decoders[i]
		previous := d.pending
		d.pending = sampleRow(next, i)
		d.scope.Attach(d.pending.Arrays()...)
		d.scope.Detach(previous.Arrays()...)
		if spec, ok := slot.request.decoder.(*speculativeDecoder); ok {
			spec.batched(previous)
		}
		s.emit(slot, []sample.Result{previous}, nil)
	}
	return nil
}

func sampleRow(result sample.Result, row int) sample.Result {
	slice := func(a *mlx.Array) *mlx.Array {
		if a == nil {
			return nil
		}
		return a.Slice(mlx.Slice(row, row+1), mlx.Slice())
	}
	return sample.Result{Token: result.Token.Slice(mlx.Slice(row, row+1)), Logprob: slice(result.Logprob), TopTokens: slice(result.TopTokens), TopLogprobs: slice(result.TopLogprobs)}
}

func (s *batchScheduler) admitPending() bool {
	admitted := false
	free := 0
	for _, slot := range s.slots {
		if slot == nil || !slot.busy() {
			free++
		}
	}
	for range free {
		select {
		case request := <-s.r.Requests:
			admitted = s.admit(request) || admitted
		default:
			return admitted
		}
	}
	return admitted
}

// New slots use the last successful observation or its failure fallback.
// Until retention has observed memory, preserve the fixed per-slot allowance.
func (r *Runner) snapshotLimit() int64 {
	if r.snapshotLimitBytes != 0 {
		return r.snapshotLimitBytes
	}
	return maxPagedOutBytes / int64(max(1, r.parallel))
}

func (s *batchScheduler) hasPrefixCache() bool {
	for _, slot := range s.slots {
		if slot != nil && (slot.cache != nil || slot.scoreCache != nil) {
			return true
		}
	}
	return false
}
