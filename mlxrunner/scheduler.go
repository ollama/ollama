package mlxrunner

import (
	"context"
	"log/slog"
	"math"
	"slices"
	"time"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/sample"
)

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
	snapshotWorkspace int64
	snapshotCheck     time.Time
	compressed        int64
	observedMemory    bool
}

func (r *Runner) onMLX(ctx context.Context, f func() error) error {
	if r.mlxThread != nil {
		return r.mlxThread.Do(ctx, f)
	}
	return f()
}

func (r *Runner) runBatchRequests(ctx context.Context) error {
	s := &batchScheduler{r: r, slots: make([]*batchSlot, max(1, r.parallel))}
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
	slog.Info("MLX request scheduler", "parallel", r.parallel, "drafting", "when-alone", "attention", "per-request")
	for {
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
		if !active {
			var idle <-chan time.Time
			if !s.snapshotCheck.IsZero() {
				idle = time.After(time.Second)
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case pending = <-incoming:
			case <-idle:
			}
		} else {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case pending = <-incoming:
			default:
			}
		}
		progressed := false
		err := r.onMLX(ctx, func() error {
			measure := false
			var before, peak int
			mlx.Scoped(func() {
				if pending != nil {
					if _, generation := pending.(request); generation && !active {
						mlx.ResetPeakMemory()
					}
					s.admit(pending)
					progressed = true
				}
				// Coalesce already waiting callers without delaying the first one.
				s.admitPending()
				step := func() {
					worked := s.step()
					progressed = progressed || worked
				}
				measure = s.hasPrefixCache() && mlx.MetalIsAvailable()
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
			s.trimSnapshots(false)
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

func (s *batchScheduler) admit(pending queuedRequest) {
	s.trimSnapshots(true)
	switch req := pending.(type) {
	case request:
		s.admitGeneration(req)
	case *evaluationRequest:
		s.admitEvaluation(req)
	case *speechRequest:
		s.admitSpeech(req)
	default:
		panic("unknown scheduler request type")
	}
	// The first prefix cache is created during admission. Seed the observation
	// before its first forward so that any resulting compression remains visible.
	if s.snapshotCheck.IsZero() {
		s.trimSnapshots(true)
	}
}

func (s *batchScheduler) admitGeneration(request request) {
	s.keepScratch = false
	if err := request.Ctx.Err(); err != nil {
		request.reject(err)
		return
	}
	best, bestMatch := -1, -1
	keys := effectiveKeyTokens(request.Tokens, request.MediaItems)
	for i, slot := range s.slots {
		if slot != nil && slot.busy() {
			continue
		}
		matched := 0
		if slot != nil && slot.cache != nil && slot.cache.root != nil {
			_, matched = findBestMatch(slot.cache.root, slot.cache.key(keys))
		}
		if matched > bestMatch {
			best, bestMatch = i, matched
		}
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
		slot.cache.limitBytes = s.r.snapshotLimit()
		if s.r.spec != nil {
			slot.speculation = newSpeculation(s.r, slot.id, s.r.spec.draft, targets, draft)
			slot.cache.draftLookahead = s.r.cache.draftLookahead
		}
		s.slots[best] = slot
	}
	req := &batchRequest{request: request}
	req.session = slot.cache.begin(request.Tokens, request.MediaItems)
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
	req.session.close()
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
			s.stepEvaluation(slot)
			progressed = true
			break
		}
		if slot == nil || slot.request == nil || slot.request.finished || slot.request.prefill == nil {
			continue
		}
		req := slot.request
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
	progressed = s.decodeSpeech() || progressed
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
		return progressed
	}
	if active == 1 {
		slot := ready[0]
		req := slot.request
		results, err := req.decoder.next(req.output.limit - req.output.generated)
		s.emit(slot, results, err)
		return true
	}
	if err := s.decode(ready); err != nil {
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

// decode advances the same pipelined decoders used for an isolated request, sharing
// one model forward and sample across their rows. A speculative decoder parks
// through its usual pipeline; when alone again its next call resumes drafting.
func (s *batchScheduler) decode(ready []*batchSlot) error {
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

func (s *batchScheduler) admitPending() {
	free := 0
	for _, slot := range s.slots {
		if slot == nil || !slot.busy() {
			free++
		}
	}
	for range free {
		select {
		case request := <-s.r.Requests:
			s.admit(request)
		default:
			return
		}
	}
}

// Other backends keep their current allowance until separately qualified.
func (r *Runner) snapshotLimit() int64 {
	if mlx.MetalIsAvailable() {
		if _, err := discover.GetMemoryHeadroom(); err == nil {
			return math.MaxInt64
		}
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

// Keep reclamation local to this runner. Revisit cross-runner pruning after
// disk paging, with headroom for each model's active KV and workspace;
// fitting both sets of weights alone need not improve throughput.
func (s *batchScheduler) trimSnapshots(admitting bool) {
	if !s.hasPrefixCache() || !mlx.MetalIsAvailable() {
		return
	}
	if !admitting && time.Since(s.snapshotCheck) < time.Second {
		return
	}
	s.snapshotCheck = time.Now()
	var caches, idle []*prefixCache
	var retained, working, next int64
	for _, slot := range s.slots {
		if slot == nil {
			continue
		}
		for _, c := range []*prefixCache{slot.cache, slot.scoreCache} {
			if c == nil {
				continue
			}
			caches = append(caches, c)
			retained += c.pagedOutBytes
			live := int64(c.liveBytes())
			busy := c == slot.activeCache()
			if busy {
				working += live
			} else {
				idle = append(idle, c)
				next = max(next, live)
			}
		}
	}
	// Allow one copy per active row, and one idle row which may be admitted.
	// Restoring history or writing a shared buffer can require a fresh copy.
	working += next
	observation, err := discover.GetMemoryHeadroom()
	if err != nil {
		for _, c := range caches {
			c.limitBytes = maxPagedOutBytes / int64(max(1, s.r.parallel))
			c.enforceEvictionPolicy()
		}
		return
	}
	for _, c := range caches {
		c.limitBytes = math.MaxInt64
	}
	growth := s.observeCompression(observation.Compressed)
	pressure := growth
	pool := int64(mlx.CacheMemory())
	storageBefore := int64(mlx.ActiveMemory()) + pool
	if (growth > 0 || observation.Immediate < 0) && pool > 0 {
		mlx.ClearCache()
		pressure = max(0, pressure-pool)
	}
	// Rising host compression asks each runner to yield cold owned history.
	// Other runners may also respond; this is bounded feedback, not a prediction
	// of which process caused pressure or a guarantee against an initial burst.
	reclaimed := reclaimColdHistory(caches, pressure)
	reserve := s.snapshotWorkspace + working
	available, reusable := observation.Available, int64(mlx.CacheMemory())
	before := mlx.ActiveMemory()
	target := physicalHistoryLimit(observation, storageBefore, int64(before), reusable, reserve)
	trimRetainedHistory(caches, idle, target)
	slog.Log(context.Background(), logutil.LevelTrace, "MLX cache memory", "before_admission", admitting, "available", available, "immediate", observation.Immediate, "compressed", observation.Compressed, "compression_growth", growth, "compression_reclaimed", reclaimed, "reusable", reusable, "workspace", s.snapshotWorkspace, "target", target, "active_before", before, "active_after", mlx.ActiveMemory(), "saved_before", retained, "working_copy", working)
}

func (s *batchScheduler) observeCompression(current int64) int64 {
	var growth int64
	if s.observedMemory {
		growth = max(0, current-s.compressed)
	}
	s.compressed, s.observedMemory = current, true
	return growth
}

// physicalHistoryLimit credits released native allocations as estimated headroom
// against the same host observation. Re-reading Mach counters can return a cached sample
// and charge a physical shortfall twice, unnecessarily discarding the frontier.
func physicalHistoryLimit(memory discover.MemoryHeadroom, before, active, reusable, reserve int64) int64 {
	released := max(0, before-active-reusable)
	target := max(1, active+memory.Available+released+reusable-reserve)
	if immediate := memory.Immediate + released; immediate < 0 {
		target = min(target, max(1, active+immediate))
	}
	return target
}
