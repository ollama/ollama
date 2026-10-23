package mlxrunner

import (
	"context"
	"log/slog"
	"slices"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model/cohere2_moe"
	"github.com/ollama/ollama/mlxrunner/model/gemma4"
	"github.com/ollama/ollama/mlxrunner/model/glimmer"
	"github.com/ollama/ollama/mlxrunner/model/laguna"
	"github.com/ollama/ollama/mlxrunner/model/nemotron_h"
	"github.com/ollama/ollama/mlxrunner/model/qwen3_5"
	"github.com/ollama/ollama/mlxrunner/model/qwen4_exp"
	"github.com/ollama/ollama/mlxrunner/sample"
)

// Only models with independent-row forward and cache support use the batch
// scheduler. Scoring, embedding and other model paths retain serial execution.
func (r *Runner) supportsBatching() bool {
	switch r.Model.(type) {
	case *gemma4.Model, *qwen3_5.Model, *cohere2_moe.Model, *glimmer.Model, *laguna.Model, *nemotron_h.Model, *qwen4_exp.Model:
		return true
	default:
		return false
	}
}

type batchSlot struct {
	id          int
	cache       *prefixCache
	targets     []cache.Cache
	speculation *speculation
	request     *batchRequest
}

type batchRequest struct {
	request   Request
	session   *cacheSession
	media     *requestMedia
	spec      *speculationSession
	prefill   *prefillCursor
	decoder   decoder
	grammar   *grammar
	output    *generationOutput
	responses []CompletionResponse
	finished  bool
}

type batchScheduler struct {
	r           *Runner
	slots       []*batchSlot
	nextPrefill int
}

func (r *Runner) onMLX(ctx context.Context, f func() error) error {
	if r.mlxThread != nil {
		return r.mlxThread.Do(ctx, f)
	}
	return f()
}

func (r *Runner) runBatchRequests(ctx context.Context) error {
	s := &batchScheduler{r: r, slots: make([]*batchSlot, r.parallel)}
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
					slot.cache.close()
				}
			})
			return nil
		})
	})
	slog.Info("MLX concurrent generation", "parallel", r.parallel, "drafting", "when-alone", "attention", "per-request")
	for {
		var request *Request
		var incoming chan Request
		active := false
		for _, slot := range s.slots {
			if slot == nil || slot.request == nil {
				incoming = r.Requests
			} else {
				active = true
			}
		}
		if !active {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case req := <-incoming:
				request = &req
			case req := <-r.EmbedRequests:
				if err := r.onMLX(ctx, func() error { return r.runEmbed(req.Ctx, req) }); err != nil {
					slog.Info("Embedding request terminated", "error", err)
				}
			}
		} else {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case req := <-incoming:
				request = &req
			default:
			}
		}
		progressed := false
		err := r.onMLX(ctx, func() error {
			mlx.Scoped(func() {
				if request != nil {
					if !active {
						mlx.ResetPeakMemory()
					}
					s.admit(*request)
					progressed = true
				}
				worked := s.step()
				progressed = progressed || worked
			})
			// Release unused allocator buffers after the round's temporaries
			// are freed, once all responses have drained.
			for _, slot := range s.slots {
				if slot != nil && slot.request != nil {
					return nil
				}
			}
			mlx.ClearCache()
			return nil
		})
		if err != nil {
			// Dispatch failed before admission; no slot owns this request yet.
			if request != nil {
				request.Grammar.close()
				close(request.Responses)
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

func (s *batchScheduler) admit(request Request) {
	if request.Ctx.Err() != nil {
		request.Grammar.close()
		close(request.Responses)
		return
	}
	best, bestMatch := -1, -1
	keys := effectiveKeyTokens(request.Tokens, request.MediaItems)
	for i, slot := range s.slots {
		if slot != nil && slot.request != nil {
			continue
		}
		matched := 0
		if slot != nil && slot.cache.root != nil {
			_, matched = findBestMatch(slot.cache.root, slot.cache.key(keys))
		}
		if matched > bestMatch {
			best, bestMatch = i, matched
		}
	}
	slot := s.slots[best]
	if slot == nil {
		targets := s.r.Model.NewCaches()
		var draft []cache.Cache
		if s.r.spec != nil {
			draft = newDraftCaches(s.r.spec.draft)
		}
		slot = &batchSlot{id: best, targets: targets, cache: newPrefixCache(slices.Concat(targets, draft))}
		slot.cache.limitBytes = maxPagedOutBytes / int64(len(s.slots))
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
	req.media.close()
	req.request.Grammar.close()
	req.finished = true
	if err != nil {
		req.responses = append(req.responses, CompletionResponse{Error: &api.StatusError{StatusCode: 500, ErrorMessage: err.Error()}})
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
	// At most one prompt chunk per decode round; rotate among prefilling slots.
	for n := range len(s.slots) {
		i := (s.nextPrefill + n) % len(s.slots)
		slot := s.slots[i]
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
			seed := req.prefill.seed()
			promptEval := time.Since(req.prefill.started)
			s.r.Sampler.Add(slot.id, req.request.SamplerOpts, req.request.Tokens)
			if req.spec != nil {
				req.decoder = req.spec.decoder(seed, req.prefill.position, req.grammar)
			} else {
				req.decoder = s.r.pipelinedDecoder(slot.id, nil, slot.targets, seed.ExpandDims(-1), req.prefill.position, req.media.rowLayout(), req.grammar)
			}
			req.output = newGenerationOutput(s.r, req.request, req.session, promptEval)
			req.prefill = nil
		}
		break
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

// decode advances the same pipelined decoders used by the serial path, sharing
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
