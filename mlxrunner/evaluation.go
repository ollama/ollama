package mlxrunner

import (
	"context"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

type activeEvaluation struct {
	request *evaluationRequest
	state   *evaluationState
}

func (r *Runner) scheduleEvaluation(ctx context.Context, open func(*batchSlot) (*evaluationState, error)) error {
	request := &evaluationRequest{ctx: ctx, open: open, done: make(chan error, 1)}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case r.Requests <- request:
	}
	// Once admitted, wait for cleanup even on cancellation. An evaluation may still
	// be writing CPU results captured by the caller while its GPU step ends.
	return <-request.done
}

func (s *batchScheduler) admitEvaluation(request *evaluationRequest) {
	if err := request.ctx.Err(); err != nil {
		request.reject(err)
		return
	}
	for i, slot := range s.slots {
		if slot != nil && slot.busy() {
			continue
		}
		if slot == nil {
			slot = &batchSlot{id: i}
			s.slots[i] = slot
		}
		p, err := request.open(slot)
		if err != nil {
			request.done <- err
			return
		}
		slot.evaluation = &activeEvaluation{request: request, state: p}
		return
	}
	panic("scheduler admitted evaluation request without a free slot")
}

func (s *batchScheduler) finishEvaluation(slot *batchSlot, err error) {
	p := slot.evaluation
	s.keepScratch = p.state.keepScratch && err == nil
	p.state.close()
	slot.evaluation = nil
	p.request.done <- err
}

func (s *batchScheduler) stepEvaluation(slot *batchSlot) {
	if rows, n := s.cachedForwardSlots(slot); len(rows) > 1 {
		s.cachedForward(rows, n)
		return
	}
	if rows := s.wholeForwardSlots(slot); len(rows) > 1 {
		s.wholeForward(rows)
		return
	}
	done, err := slot.evaluation.state.step(slot.evaluation.request.ctx)
	if done || err != nil {
		s.finishEvaluation(slot, err)
	}
}

// forwardHidden evaluates the input and consumes its hidden states on the MLX
// worker. No MLX array crosses back to the handler. Cacheless models consume whole rows;
// causal models retain prefix reuse and yield between prompt chunks.
func (r *Runner) forwardHidden(ctx context.Context, input *model.PreparedRequest, segments []model.Segment, consume func(*mlx.Array) error) (int, error) {
	items, err := bindItems(input, segments)
	if err != nil {
		return 0, err
	}
	cached := 0
	err = r.scheduleEvaluation(ctx, func(slot *batchSlot) (*evaluationState, error) {
		p := r.openEvaluation(slot, input, items, consume)
		p.keepScratch = true
		cached = p.position
		return p, nil
	})
	return cached, err
}

func (r *Runner) openEvaluation(slot *batchSlot, input *model.PreparedRequest, items []mediaItem, consume func(*mlx.Array) error) *evaluationState {
	p := &evaluationState{r: r, input: input, consume: consume}
	p.media = r.openMedia(request{Tokens: input.Tokens, MediaItems: items, Layout: input.Layout})
	if slot.scoreCache == nil {
		cs := r.Model.NewCaches()
		if len(cs) > 0 {
			slot.scoreHidden = cache.NewHiddenCache()
			slot.scoreCache = newPrefixCache(append(cs, slot.scoreHidden))
			slot.scoreCache.limitBytes = maxPagedOutBytes / int64(max(1, r.parallel))
		}
	}
	if slot.scoreCache != nil {
		p.session = slot.scoreCache.beginScore(input.Tokens, items)
		p.session.schedulePrefillSnapshots(scoreSnapshots(len(input.Tokens), input.ScorePrefix))
		p.hidden = slot.scoreHidden
		p.position = len(input.Tokens) - len(p.session.remaining)
		p.media.free(p.position)
	}
	return p
}

type evaluationState struct {
	r           *Runner
	input       *model.PreparedRequest
	consume     func(*mlx.Array) error
	session     *cacheSession
	hidden      *cache.HiddenCache
	media       *requestMedia
	position    int
	keepScratch bool
}

func (p *evaluationState) step(ctx context.Context) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if p.session == nil {
		b := p.inputBatch(0, len(p.input.Tokens))
		out := mlx.ScopedArrays(func() []*mlx.Array {
			h, _ := p.r.Model.Forward(b, nil)
			return []*mlx.Array{h}
		})[0]
		return true, p.consume(out)
	}
	if p.position < len(p.input.Tokens) {
		n := p.media.extendChunk(p.position, min(prefillChunkSize(), len(p.input.Tokens)-p.position), len(p.input.Tokens))
		mlx.Scoped(func() {
			h, _ := p.r.Model.Forward(p.inputBatch(p.position, n), cache.Layers(p.session.caches[:len(p.session.caches)-1]))
			p.hidden.Append(h)
		})
		var state []*mlx.Array
		for _, c := range p.session.caches {
			if c != nil {
				state = append(state, c.State()...)
			}
		}
		mlx.Eval(state...)
		p.position += n
		p.media.free(p.position)
	}
	if p.position < len(p.input.Tokens) {
		return false, nil
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	return true, p.consume(p.hidden.State()[0])
}

func (p *evaluationState) inputBatch(offset, n int) *batch.Batch {
	return &batch.Batch{
		InputIDs:   mlx.FromValues(p.input.Tokens[offset:offset+n], 1, n),
		SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{int32(n)},
		Media: p.media.batchMedia(offset, n), Layout: p.media.rowLayout(),
	}
}

func (p *evaluationState) close() {
	if p.session != nil {
		p.session.close()
	}
	p.media.close()
}

// wholeForward evaluates compatible cacheless rows together. Each evaluation gets
// only its real tokens; padding never reaches pooling or a decision head.
func (s *batchScheduler) wholeForward(slots []*batchSlot) {
	length := 0
	for _, slot := range slots {
		p := slot.evaluation.state
		length = max(length, len(p.input.Tokens))
	}
	logutil.Trace("mlx whole forward batch", "rows", len(slots), "tokens", length)
	b := forwardBatch(slots, length)
	hidden := mlx.ScopedArrays(func() []*mlx.Array {
		h, _ := s.r.Model.Forward(b, nil)
		return []*mlx.Array{h}
	})[0]
	for i, slot := range slots {
		p := slot.evaluation.state
		err := slot.evaluation.request.ctx.Err()
		if err == nil {
			err = p.consume(hidden.Slice(mlx.Slice(i, i+1), mlx.Slice(0, len(p.input.Tokens)), mlx.Slice()))
		}
		s.finishEvaluation(slot, err)
	}
}

// Whole rows cannot be chunked without changing bidirectional attention. Keep
// padding plus real tokens within one prompt chunk when combining requests.
func (s *batchScheduler) wholeForwardSlots(first *batchSlot) []*batchSlot {
	p := first.evaluation.state
	if p.session != nil {
		return nil
	}
	slots := []*batchSlot{first}
	length := len(p.input.Tokens)
	for _, slot := range s.slots {
		if slot == nil || slot == first || slot.evaluation == nil || slot.evaluation.request.ctx.Err() != nil {
			continue
		}
		other := slot.evaluation.state
		if other.session != nil {
			continue
		}
		n := max(length, len(other.input.Tokens))
		if n*(len(slots)+1) <= prefillChunkSize() {
			slots = append(slots, slot)
			length = n
		}
	}
	return slots
}

// forwardBatch preserves each request's absolute positions and media identity.
// Cacheless forwards may pad; cached forwards pass a shared unpadded length.
func forwardBatch(slots []*batchSlot, length int) *batch.Batch {
	tokens := make([]int32, len(slots)*length)
	b := &batch.Batch{SeqOffsets: make([]int32, len(slots)), SeqQueryLens: make([]int32, len(slots)), Layout: make([]any, len(slots))}
	for i, slot := range slots {
		p := slot.evaluation.state
		n := min(length, len(p.input.Tokens)-p.position)
		copy(tokens[i*length:], p.input.Tokens[p.position:p.position+n])
		b.SeqOffsets[i], b.SeqQueryLens[i] = int32(p.position), int32(n)
		b.Layout[i] = p.input.Layout
		for _, item := range p.media.batchMedia(p.position, n) {
			item.Seq = i
			b.Media = append(b.Media, item)
		}
	}
	b.InputIDs = mlx.FromValues(tokens, len(slots), length)
	return b
}

func (s *batchScheduler) cachedForwardSlots(first *batchSlot) ([]*batchSlot, int) {
	p := first.evaluation.state
	if p.session == nil || p.position == len(p.input.Tokens) {
		return nil, 0
	}
	slots := []*batchSlot{first}
	n := min(prefillChunkSize(), len(p.input.Tokens)-p.position)
	for _, slot := range s.slots {
		if slot == nil || slot == first || slot.evaluation == nil || slot.evaluation.request.ctx.Err() != nil {
			continue
		}
		other := slot.evaluation.state
		if other.session == nil || other.position == len(other.input.Tokens) {
			continue
		}
		candidate := min(n, len(other.input.Tokens)-other.position, prefillChunkSize()/(len(slots)+1))
		rows := append(slots, slot)
		// Reducing a common chunk can expose a different media boundary in
		// another row. Stop only when all rows accept the same length.
		for candidate > 0 {
			before := candidate
			for _, row := range rows {
				p := row.evaluation.state
				boundary := p.media.extendChunk(p.position, candidate, len(p.input.Tokens))
				if boundary > candidate {
					candidate = 0 // An atomic expansion cannot fit this cohort.
					break
				}
				candidate = min(candidate, boundary)
			}
			if candidate == before {
				break
			}
		}
		if candidate > 0 {
			slots, n = rows, candidate
		}
	}
	return slots, n
}

func (s *batchScheduler) cachedForward(slots []*batchSlot, n int) {
	logutil.Trace("mlx cached forward batch", "rows", len(slots), "tokens", n)
	rows := make([][]cache.Cache, len(slots))
	for i, slot := range slots {
		cs := slot.evaluation.state.session.caches
		rows[i] = cs[:len(cs)-1] // Hidden outputs are appended after the forward.
	}
	layers, err := cache.Batch(rows)
	if err != nil {
		for _, slot := range slots {
			s.finishEvaluation(slot, err)
		}
		return
	}
	b := forwardBatch(slots, n)
	hidden := mlx.ScopedArrays(func() []*mlx.Array {
		h, _ := s.r.Model.Forward(b, layers)
		return []*mlx.Array{h}
	})[0]
	var state []*mlx.Array
	for i, slot := range slots {
		p := slot.evaluation.state
		// A completed request must not retain its peers' hidden allocations.
		p.hidden.Append(hidden.TakeAxis(mlx.FromValues([]int32{int32(i)}, 1), 0))
		for _, c := range p.session.caches {
			if c != nil {
				state = append(state, c.State()...)
			}
		}
	}
	mlx.Eval(state...)
	for _, slot := range slots {
		p := slot.evaluation.state
		p.position += n
		p.media.free(p.position)
		err := slot.evaluation.request.ctx.Err()
		if err == nil && p.position == len(p.input.Tokens) {
			err = p.consume(p.hidden.State()[0])
		}
		if err != nil || p.position == len(p.input.Tokens) {
			s.finishEvaluation(slot, err)
		}
	}
}
