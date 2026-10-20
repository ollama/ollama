package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlx/mlxthread"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/sample"
)

type schedulingModel struct {
	fakeMTPModel
	batches []int
}

func (m *schedulingModel) NewCaches() []cache.Cache { return []cache.Cache{cache.NewKVCache()} }
func (m *schedulingModel) Forward(b *batch.Batch, cs []cache.Layer) (*mlx.Array, *mlx.Array) {
	B, L := b.InputIDs.Dim(0), b.InputIDs.Dim(1)
	m.batches = append(m.batches, B)
	m.layouts = append(m.layouts, b.Layout)
	kv := b.InputIDs.AsType(mlx.DTypeFloat32).Reshape(B, 1, L, 1)
	cs[0].(cache.Attention).Update(b, kv, kv)
	ids := b.InputIDs.Ints()
	if m.predict != nil {
		for i, id := range ids {
			ids[i] = m.predict[id]
		}
	}
	out := oneHotLogits(ids).Reshape(B, L, mtpTestVocab)
	return out, out
}

func TestBatchSchedulerLifecycle(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096), parallel: 2}
		r.cache = newPrefixCache(m.NewCaches())
		r.spec = newSpeculation(r, pipelineSlot, &fakeMTPDraft{}, r.cache.caches, nil)
		defer r.cache.close()
		s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
		defer func() {
			for _, slot := range s.slots {
				if slot != nil {
					slot.cache.close()
				}
			}
			r.Sampler.Free()
		}()
		request := func(ctx context.Context, id int32, n int) Request {
			return Request{
				Ctx: ctx, Tokens: []int32{id, id, id}, Responses: make(chan CompletionResponse, 1),
				CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: n}}, SamplerOpts: sample.Options{Temperature: 0, Logprobs: true, TopLogprobs: 2},
			}
		}
		step := func() {
			t.Helper()
			mlx.Scoped(func() {
				s.step()
			})
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		a := request(ctx, 1, 20)
		mlx.Scoped(func() { s.admit(a) })
		step()
		step()
		step()
		// A has started streaming before the second request arrives.
		if len(a.Responses) == 0 {
			t.Fatal("first request has not emitted")
		}
		b := request(context.Background(), 2, 4)
		b.Tokens = slices.Repeat([]int32{2}, prefillChunkSize()+3)
		mlx.Scoped(func() { s.admit(b) })
		text := ""
		var final CompletionResponse
		for i := 0; i < 30 && !final.Done; i++ {
			step()
			select {
			case response := <-b.Responses:
				if response.Error != nil {
					t.Fatal(response.Error)
				}
				text += response.Content
				if response.Done {
					final = response
				}
			default:
			}
			// Let A decode while B prefills multiple chunks, then stop reading A.
			if i < 3 && len(a.Responses) > 0 {
				<-a.Responses
			}
		}
		if !final.Done || final.EvalCount != 4 {
			t.Fatalf("second request did not finish: %+v", final)
		}
		if want := strings.Repeat(tok.Decode([]int32{2}), 4); text != want {
			t.Fatalf("second response %q, want %q", text, want)
		}
		batched := false
		for _, n := range m.batches {
			batched = batched || n > 1
		}
		if !batched {
			t.Fatalf("never dispatched a shared forward: %v", m.batches)
		}
		if s.slots[0].request == nil {
			t.Fatal("slow first request unexpectedly finished")
		}
		cancel()
		step()
		// Reuse the cancelled slot for a different conversation; no token history leaks.
		c := request(context.Background(), 3, 2)
		mlx.Scoped(func() { s.admit(c) })
		text = ""
		final = CompletionResponse{}
		for i := 0; i < 20 && !final.Done; i++ {
			step()
			select {
			case response := <-c.Responses:
				text += response.Content
				if response.Done {
					final = response
				}
				if response.Error != nil {
					t.Fatal(response.Error)
				}
			default:
			}
		}
		if !final.Done || text != strings.Repeat(tok.Decode([]int32{3}), 2) {
			t.Fatalf("reused slot response %q: %+v", text, final)
		}
		step()
		// An idle slot still retains ordinary prefix reuse.
		again := request(context.Background(), 2, 1)
		mlx.Scoped(func() { s.admit(again) })
		final = CompletionResponse{}
		for i := 0; i < 20 && !final.Done; i++ {
			step()
			select {
			case response := <-again.Responses:
				if response.Done {
					final = response
				}
			default:
			}
		}
		if !final.Done || final.PromptEvalCachedCount == nil || *final.PromptEvalCachedCount < 2 {
			t.Fatalf("repeat lost cached prompt: %+v", final)
		}
		step()
	})
}

func TestBatchSchedulerWaitingGrammarDoesNotBlockDecode(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096), parallel: 2}
		s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		a := Request{Ctx: ctx, Tokens: []int32{1}, Responses: make(chan CompletionResponse, 8), Grammar: &grammarCompilation{done: make(chan struct{})}, CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 2}}}
		b := Request{Ctx: context.Background(), Tokens: []int32{2}, Responses: make(chan CompletionResponse, 8), CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 2}}}
		mlx.Scoped(func() { s.admit(a); s.admit(b) })
		for range 10 {
			mlx.Scoped(func() {
				s.step()
			})
		}
		done := false
		for len(b.Responses) > 0 {
			done = done || (<-b.Responses).Done
		}
		if !done {
			t.Fatal("grammar compilation stalled another request")
		}
		cancel()
		mlx.Scoped(func() { s.step() })
		for _, slot := range s.slots {
			slot.cache.close()
		}
		r.Sampler.Free()
	})
}

// This head uses ordinary KV storage so the test checks real snapshot rollback
// and the draft's look-ahead pairing across repeated mode transitions.
type schedulingDraft struct {
	fakeMTPDraft
}

func (d *schedulingDraft) NewCaches() []cache.Cache { return []cache.Cache{cache.NewKVCache()} }
func (d *schedulingDraft) Forward(b *batch.Batch, _, cs []cache.Layer) (*mlx.Array, *mlx.Array) {
	L := b.InputIDs.Dim(1)
	kv := b.InputIDs.AsType(mlx.DTypeFloat32).Reshape(1, 1, L, 1)
	cs[0].(cache.Attention).Update(b, kv, kv)
	ids := b.InputIDs.Ints()
	for i, id := range ids {
		ids[i] = d.predict[id]
	}
	out := oneHotLogits(ids)
	return out, out
}

func TestBatchSchedulerDraftHandoffs(t *testing.T) {
	for _, tc := range []struct{ cached, reject bool }{{false, false}, {false, true}, {true, false}, {true, true}} {
		t.Run(fmt.Sprintf("cached=%t/reject=%t", tc.cached, tc.reject), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				predict := map[int32]int32{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 1}
				draftPredict := maps.Clone(predict)
				if tc.reject {
					draftPredict[3] = 1
				}
				tok := newTestTokenizer(t, []int32{7})
				m := &schedulingModel{fakeMTPModel: fakeMTPModel{predict: predict, tok: tok}}
				r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096), parallel: 2}
				r.cache = newPrefixCache(m.NewCaches())
				var draft model.DraftModel = &fakeMTPDraft{predict: draftPredict}
				if tc.cached {
					draft = &schedulingDraft{fakeMTPDraft: fakeMTPDraft{predict: draftPredict}}
				}
				r.spec = newSpeculation(r, pipelineSlot, draft, r.cache.caches, nil)
				s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
				defer func() {
					for _, slot := range s.slots {
						if slot != nil {
							if slot.request != nil {
								s.finish(slot, context.Canceled)
							}
							slot.cache.close()
						}
					}
					r.cache.close()
					r.Sampler.Free()
				}()
				request := func(ctx context.Context, seed int32, budget int) Request {
					return Request{
						Ctx: ctx, Tokens: []int32{0, 0, seed}, Responses: make(chan CompletionResponse, 128),
						CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: budget}},
						SamplerOpts:       sample.Options{Temperature: 0.7, UseSeed: true, Seed: int(seed), RepeatPenalty: 1.1, RepeatLastN: 8},
					}
				}
				step := func() {
					t.Helper()
					mlx.Scoped(func() {
						s.step()
						for _, slot := range s.slots {
							if slot == nil || slot.request == nil {
								continue
							}
							req := slot.request
							// Every target write follows the request's own token chain,
							// even across speculative rejection, join, leave and cancel.
							want := slices.Clone(req.request.Tokens)
							for len(want) < slot.targets[0].Offset() {
								want = append(want, predict[want[len(want)-1]])
							}
							if state := slot.targets[0].State(); len(state) > 0 {
								for i, token := range state[0].Floats() {
									if int32(token) != want[i] {
										t.Fatalf("slot %d target position %d: got %v, want %d", slot.id, i, token, want[i])
									}
								}
							}
						}
					})
				}
				a := request(context.Background(), 1, 20)
				mlx.Scoped(func() { s.admit(a); pinDraftLimit(s.slots[0].request.spec, 3) })
				first := s.slots[0].request
				step()
				step()
				if first.spec.stats.drafted == 0 {
					t.Fatal("first request did not draft while alone")
				}
				before := first.spec.stats.drafted
				b := request(context.Background(), 4, 60)
				mlx.Scoped(func() { s.admit(b); pinDraftLimit(s.slots[1].request.spec, 3) })
				second := s.slots[1].request
				for range 30 {
					step()
					if first.finished {
						break
					}
				}
				if !first.finished || second.finished || first.spec.stats.drafted != before {
					t.Fatalf("unexpected overlapping phase: first done=%t second done=%t drafts=%d->%d", first.finished, second.finished, before, first.spec.stats.drafted)
				}
				if !slices.Contains(m.batches, 2) {
					t.Fatal("requests never shared a forward")
				}
				for range 3 {
					step()
				}
				if second.spec.stats.drafted == 0 {
					t.Fatal("slot one did not resume drafting after slot zero finished")
				}
				before = second.spec.stats.drafted
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				c := request(ctx, 6, 30)
				mlx.Scoped(func() { s.admit(c); pinDraftLimit(s.slots[0].request.spec, 3) })
				third := s.slots[0].request
				step()
				step()
				if second.spec.stats.drafted != before {
					t.Fatal("second request kept drafting while third was active")
				}
				cancel()
				for range 70 {
					step()
					if second.finished {
						break
					}
				}
				step() // deliver final responses
				if !second.finished || !third.finished || second.spec.stats.drafted <= before {
					t.Fatal("cancellation did not let the survivor resume and finish")
				}
				for _, req := range []Request{a, b} {
					got, final := collectResponses(req.Responses)
					var want strings.Builder
					next := req.Tokens[len(req.Tokens)-1]
					for range req.Options.NumPredict {
						next = predict[next]
						want.WriteString(tok.Decode([]int32{next}))
					}
					if !final.Done || final.EvalCount != req.Options.NumPredict || got != want.String() {
						t.Fatalf("seed %d result %q (%+v), want %q", req.Tokens[2], got, final, want.String())
					}
				}
				// Closing a completed or cancelled request levels its look-ahead
				// draft cache with the target before prefix snapshots are retained.
				for _, slot := range s.slots {
					if !tc.cached {
						continue
					}
					target, draft := slot.targets[0], slot.speculation.draftKV[0]
					if target.Offset() != draft.Offset() {
						t.Fatalf("slot %d target/draft offsets %d/%d", slot.id, target.Offset(), draft.Offset())
					}
				}
			})
		})
	}
}

func TestBatchSchedulerCancelledAdmission(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s := &batchScheduler{slots: make([]*batchSlot, 2)}
	responses := make(chan CompletionResponse, 1)
	s.admit(Request{Ctx: ctx, Responses: responses})
	if _, open := <-responses; open || s.slots[0] != nil || s.slots[1] != nil {
		t.Fatal("cancelled request opened a cache or left its response channel open")
	}
}

func TestBatchSchedulerLayouts(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
		defer r.Sampler.Free()
		s := &batchScheduler{r: r}
		// Distinct opaque layouts stand for different image position tables. The
		// middle request is text-only; row order changes as clients join and leave.
		layouts := []any{"image-a", nil, "image-b"}
		for i, layout := range layouts {
			cs := m.NewCaches()
			defer cs[0].Free()
			r.Sampler.Add(i, sample.Options{Temperature: 0}, []int32{int32(i + 1)})
			var rowLayout []any
			if layout != nil {
				rowLayout = []any{layout}
			}
			d := r.pipelinedDecoder(i, nil, cs, mlx.FromValues([]int32{int32(i + 1)}, 1, 1), 0, rowLayout, nil)
			defer d.close()
			req := Request{CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 100}}}
			s.slots = append(s.slots, &batchSlot{id: i, targets: cs, request: &batchRequest{decoder: d, output: newGenerationOutput(r, req, &cacheSession{}, 0)}})
		}
		for _, rows := range [][]int{{0, 1, 2}, {2, 0}, {1, 2}, {1}} {
			var ready []*batchSlot
			var want []any
			for _, row := range rows {
				ready = append(ready, s.slots[row])
				want = append(want, layouts[row])
			}
			if err := s.decode(ready); err != nil {
				t.Fatal(err)
			}
			got := m.layouts[len(m.layouts)-1]
			if len(rows) == 1 && rows[0] == 1 {
				want = nil
			}
			if !slices.Equal(got, want) {
				t.Fatalf("rows %v layout = %v, want %v", rows, got, want)
			}
		}
	})
}

func TestBatchSchedulerCacheErrorFinishesCohort(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096), parallel: 2}
		defer r.Sampler.Free()
		s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
		// Both are valid attention caches in isolation, but cannot be combined as
		// one layer. Exercise the scheduler's error delivery and subsequent reuse.
		for i, c := range []cache.Cache{cache.NewKVCache(), cache.NewRotatingKVCache(8)} {
			cs := []cache.Cache{c}
			s.slots[i] = &batchSlot{id: i, targets: cs, cache: newPrefixCache(cs)}
			defer s.slots[i].cache.close()
		}
		requests := make([]Request, 2)
		for i := range requests {
			requests[i] = Request{Ctx: context.Background(), Tokens: []int32{int32(i + 1)}, Responses: make(chan CompletionResponse, 8), CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 20}}}
			mlx.Scoped(func() { s.admit(requests[i]) })
		}
		for range 8 {
			mlx.Scoped(func() { s.step() })
		}
		for i, req := range requests {
			sawError, closed := false, false
			for !closed {
				select {
				case response, ok := <-req.Responses:
					closed = !ok
					if response.Error != nil {
						sawError = strings.Contains(response.Error.Error(), "incompatible")
					}
				default:
					t.Fatalf("row %d left hanging after cache error", i)
				}
			}
			if !sawError || s.slots[i].request != nil {
				t.Fatalf("row %d did not finish with cache error", i)
			}
		}
		again := Request{Ctx: context.Background(), Tokens: []int32{3}, Responses: make(chan CompletionResponse, 8), CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 2}}}
		mlx.Scoped(func() { s.admit(again) })
		for range 8 {
			mlx.Scoped(func() { s.step() })
		}
		var final CompletionResponse
		for response := range again.Responses {
			if response.Error != nil {
				t.Fatal(response.Error)
			}
			if response.Done {
				final = response
			}
		}
		if !final.Done || final.EvalCount != 2 {
			t.Fatalf("request after cohort failure: %+v", final)
		}
	})
}

// A request removed from admission still belongs to the scheduler if the MLX
// worker rejects dispatch. Its HTTP reader must not wait forever for a response.
func TestBatchSchedulerRejectedDispatch(t *testing.T) {
	worker, err := mlxthread.Start("batch-stopped", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := worker.Stop(context.Background(), nil); err != nil {
		t.Fatal(err)
	}
	r := &Runner{parallel: 2, Requests: make(chan Request), mlxThread: worker}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- r.runBatchRequests(ctx) }()
	req := Request{Ctx: ctx, Responses: make(chan CompletionResponse), Grammar: &grammarCompilation{done: make(chan struct{})}}
	select {
	case r.Requests <- req:
	case <-ctx.Done():
		t.Fatal("scheduler did not accept the request")
	}
	select {
	case err := <-done:
		if !errors.Is(err, mlxthread.ErrStopped) {
			t.Fatalf("scheduler exit = %v", err)
		}
	case <-ctx.Done():
		t.Fatal("scheduler did not stop")
	}
	select {
	case _, open := <-req.Responses:
		if open {
			t.Fatal("unexpected response after rejected dispatch")
		}
	default:
		t.Fatal("request left waiting after rejected dispatch")
	}
	if !req.Grammar.abandoned {
		t.Fatal("request grammar not closed")
	}
}

func TestBatchSchedulerShutdownWithBlockedReaders(t *testing.T) {
	worker := mlxtest.Worker(t)
	r := &Runner{parallel: 2, Requests: make(chan Request), mlxThread: worker}
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		r.Model = &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}
		r.Tokenizer = tok
		r.Sampler = sample.New(4096)
	})
	defer func() {
		_ = worker.Do(context.Background(), func() error { r.Sampler.Free(); return nil })
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- r.runBatchRequests(ctx) }()
	responses := []chan CompletionResponse{make(chan CompletionResponse, 1), make(chan CompletionResponse, 1)}
	for i, ch := range responses {
		req := Request{Ctx: context.Background(), Tokens: []int32{int32(i + 1)}, Responses: ch, CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 1000}}}
		select {
		case r.Requests <- req:
		case <-ctx.Done():
			t.Fatal("scheduler did not accept request", i)
		}
		select {
		case <-ch:
		case <-ctx.Done():
			t.Fatal("request did not begin streaming", i)
		}
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("scheduler exit = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("shutdown stalled with blocked readers")
	}
	for i, ch := range responses {
	drain:
		for {
			select {
			case _, open := <-ch:
				if !open {
					break drain
				}
			default:
				t.Fatal("shutdown left response channel open", i)
			}
		}
	}
}

type recurrentSchedulingModel struct{ fakeMTPModel }

func (m *recurrentSchedulingModel) NewCaches() []cache.Cache {
	return []cache.Cache{cache.NewRecurrentCache(2, 3, 16, 128, 128)}
}

func (m *recurrentSchedulingModel) Forward(b *batch.Batch, cs []cache.Layer) (*mlx.Array, *mlx.Array) {
	c := cs[0].(cache.Recurrent)
	h := c.Get(b, mlx.DTypeFloat32)
	delta := mlx.AddScalar(h.DeltaState(), 1)
	c.Put(b, []*mlx.Array{h.ConvState()}, []*mlx.Array{delta})
	out := oneHotLogits(b.InputIDs.Ints()).Reshape(b.InputIDs.Dim(0), b.InputIDs.Dim(1), mtpTestVocab)
	// Like a recurrent kernel, producing hidden state evaluates the shared
	// state allocation, but need not evaluate the cache's row copies.
	out = out.Add(mlx.Sum(delta.Reshape(-1), 0, false).Multiply(mlx.FromValue(float32(0))))
	return out, out
}

func TestBatchSchedulerMaterializesParkedState(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		tok := newTestTokenizer(t, []int32{7})
		m := &recurrentSchedulingModel{fakeMTPModel{tok: tok}}
		r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096)}
		defer r.Sampler.Free()
		s := &batchScheduler{r: r}
		defer func() {
			for _, slot := range s.slots {
				slot.request.decoder.close()
				slot.targets[0].Free()
			}
		}()
		mlx.DefaultStream().Synchronize()
		before := mlx.ActiveMemory()
		mlx.Scoped(func() {
			for id := range 4 {
				cs := m.NewCaches()
				r.Sampler.Add(id, sample.Options{Temperature: 0}, nil)
				d := r.pipelinedDecoder(id, nil, cs, mlx.FromValues([]int32{int32(id + 1)}, 1, 1), 0, nil, nil)
				req := Request{CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 100}}}
				s.slots = append(s.slots, &batchSlot{id: id, targets: cs, request: &batchRequest{decoder: d, output: newGenerationOutput(r, req, &cacheSession{}, 0)}})
			}
			if err := s.decode(s.slots); err != nil {
				t.Fatal(err)
			}
		})
		// Leave one reader parked without another forward or explicit State
		// evaluation. Closing peers must not leave their state pinned by it.
		for _, slot := range s.slots[1:] {
			slot.request.decoder.close()
			slot.targets[0].Free()
			r.Sampler.Remove(slot.id)
		}
		s.slots = s.slots[:1]
		mlx.DefaultStream().Synchronize()
		rowBytes := s.slots[0].targets[0].State()[1].NumBytes()
		if held := mlx.ActiveMemory() - before; held >= 2*rowBytes {
			t.Fatalf("parked row retained %d bytes, want one %d-byte state plus small sampling buffers", held, rowBytes)
		}
	})
}
