package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	sampler "github.com/ollama/ollama/mlxrunner/sample"
)

// probeTestRunner starts one plain round away from a backed-off probe. The
// scripted model always accepts, but the controller's prior observations favor
// plain decoding so an unconsumed probe cannot be replaced by its selection.
func probeTestRunner(t *mlxtest.T, opts sampler.Options) (*Runner, *fakeMTPDraft, *snapshotTracker) {
	t.Helper()
	predict := map[int32]int32{1: 2, 2: 3, 3: 1}
	r := mtpTestRunner(t, predict, []int32{7}, opts)
	caches, snapshots := newMTPTestCaches(1)
	draft := &fakeMTPDraft{predict: predict}
	r.cache.caches = caches
	r.spec = newSpeculation(r, pipelineSlot, draft, caches, nil)
	r.spec.depth = probeTestDepth()
	return r, draft, snapshots
}

func probeTestDepth() *depthController {
	c := newDepthController()
	c.drafterLimit = 1
	c.cost.observe(0, time.Millisecond)
	c.cost.observe(1, 10*time.Millisecond)
	for range 10 {
		c.acc.observe(1, 0)
	}
	c.probeInterval = depthProbeIntervalMax
	c.probeSince = depthProbeIntervalMax - 1
	return c
}

// The first request decodes once while its peer prefills. Their budgets leave
// both with remaining tokens when the shared controller schedules its probe.
func probeTestScheduler(t *mlxtest.T, ctx context.Context, remaining int) (*batchScheduler, *schedulingModel, *depthController) {
	t.Helper()
	tok := newTestTokenizer(t, []int32{7})
	predict := map[int32]int32{1: 1, 2: 2}
	m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok, predict: predict}}
	r := &Runner{Model: m, Tokenizer: tok, Sampler: sampler.New(4096), parallel: 2}
	r.cache = newPrefixCache(m.NewCaches())
	r.spec = newSpeculation(r, pipelineSlot, &fakeMTPDraft{predict: predict}, r.cache.caches, nil)
	c := probeTestDepth()
	s := &batchScheduler{r: r, slots: make([]*batchSlot, 2), drafting: batchDrafting{
		depths: map[draftBatchSize]*depthController{{rows: 2, drafting: 2}: c},
	}}
	t.Cleanup(func() {
		for _, slot := range s.slots {
			if slot == nil {
				continue
			}
			if slot.request != nil {
				s.finish(slot, context.Canceled)
			}
			slot.cache.close()
		}
		r.cache.close()
		r.Sampler.Free()
	})
	mlx.Scoped(func() {
		for i := range 2 {
			requestCtx := context.Background()
			if i == 0 {
				requestCtx = ctx
			}
			s.admit(request{
				Ctx: requestCtx, Tokens: []int32{int32(i + 1), int32(i + 1)}, Responses: make(chan CompletionResponse, 32),
				CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: remaining + 2 - i}},
			})
		}
	})
	for range 2 {
		mlx.Scoped(func() { s.step() })
	}
	if c.scheduled != 1 || !c.probed {
		t.Fatalf("shared controller did not schedule a probe: depth=%d probed=%v", c.scheduled, c.probed)
	}
	for _, slot := range s.slots {
		req := slot.request
		if req == nil || req.finished || req.output.limit-req.output.generated != remaining {
			t.Fatalf("slot %d did not reach the probe with %d tokens remaining", slot.id, remaining)
		}
	}
	return s, m, c
}

func TestBatchSchedulerProbeSurvivesRequestBudget(t *testing.T) {
	for _, remaining := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("remaining%d", remaining), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				s, m, c := probeTestScheduler(t, context.Background(), remaining)
				requests := []request{s.slots[0].request.request, s.slots[1].request.request}
				start := len(m.forwards)
				for range 4 {
					mlx.Scoped(func() { s.step() })
				}
				shared := false
				for i := start; i < len(m.forwards); i++ {
					shared = shared || m.batches[i] == 2 && m.forwards[i].n > 1
				}
				if shared != (remaining == 3) || c.probed != (remaining < 3) {
					t.Fatalf("remaining %d: shared verification=%v probe pending=%v", remaining, shared, c.probed)
				}
				for i, req := range requests {
					if s.slots[i].request != nil {
						t.Fatalf("slot %d did not finish within its token budget", i)
					}
					got, final := collectResponses(req.Responses)
					want := strings.Repeat(s.r.Tokenizer.Decode(req.Tokens[:1]), req.Options.NumPredict)
					if !final.Done || final.EvalCount != req.Options.NumPredict || got != want {
						t.Fatalf("slot %d output=%q final=%+v, want %q", i, got, final, want)
					}
				}
				if remaining == 3 {
					return
				}
				if c.scheduled != 1 || c.probeSince != 0 {
					t.Fatalf("unexecuted probe changed: depth=%d since=%d", c.scheduled, c.probeSince)
				}
				// A fresh pair must execute the retained probe, rather than wait
				// another 512 rounds for the controller to try drafting again.
				mlx.Scoped(func() {
					for _, req := range requests {
						req.Responses = make(chan CompletionResponse, 32)
						req.Options.NumPredict = 20
						s.admit(req)
					}
				})
				start = len(m.forwards)
				for range 4 {
					mlx.Scoped(func() { s.step() })
				}
				for i := start; i < len(m.forwards); i++ {
					shared = shared || m.batches[i] == 2 && m.forwards[i].n > 1
				}
				if !shared || c.probed {
					t.Fatalf("fresh pair did not consume retained probe: shared=%v pending=%v", shared, c.probed)
				}
			})
		})
	}
}

func TestBatchSchedulerProbeSurvivesCancellation(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		s, m, c := probeTestScheduler(t, ctx, 20)
		req := s.slots[0].request.request
		peer := s.slots[1].request
		before := peer.output.generated
		cancel()
		mlx.Scoped(func() { s.step() })
		if c.scheduled != 1 || !c.probed || c.probeSince != 0 {
			t.Fatalf("cancellation consumed shared probe: depth=%d pending=%v since=%d", c.scheduled, c.probed, c.probeSince)
		}
		if s.slots[0].request != nil || peer.finished || peer.output.generated <= before {
			t.Fatal("cancellation did not release its slot and let the peer progress")
		}
		req.Ctx = context.Background()
		req.Responses = make(chan CompletionResponse, 32)
		mlx.Scoped(func() {
			s.admit(req)
		})
		start := len(m.forwards)
		for range 4 {
			mlx.Scoped(func() { s.step() })
		}
		shared := false
		for i := start; i < len(m.forwards); i++ {
			shared = shared || m.batches[i] == 2 && m.forwards[i].n > 1
		}
		if !shared || c.probed {
			t.Fatalf("pair after cancellation did not consume retained probe: shared=%v pending=%v", shared, c.probed)
		}
	})
}

func TestSpeculativeProbeSurvivesRequestBudget(t *testing.T) {
	for _, remaining := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("remaining%d", remaining), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				r, draft, _ := probeTestRunner(t, sampler.Options{})
				spec := r.spec.open(request{}, nil)
				d := spec.decoder(mlx.FromValues([]int32{1}, 1), 0, nil)
				defer func() {
					if d != nil {
						d.close()
						spec.close()
					}
				}()
				out, err := d.next(remaining + 1)
				if err != nil {
					t.Fatal(err)
				}
				ids := resultIDs(out)
				if !slices.Equal(ids, []int32{2}) || d.inner == nil || spec.limit != 1 || !r.spec.depth.probed {
					t.Fatalf("plain round did not schedule probe: ids=%v inner=%v limit=%d probed=%v", ids, d.inner != nil, spec.limit, r.spec.depth.probed)
				}
				for left := remaining; left > 0; {
					out, err = d.next(left)
					if err != nil {
						t.Fatal(err)
					}
					if len(out) == 0 || len(out) > left {
						t.Fatalf("returned %d tokens with %d remaining", len(out), left)
					}
					ids = append(ids, resultIDs(out)...)
					left -= len(out)
				}
				if want := []int32{2, 3, 1, 2}[:remaining+1]; !slices.Equal(ids, want) {
					t.Fatalf("output=%v, want %v", ids, want)
				}
				if remaining == 3 {
					if spec.stats.drafted != 1 || len(draft.calls) != 1 || r.spec.depth.probed {
						t.Fatalf("available probe not consumed: drafted=%d calls=%d probed=%v", spec.stats.drafted, len(draft.calls), r.spec.depth.probed)
					}
					return
				}
				if c := r.spec.depth; c.scheduled != 1 || !c.probed || c.probeSince != 0 || spec.stats.drafted != 0 {
					t.Fatalf("unexecuted probe consumed: scheduled=%d probed=%v since=%d drafted=%d", c.scheduled, c.probed, c.probeSince, spec.stats.drafted)
				}

				d.close()
				spec.close()
				d = nil
				if !r.cache.caches[0].Restore(nil, 0) {
					t.Fatal("reset target cache for next request")
				}
				spec = r.spec.open(request{}, nil)
				d = spec.decoder(mlx.FromValues([]int32{1}, 1), 0, nil)
				if spec.limit != 1 {
					t.Fatalf("reopened request depth=%d, want pending probe", spec.limit)
				}
				// The fresh drafter first primes, then drains, then can propose.
				for i := range 3 {
					out, err = d.next(8 - i)
					if err != nil {
						t.Fatal(err)
					}
					resultIDs(out)
					if i < 2 && (spec.limit != 1 || len(draft.calls) != 0) {
						t.Fatalf("probe changed during priming/drain: step=%d depth=%d calls=%d", i, spec.limit, len(draft.calls))
					}
				}
				if spec.stats.drafted != 1 || len(draft.calls) != 1 || r.spec.depth.probed {
					t.Fatalf("reopened request did not exercise probe: drafted=%d calls=%d probed=%v", spec.stats.drafted, len(draft.calls), r.spec.depth.probed)
				}
			})
		})
	}
}

func TestSpeculativeProbeDisabledRequest(t *testing.T) {
	for _, opts := range []sampler.Options{{Logprobs: true}, {TopLogprobs: 2}} {
		t.Run(fmt.Sprintf("logprobs%v_top%d", opts.Logprobs, opts.TopLogprobs), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				r, draft, _ := probeTestRunner(t, opts)
				c := r.spec.depth
				if depth := c.next(); depth != 1 {
					t.Fatalf("scheduled depth=%d, want probe", depth)
				}
				spec := r.spec.open(request{SamplerOpts: opts}, nil)
				d := spec.decoder(mlx.FromValues([]int32{1}, 1), 0, nil)
				defer func() {
					d.close()
					spec.close()
				}()
				for left := 3; left > 0; left-- {
					out, err := d.next(left)
					if err != nil || len(resultIDs(out)) != 1 {
						t.Fatalf("disabled decode: err=%v tokens=%d", err, len(out))
					}
				}
				if spec.enabled || spec.limit != 0 || len(draft.calls) != 0 || c.scheduled != 1 || !c.probed || c.probeSince != 0 {
					t.Fatalf("disabled request changed probe: enabled=%v limit=%d calls=%d scheduled=%d probed=%v since=%d", spec.enabled, spec.limit, len(draft.calls), c.scheduled, c.probed, c.probeSince)
				}
			})
		})
	}
}

func TestSpeculativeProbeSurvivesCancellation(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		r, draft, snapshots := probeTestRunner(t, sampler.Options{})
		req := request{Responses: make(chan CompletionResponse, 4), CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 2}}}
		spec := r.spec.open(req, nil)
		d := spec.decoder(mlx.FromValues([]int32{1}, 1), 0, nil)
		defer func() {
			if d != nil {
				d.close()
				spec.close()
			}
		}()
		for _, left := range []int{3, 2} {
			out, err := d.next(left)
			if err != nil || len(resultIDs(out)) != 1 {
				t.Fatalf("prime/drain: %v", err)
			}
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		session := &cacheSession{caches: r.cache.caches}
		err := driveTestDecoder(ctx, r, req, session, d, 0)
		d = nil // driveTestDecoder closes the decoder and speculation session.
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled decode: %v", err)
		}
		if c := r.spec.depth; c.scheduled != 1 || !c.probed || c.probeSince != 0 || len(draft.calls) != 0 {
			t.Fatalf("cancellation consumed probe: scheduled=%d probed=%v since=%d calls=%d", c.scheduled, c.probed, c.probeSince, len(draft.calls))
		}
		for i, snapshot := range snapshots.all {
			if snapshot.closeCount == 0 {
				t.Fatalf("snapshot %d leaked on cancellation", i)
			}
		}
		next := r.spec.open(request{}, nil)
		defer next.close()
		if next.limit != 1 {
			t.Fatalf("request after cancellation starts at depth%d, want pending probe", next.limit)
		}
	})
}
