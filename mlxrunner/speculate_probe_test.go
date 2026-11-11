package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"slices"
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
	c := r.spec.depth
	c.drafterLimit = 1
	c.cost.observe(0, time.Millisecond)
	c.cost.observe(1, 10*time.Millisecond)
	for range 10 {
		c.acc.observe(1, 0)
	}
	c.probeInterval = depthProbeIntervalMax
	c.probeSince = depthProbeIntervalMax - 1
	return r, draft, snapshots
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
