package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/sample"
)

func testRunnerCheckpoints(dir string) *runnerCheckpoints {
	return &runnerCheckpoints{manifest: &manifest.Manifest{}, dir: dir, ready: make(chan struct{}, 1), stores: make(map[string]*checkpointCache), pending: make(map[*prefixCache]*pendingCheckpoint), idleAge: time.Hour}
}

func TestCheckpointWakeDeadlines(t *testing.T) {
	for _, name := range []string{"coalesce", "retry", "idle", "warm", "writing", "immediate"} {
		t.Run(name, func(t *testing.T) {
			p := testRunnerCheckpoints(t.TempDir())
			c := newPrefixCache(nil)
			due := time.Now().Add(10 * time.Millisecond)
			q := &pendingCheckpoint{cache: c, store: &checkpointCache{}, nodes: []*trieNode{{}}, writeAfter: due}
			p.pending[c] = q
			switch name {
			case "retry":
				q.writeAfter = time.Time{}
				q.retry = due
			case "idle":
				q.nodes = nil
				q.used = due.Add(-p.idleAge)
			case "warm":
				q.nodes = nil
				p.idleAge = -1
			case "writing":
				q.store.job = &checkpointWrite{}
			case "immediate":
				q.writeAfter = time.Time{}
				due = time.Time{}
			}
			timer := p.timer()
			if name == "warm" || name == "writing" {
				if timer != nil {
					t.Fatal("policy or I/O completion should wait without polling")
				}
				return
			}
			select {
			case at := <-timer:
				if at.Before(due) {
					t.Fatal("timer fired before the policy deadline")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("policy deadline did not wake the scheduler")
			}
		})
	}
}

func TestCheckpointCompletionBeforeWait(t *testing.T) {
	d := &checkpointCache{job: &checkpointWrite{}, ready: make(chan struct{}, 1)}
	d.io(func() error { return nil })
	// Let the operation finish before registering interest in its notification.
	// A drain must not require a polling timeout to notice this completion.
	if err := <-d.job.done; err != nil {
		t.Fatal(err)
	}
	select {
	case <-d.ready:
	case <-time.After(5 * time.Second):
		t.Fatal("completion notification was lost before the waiter arrived")
	}
}

func TestCheckpointIdleSchedulerWrites(t *testing.T) {
	worker := mlxtest.Worker(t)
	root := t.TempDir()
	r := &Runner{Model: &cachedRowTestModel{}, parallel: 1, Requests: make(chan queuedRequest), mlxThread: worker, checkpoints: testRunnerCheckpoints(root)}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- r.runBatchRequests(ctx, ctx.Done()) }()
	defer func() {
		cancel()
		<-done
		if err := worker.Do(context.Background(), func() error { r.checkpoints.close(); return nil }); err != nil {
			t.Fatal(err)
		}
	}()
	input := &model.PreparedRequest{Tokens: []int32{1, 2, 3}}
	req := &evaluationRequest{ctx: ctx, done: make(chan error, 1), open: func(slot *batchSlot, peers []*prefixCache) (*evaluationState, error) {
		return r.openEvaluation(slot, input, nil, func(h *mlx.Array) error { mlx.Eval(h); return nil }), nil
	}}
	r.Requests <- req
	if err := <-req.done; err != nil {
		t.Fatal(err)
	}
	// Observe only filesystem publication. There is no next request, shutdown,
	// flush call or access to scheduler-owned state to make the writer progress.
	deadline := time.After(5 * time.Second)
	tick := time.NewTicker(10 * time.Millisecond)
	defer tick.Stop()
	for {
		files, err := filepath.Glob(filepath.Join(root, "*", "*.kv"))
		if err != nil {
			t.Fatal(err)
		}
		if len(files) > 0 {
			return
		}
		select {
		case <-deadline:
			t.Fatal("idle scheduler did not publish the completed request")
		case <-tick.C:
		}
	}
}

func TestCheckpointFlushSkipsCompactedNodes(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	root := t.TempDir()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			session := c.beginScore([]int32{1, 2, 3}, nil)
			checkpointTestFeed(c.caches, session.remaining)
			session.close()
			r := &Runner{checkpoints: testRunnerCheckpoints(root)}
			defer r.checkpoints.close()
			r.checkpoints.completed(session)
			q := r.checkpoints.pending[c]
			q.nodes = append([]*trieNode{{}, {}, {}}, q.nodes...)
			if err := r.flushCheckpoints(ctx); err != nil {
				t.Fatal(err)
			}
			if len(q.store.saved) != 1 || !q.canRelease() {
				t.Fatal("stale prompt nodes stranded the final reply write")
			}
		})
	})
}

func TestCheckpointFlushDuringCompaction(t *testing.T) {
	for _, target := range []string{"prompt", "reply", "published-prompt"} {
		t.Run(target, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			root := t.TempDir()
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					c := newPrefixCache(checkpointTestCaches())
					defer c.close()
					s := c.beginScore([]int32{1, 2, 3, 4, 5, 6, 7}, nil)
					s.schedulePrefillSnapshots([]int{4})
					checkpointTestFeed(c.caches, s.remaining)
					s.close()
					r := &Runner{checkpoints: testRunnerCheckpoints(root)}
					defer r.checkpoints.close()
					r.checkpoints.completed(s)
					<-r.checkpoints.ready
					q := r.checkpoints.pending[c]
					node := c.activePath[1]
					if target == "reply" {
						node = c.activePath[2]
					}
					if err := q.store.start(c, node, nil); err != nil {
						t.Fatal(err)
					}
					q.nodes = nil
					// Consume completed I/O on the MLX thread. Stop either after
					// staging starts or just before accepting the published file.
					for {
						select {
						case <-q.store.ready:
						case <-ctx.Done():
							t.Fatal(ctx.Err())
						}
						if target == "published-prompt" && q.store.job.published {
							break
						}
						if _, err := q.store.step(); err != nil {
							t.Fatal(err)
						}
						if target != "published-prompt" {
							break
						}
					}
					// Eviction merges the prompt and reply in place, without
					// beginning another request or changing the cache epoch.
					c.evictNode(c.activePath[1])
					if err := r.flushCheckpoints(ctx); err != nil {
						t.Fatal(err)
					}
					if !q.canRelease() {
						t.Fatal("compaction stranded the completed reply")
					}
					for _, saved := range q.store.saved {
						if saved.file.Path() == q.saved && len(saved.record.Keys) != 7 {
							t.Fatalf("marked %d-token prompt as a durable copy of the 7-token reply", len(saved.record.Keys))
						}
					}
				})
			})
		})
	}
}

func TestCheckpointSchedulerShutdown(t *testing.T) {
	worker := mlxtest.Worker(t)
	root := t.TempDir()
	r := &Runner{Model: &cachedRowTestModel{}, parallel: 1, Requests: make(chan queuedRequest), mlxThread: worker, checkpoints: testRunnerCheckpoints(root)}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- r.runBatchRequests(ctx, ctx.Done()) }()
	input := &model.PreparedRequest{Tokens: []int32{1, 2, 3, 4, 5}}
	req := &evaluationRequest{ctx: ctx, done: make(chan error, 1), open: func(slot *batchSlot, peers []*prefixCache) (*evaluationState, error) {
		return r.openEvaluation(slot, input, nil, func(h *mlx.Array) error { mlx.Eval(h); return nil }), nil
	}}
	select {
	case r.Requests <- req:
	case <-time.After(10 * time.Second):
		t.Fatal("scheduler did not accept input")
	}
	select {
	case err := <-req.done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("evaluation did not finish")
	}
	// No explicit flush: normal cancellation must finish the queued save
	// before releasing the completed slot's cache.
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("scheduler shutdown: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("scheduler did not shut down")
	}
	if err := worker.Do(context.Background(), func() error {
		r.checkpoints.close()
		r.checkpoints = testRunnerCheckpoints(root)
		defer r.checkpoints.close()
		c := newPrefixCache(append(r.Model.NewCaches(), cache.NewHiddenCache()))
		defer c.close()
		d := r.checkpoints.open(c)
		if d == nil || len(d.saved) != 1 {
			return fmt.Errorf("shutdown discarded the completed checkpoint")
		}
		restore := d.restore(c, input.Tokens, nil, len(input.Tokens), 0)
		if restore == nil {
			return fmt.Errorf("shutdown checkpoint cannot be restored")
		}
		defer restore.close(false)
		for {
			finished, err := restore.step(context.Background())
			if err != nil {
				return err
			}
			if finished {
				break
			}
		}
		if err := restore.commit(); err != nil {
			return err
		}
		if c.minCacheOffset() != len(input.Tokens) {
			return fmt.Errorf("shutdown saved incomplete state")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestCheckpointGenerationSchedulerRestart(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	root := t.TempDir()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			var baseline string
			for round := range 2 {
				tok := newTestTokenizer(t, []int32{7})
				m := &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}
				r := &Runner{Model: m, Tokenizer: tok, Sampler: sample.New(4096), parallel: 2, checkpoints: testRunnerCheckpoints(root)}
				s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
				req := request{Ctx: context.Background(), Tokens: slices.Repeat([]int32{2}, 12), Responses: make(chan CompletionResponse, 32), CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 3}}, SamplerOpts: sample.Options{Temperature: 0}}
				mlx.Scoped(func() { s.admit(req) })
				if round == 1 && s.slots[0].request.restore == nil {
					t.Fatal("restarted generation did not schedule a disk restore")
				}
				for i := 0; i < 100 && s.slots[0].busy(); i++ {
					mlx.Scoped(func() { s.step() })
				}
				if s.slots[0].busy() {
					t.Fatal("generation stalled")
				}
				var text string
				var final CompletionResponse
				for response := range req.Responses {
					if response.Error != nil {
						t.Fatal(response.Error)
					}
					text += response.Content
					if response.Done {
						final = response
					}
				}
				if !final.Done || final.EvalCount != 3 || final.PromptEvalCachedCount == nil {
					t.Fatalf("incomplete generation: %+v", final)
				}
				if round == 0 {
					baseline = text
				} else if text != baseline || *final.PromptEvalCachedCount != 8 {
					t.Fatalf("restart changed output or reuse: %q vs %q, cached=%d", text, baseline, *final.PromptEvalCachedCount)
				}
				if err := r.flushCheckpoints(ctx); err != nil {
					t.Fatal(err)
				}
				files := 0
				for _, d := range r.checkpoints.stores {
					files += len(d.saved)
				}
				if files != 2 {
					t.Fatalf("repeat rewrote durable prompt/reply state: %d files", files)
				}
				if round == 1 {
					// Interrupt after one layer is restored, before a forward pass.
					// The surviving checkpoint must still work for another request.
					s.slots[0].cache.close()
					requestCtx, cancel := context.WithCancel(context.Background())
					canceled := req
					canceled.Ctx = requestCtx
					canceled.Responses = make(chan CompletionResponse, 32)
					mlx.Scoped(func() { s.admit(canceled); s.step() })
					if s.slots[0].request.restore == nil || s.slots[0].cache.minCacheOffset() == 0 {
						t.Fatal("cancellation fixture did not reach a partial restore")
					}
					cancel()
					mlx.Scoped(func() { s.step() })
					for response := range canceled.Responses {
						t.Fatalf("canceled restore emitted a response: %+v", response)
					}
					if s.slots[0].cache.minCacheOffset() != 0 {
						t.Fatal("canceled request retained partial state")
					}
					retry := req
					retry.Responses = make(chan CompletionResponse, 32)
					mlx.Scoped(func() { s.admit(retry) })
					if s.slots[0].request.restore == nil {
						t.Fatal("cancellation discarded a valid checkpoint")
					}
					for i := 0; i < 100 && s.slots[0].busy(); i++ {
						mlx.Scoped(func() { s.step() })
					}
					if s.slots[0].busy() {
						t.Fatal("retry stalled after cancellation")
					}
					var text string
					for response := range retry.Responses {
						if response.Error != nil {
							t.Fatal(response.Error)
						}
						text += response.Content
					}
					if text != baseline {
						t.Fatalf("retry after cancellation: %q vs %q", text, baseline)
					}
					if err := r.flushCheckpoints(ctx); err != nil {
						t.Fatal(err)
					}
				}
				r.checkpoints.idleAge = 0
				mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
				if s.slots[0].cache.minCacheOffset() != 0 || s.slots[0].cache.root != nil {
					t.Fatal("persisted idle cache was not released")
				}
				r.checkpoints.close()
				s.slots[0].cache.close()
				r.Sampler.Free()
			}
		})
	})
}

func TestCheckpointReadoutSchedulerRestart(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	root := t.TempDir()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			var baseline []float32
			for round := range 2 {
				m := &cachedRowTestModel{}
				r := &Runner{Model: m, parallel: 2, checkpoints: testRunnerCheckpoints(root)}
				s := &batchScheduler{r: r, slots: make([]*batchSlot, 2)}
				input := &model.PreparedRequest{Tokens: []int32{1, 2, 3, 4, 5}}
				var state *evaluationState
				var got []float32
				req := &evaluationRequest{ctx: context.Background(), done: make(chan error, 1), open: func(slot *batchSlot, peers []*prefixCache) (*evaluationState, error) {
					state = r.openEvaluation(slot, input, nil, func(h *mlx.Array) error { got = h.Floats(); return nil })
					return state, nil
				}}
				mlx.Scoped(func() { s.admit(req) })
				if round == 1 && state.restore == nil {
					t.Fatal("restarted readout did not schedule a disk restore")
				}
				for i := 0; i < 100 && s.slots[0].busy(); i++ {
					mlx.Scoped(func() { s.step() })
				}
				if s.slots[0].busy() {
					t.Fatal("readout stalled")
				}
				if err := <-req.done; err != nil {
					t.Fatal(err)
				}
				if round == 0 {
					baseline = got
				} else if !slices.Equal(got, baseline) || state.cached != len(input.Tokens) || len(m.shapes) != 0 {
					t.Fatalf("restart changed readout or recomputed it: %v vs %v, cached=%d, shapes=%v", got, baseline, state.cached, m.shapes)
				}
				if err := r.flushCheckpoints(ctx); err != nil {
					t.Fatal(err)
				}
				r.checkpoints.close()
				s.slots[0].scoreCache.close()
			}
		})
	})
}

func TestCheckpointFlushCancellation(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			s := c.beginScore([]int32{1, 2, 3}, nil)
			checkpointTestFeed(c.caches, s.remaining)
			s.close()
			r := &Runner{checkpoints: testRunnerCheckpoints(root)}
			defer r.checkpoints.close()
			r.checkpoints.completed(s)
			canceled, stop := context.WithCancel(ctx)
			stop()
			if err := r.flushCheckpoints(canceled); !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled flush returned %v", err)
			}
			d := r.checkpoints.open(c)
			guard, err := d.store.lock(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer guard.Close()
			mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
			blocked, stop := context.WithTimeout(ctx, 50*time.Millisecond)
			defer stop()
			if err := r.flushCheckpoints(blocked); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("flush waiting for a peer's allocation lock returned %v", err)
			}
			if d.job == nil || c.minCacheOffset() != 3 {
				t.Fatal("flush timeout lost the pending write or live state")
			}
			guard.Close()
			if err := r.flushCheckpoints(ctx); err != nil {
				t.Fatal(err)
			}
			if len(d.saved) != 1 || c.minCacheOffset() != 3 {
				t.Fatalf("flush lost state: saved=%d, live=%d", len(d.saved), c.minCacheOffset())
			}
		})
	})
}

func TestCheckpointFlushDiskLimit(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			s := c.beginScore([]int32{1, 2, 3}, nil)
			checkpointTestFeed(c.caches, s.remaining)
			s.close()
			r := &Runner{checkpoints: testRunnerCheckpoints(root)}
			defer r.checkpoints.close()
			r.checkpoints.completed(s)
			d := r.checkpoints.open(c)
			d.limit, d.store.limit = 1, 1
			if err := r.flushCheckpoints(ctx); err == nil {
				t.Fatal("flush reported success without room for a checkpoint")
			}
			if len(d.saved) != 0 || d.job != nil || c.minCacheOffset() != 3 {
				t.Fatalf("failed flush changed live cache or left a writer: saved=%d, live=%d, writer=%v", len(d.saved), c.minCacheOffset(), d.job != nil)
			}
			if q := r.checkpoints.pending[c]; q == nil || !q.retry.After(time.Now()) {
				t.Fatal("failed write did not schedule a future retry")
			}
			timer := r.checkpoints.timer()
			if timer == nil {
				t.Fatal("failed write has no deadline to wake the idle scheduler")
			}
			select {
			case <-timer:
				t.Fatal("failed write retried immediately instead of honoring backoff")
			default:
			}
		})
	})
}

func TestCheckpointIdleAfterDiskEviction(t *testing.T) {
	for _, removal := range []string{"none", "budget", "peer"} {
		t.Run(removal, func(t *testing.T) {
			root := t.TempDir()
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					c := newPrefixCache(checkpointTestCaches())
					defer c.close()
					s := c.beginScore([]int32{1, 2, 3}, nil)
					checkpointTestFeed(c.caches, s.remaining)
					s.close()
					r := &Runner{checkpoints: testRunnerCheckpoints(root)}
					defer r.checkpoints.close()
					r.checkpoints.completed(s)
					if err := r.flushCheckpoints(ctx); err != nil {
						t.Fatal(err)
					}
					d := r.checkpoints.open(c)
					if len(d.saved) != 1 {
						t.Fatalf("saved %d checkpoints, want 1", len(d.saved))
					}
					switch removal {
					case "budget":
						d.limit = 0
						if !d.trim(0) || len(d.saved) != 0 {
							t.Fatal("disk eviction did not remove the checkpoint")
						}
					case "peer":
						if err := os.Remove(d.saved[0].file.Path()); err != nil {
							t.Fatal(err)
						}
					}
					r.checkpoints.idleAge = 0
					mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
					want := 3
					if removal == "none" {
						want = 0
					}
					if got := c.minCacheOffset(); got != want {
						t.Fatalf("idle cache contains %d tokens, want %d after %s removal", got, want, removal)
					}
				})
			})
		})
	}
}

func TestCheckpointPressureReclaimsSavedIdle(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			r := &Runner{checkpoints: testRunnerCheckpoints(root)}
			defer r.checkpoints.close()
			var caches []*prefixCache
			for i := range 3 {
				c := newPrefixCache(checkpointTestCaches())
				defer c.close()
				caches = append(caches, c)
				s := c.beginScore([]int32{int32(i + 1), 20, 30}, nil)
				checkpointTestFeed(c.caches, s.remaining)
				s.close()
				r.checkpoints.completed(s)
				if i < 2 {
					if err := r.flushCheckpoints(ctx); err != nil {
						t.Fatal(err)
					}
				}
			}
			// The first row is active, the second idle and saved, and the last
			// idle but not saved. Reclamation must select only the second row.
			if got := r.checkpoints.reclaimSaved(caches, caches[1:], nil, 1); got <= 0 {
				t.Fatalf("reclaiming saved idle state released %d bytes", got)
			}
			for i, want := range []int{3, 0, 3} {
				if got := caches[i].minCacheOffset(); got != want {
					t.Fatalf("cache %d contains %d tokens, want %d", i, got, want)
				}
			}
			d := r.checkpoints.open(caches[1])
			restore := d.restore(caches[1], []int32{2, 20, 30}, nil, 3, 0)
			if restore == nil {
				t.Fatal("pressure reclamation lost the durable checkpoint")
			}
			for {
				done, err := restore.step(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if done {
					break
				}
			}
			if err := restore.commit(); err != nil {
				t.Fatal(err)
			}
			if got := caches[1].caches[0].State()[0].AsType(mlx.DTypeInt32).Ints(); !slices.Equal(got, []int32{2, 20, 30}) {
				t.Fatalf("restored %v after pressure reclamation", got)
			}
		})
	})
}

func TestCheckpointRetentionPolicyReclaimsSavedIdle(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("adaptive retention is enabled on Metal")
		}
		r := &Runner{parallel: 2, checkpoints: testRunnerCheckpoints(root)}
		defer r.checkpoints.close()
		scheduler := &batchScheduler{r: r, snapshotWorkspace: math.MaxInt64 / 2}
		defer func() {
			if scheduler.memory != nil {
				scheduler.memory.Close()
			}
			if scheduler.snapshotTimer != nil {
				scheduler.snapshotTimer.Stop()
			}
		}()
		for i := range 2 {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			session := c.beginScore([]int32{int32(i + 1), 20, 30}, nil)
			checkpointTestFeed(c.caches, session.remaining)
			session.close()
			r.checkpoints.completed(session)
			scheduler.slots = append(scheduler.slots, &batchSlot{cache: c})
		}
		if err := r.flushCheckpoints(ctx); err != nil {
			t.Fatal(err)
		}
		active, idle := scheduler.slots[0], scheduler.slots[1]
		active.request = &batchRequest{}
		saved := r.checkpoints.pending[idle.cache].saved
		// Force a shortage through the workspace reserve, without allocating
		// memory or depending on host pressure. Only the idle row may be freed.
		scheduler.trimSnapshots(nil, -1)
		if active.cache.minCacheOffset() != 3 || idle.cache.minCacheOffset() != 0 {
			t.Fatal("retention policy did not preserve active state and reclaim idle state")
		}
		if _, pending := r.checkpoints.pending[idle.cache]; pending {
			t.Fatal("retention discarded history without releasing its saved checkpoint owner")
		}
		if _, err := os.Stat(saved); err != nil {
			t.Fatalf("retention removed the durable checkpoint: %v", err)
		}
	})
}

func TestCheckpointPressureDrainsPending(t *testing.T) {
	for _, disk := range []string{"available", "locked", "full"} {
		t.Run(disk, func(t *testing.T) {
			root := t.TempDir()
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					c := newPrefixCache(checkpointTestCaches())
					defer c.close()
					s := c.beginScore([]int32{1, 2, 3}, nil)
					checkpointTestFeed(c.caches, s.remaining)
					s.close()
					r := &Runner{checkpoints: testRunnerCheckpoints(root)}
					defer r.checkpoints.close()
					r.checkpoints.completed(s)
					d := r.checkpoints.open(c)
					r.checkpoints.pending[c].writeAfter = time.Now().Add(time.Hour)
					if disk == "locked" {
						guard, err := d.store.lock(ctx)
						if err != nil {
							t.Fatal(err)
						}
						defer guard.Close()
					} else if disk == "full" {
						d.limit, d.store.limit = 1, 1
					}
					drainCtx := ctx
					if disk == "locked" {
						var stop context.CancelFunc
						drainCtx, stop = context.WithTimeout(ctx, 100*time.Millisecond)
						defer stop()
					}
					ready := r.checkpoints.finishPending(drainCtx, []*prefixCache{c}, []*prefixCache{c}, nil)
					released := r.checkpoints.reclaimSaved([]*prefixCache{c}, []*prefixCache{c}, nil, 1)
					if ready != (released > 0) {
						t.Fatal("pending completion did not identify reclaimable state")
					}
					if disk == "available" {
						if released <= 0 || c.minCacheOffset() != 0 || len(d.saved) != 1 {
							t.Fatalf("drain lost state: released=%d live=%d saved=%d", released, c.minCacheOffset(), len(d.saved))
						}
					} else if released != 0 || c.minCacheOffset() != 3 || len(d.saved) != 0 {
						t.Fatalf("failed drain changed state: released=%d live=%d saved=%d", released, c.minCacheOffset(), len(d.saved))
					}
				})
			})
		})
	}
}

func TestCheckpointReclamationWaitsForReply(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			tokens := []int32{1, 2, 3, 4, 5, 6, 7}
			s := c.beginScore(tokens, nil)
			s.schedulePrefillSnapshots([]int{4})
			checkpointTestFeed(c.caches, s.remaining)
			s.close()
			r := &Runner{checkpoints: testRunnerCheckpoints(root)}
			defer r.checkpoints.close()
			r.checkpoints.completed(s)
			d := r.checkpoints.open(c)
			for len(d.saved) == 0 {
				mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
				if len(d.saved) > 0 {
					break
				}
				select {
				case <-r.checkpoints.wake():
				case <-r.checkpoints.timer():
				case <-ctx.Done():
					t.Fatal("prompt checkpoint did not finish")
				}
			}
			if len(d.saved) != 1 || len(d.saved[0].record.Keys) != 4 {
				t.Fatal("fixture did not stop after saving only the prompt")
			}
			if got := r.checkpoints.reclaimSaved([]*prefixCache{c}, []*prefixCache{c}, nil, 1); got != 0 || c.minCacheOffset() != len(tokens) {
				t.Fatalf("prompt checkpoint allowed reclaiming an unsaved reply: released=%d, live=%d", got, c.minCacheOffset())
			}

			if err := r.flushCheckpoints(ctx); err != nil {
				t.Fatal(err)
			}
			if got := r.checkpoints.reclaimSaved([]*prefixCache{c}, []*prefixCache{c}, nil, 1); got <= 0 || c.minCacheOffset() != 0 {
				t.Fatalf("complete checkpoint was not reclaimed: released=%d, live=%d", got, c.minCacheOffset())
			}
			restore := d.restore(c, tokens, nil, len(tokens), 0)
			if restore == nil {
				t.Fatal("reclamation lost the completed reply")
			}
			for {
				done, err := restore.step(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if done {
					break
				}
			}
			if err := restore.commit(); err != nil {
				t.Fatal(err)
			}
			if got := c.caches[0].State()[0].AsType(mlx.DTypeInt32).Ints(); !slices.Equal(got, tokens) {
				t.Fatalf("restored %v, want prompt and reply %v", got, tokens)
			}
		})
	})
}

func TestCheckpointCoalescesCompletedRequests(t *testing.T) {
	root := t.TempDir()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			r := &Runner{checkpoints: testRunnerCheckpoints(root)}
			defer r.checkpoints.close()
			for _, token := range []int32{1, 2, 3} {
				s := c.beginScore([]int32{token, 20, 30}, nil)
				checkpointTestFeed(c.caches, s.remaining)
				s.close()
				r.checkpoints.completed(s)
				q := r.checkpoints.pending[c]
				if q.writeAfter.Sub(q.used) != checkpointWriteDelay {
					t.Fatal("completion did not schedule a deferred write")
				}
				// Keep this assertion independent of how slowly the test host runs.
				q.writeAfter = time.Now().Add(time.Hour)
				mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
				if q.store.job != nil || len(q.store.saved) != 0 {
					t.Fatal("request burst started an obsolete write")
				}
			}
			if err := r.flushCheckpoints(ctx); err != nil {
				t.Fatal(err)
			}
			d := r.checkpoints.open(c)
			if len(d.saved) != 1 || !slices.Equal(d.saved[0].record.Keys, c.key([]uint32{3, 20, 30})) {
				t.Fatal("unload did not bypass coalescing and save the latest request")
			}
			repeat := c.beginScore([]int32{3, 20, 30}, nil)
			repeat.close()
			r.checkpoints.completed(repeat)
			r.checkpoints.pending[c].writeAfter = time.Now().Add(time.Hour)
			mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
			if got := r.checkpoints.reclaimSaved([]*prefixCache{c}, []*prefixCache{c}, nil, 1); got <= 0 {
				t.Fatal("coalescing delayed reclamation of an already durable checkpoint")
			}
		})
	})
}

func maintainTestCheckpoints(p *runnerCheckpoints) {
	var slots []*prefixCache
	for c := range p.pending {
		slots = append(slots, c)
	}
	p.release(slots, p.maintain(slots))
}
