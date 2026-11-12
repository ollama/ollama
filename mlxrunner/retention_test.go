package mlxrunner

import (
	"context"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/sample"
)

func TestAdmissionReclaimsIdleHistory(t *testing.T) {
	ctx := t.Context()
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("adaptive host-memory retention is enabled on Metal")
		}
		tok := newTestTokenizer(t, []int32{7})
		r := &Runner{Model: &schedulingModel{fakeMTPModel: fakeMTPModel{tok: tok}}, Tokenizer: tok, Sampler: sample.New(4096)}
		defer r.Sampler.Free()
		s := &batchScheduler{r: r, slots: make([]*batchSlot, 1)}
		defer func() {
			if slot := s.slots[0]; slot != nil {
				if slot.request != nil {
					s.finish(slot, context.Canceled)
				}
				slot.cache.close()
			}
		}()
		for round := range 2 {
			if round == 1 {
				if s.slots[0].cache.minCacheOffset() < 3 {
					t.Fatal("first request did not retain its prefix")
				}
				// Simulate a workspace requirement exceeding host capacity without
				// allocating it or depending on other processes to create pressure.
				s.snapshotWorkspace = math.MaxInt64 / 2
			}
			req := request{
				Ctx: ctx, Tokens: []int32{1, 1, 1}, Responses: make(chan CompletionResponse, 8),
				CompletionRequest: CompletionRequest{Options: api.Options{NumPredict: 2}},
			}
			mlx.Scoped(func() { s.admit(req) })
			if s.snapshotCheck.IsZero() || !s.observedMemory {
				t.Fatal("admission did not initialize memory observation")
			}
			if offset := s.slots[0].cache.minCacheOffset(); offset != 0 {
				t.Fatalf("round %d: admission retained %d tokens, want a cold start", round, offset)
			}
			for i := 0; i < 20 && s.slots[0].request != nil; i++ {
				mlx.Scoped(func() { s.step() })
			}
			if s.slots[0].request != nil {
				t.Fatalf("round %d: request did not finish", round)
			}
			text, final := collectResponses(req.Responses)
			if want := strings.Repeat(tok.Decode([]int32{1}), 2); text != want || !final.Done || final.EvalCount != 2 {
				t.Fatalf("round %d: output %q (%+v), want %q", round, text, final, want)
			}
			if final.PromptEvalCachedCount == nil || *final.PromptEvalCachedCount != 0 || final.PromptEvalCount != 3 {
				t.Fatalf("round %d: expected recomputation of all three tokens, got %+v", round, final)
			}
		}
	})
}

func TestPressureEvictionPreservesSharedPrefix(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		pc := env.pc
		cold := []int32{1, 2, 3, 4, 5, 11, 12, 13}
		hot := []int32{1, 2, 3, 4, 5, 21, 22, 23}
		simulateRequest(t, pc, cold, []int32{14, 15}, 5)
		simulateRequest(t, pc, hot, []int32{24, 25}, 5)
		frontier := pc.activePath[len(pc.activePath)-1]
		before := frontier.endOffset
		for evictColdHistory([]*prefixCache{pc}, nil) {
			checkTrieInvariants(t, pc.root)
		}
		if pc.activePath[len(pc.activePath)-1] != frontier || frontier.endOffset != before {
			t.Fatal("pressure changed the busy conversation's frontier")
		}
		// The cold branch is gone, but the hot path still has every slice.
		_, matched := findBestMatch(pc.root, pc.key(effectiveKeyTokens(cold, nil)))
		if matched != 5-pc.draftLookahead {
			t.Fatalf("cold branch retained %d keys; want shared prefix %d", matched, 5-pc.draftLookahead)
		}
		extended := append(slices.Clone(hot), 24, 25, 26)
		session := pc.begin(extended, nil)
		if reused := len(extended) - len(session.remaining); reused != before {
			t.Fatalf("hot path reused %d tokens; want %d", reused, before)
		}
		env.assertAllTokens(t, "preserved history", extended[:before])
	})
}

func TestPressureEvictionCanReleaseIdleFrontier(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		pc := env.pc
		prompt := []int32{1, 2, 3, 4, 5, 6, 7, 8}
		simulateRequest(t, pc, prompt, []int32{9, 10}, 5)
		if evictColdHistory([]*prefixCache{pc}, nil) {
			t.Fatal("a busy frontier must not be evicted")
		}
		if !evictColdHistory([]*prefixCache{pc}, []*prefixCache{pc}) {
			t.Fatal("an idle frontier should yield to pressure")
		}
		frontier := pc.activePath[len(pc.activePath)-1]
		if frontier.endOffset != 5-pc.draftLookahead {
			t.Fatalf("remaining prefix = %d, want %d", frontier.endOffset, 5-pc.draftLookahead)
		}
		// Discarding the suffix must not invalidate the retained shared prefix.
		session := pc.begin(prompt, nil)
		if len(prompt)-len(session.remaining) != frontier.endOffset {
			t.Fatal("remaining prefix is no longer reusable")
		}
		env.assertAllTokens(t, "retained prefix", prompt[:frontier.endOffset])
		for evictColdHistory([]*prefixCache{pc}, []*prefixCache{pc}) {
		}
		if pc.minCacheOffset() != 0 || pc.pagedOutBytes != 0 || len(pc.root.children) != 0 {
			t.Fatal("empty idle history retained live state")
		}
		result := simulateRequest(t, pc, prompt, []int32{9, 10}, 5)
		if len(result.remaining) != len(prompt) {
			t.Fatal("discarded history unexpectedly reused")
		}
	})
}

func TestPressureEvictionOrdersColdLeavesAcrossSlots(t *testing.T) {
	old, recent := newTransformerEnv(), newTransformerEnv()
	defer old.pc.close()
	defer recent.pc.close()
	for _, env := range []*testEnv{old, recent} {
		simulateRequest(t, env.pc, []int32{1, 2, 3}, []int32{4})
		simulateRequest(t, env.pc, []int32{8, 9, 10}, []int32{11})
	}
	old.pc.root.children[0].lastUsed = time.Unix(1, 0)
	recent.pc.root.children[0].lastUsed = time.Unix(2, 0)
	if !evictColdHistory([]*prefixCache{recent.pc, old.pc}, nil) {
		t.Fatal("no cold history evicted")
	}
	if len(old.pc.root.children) != 1 || len(recent.pc.root.children) != 2 {
		t.Fatal("eviction did not select the oldest leaf across slots")
	}
}

func TestRetainedHistoryPressureReleasesLazyBacking(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.GPUIsAvailable() {
			t.Skip("GPU allocator counters required")
		}
		mlx.DefaultStream().Synchronize()
		baseline := mlx.ActiveMemory()
		kv := cache.NewKVCache()
		pc := newPrefixCache([]cache.Cache{kv})
		defer pc.close()
		pc.ensureRoot()
		mlx.Scoped(func() {
			b := &batch.Batch{InputIDs: mlx.Zeros(mlx.DTypeInt32, 1, 64), SeqQueryLens: []int32{64}}
			kv.Update(b, mlx.Zeros(mlx.DTypeFloat32, 1, 4, 64, 8), mlx.Zeros(mlx.DTypeFloat32, 1, 4, 64, 8))
			mlx.Eval(kv.State()...)
		})
		n := pc.root.appendChild(make([]trieKey, 64), 64)
		pc.activePath = append(pc.activePath, n)
		n.setSnapshots([]cache.Snapshot{kv.Snapshot(0)}, &pc.pagedOutBytes)
		pending := cache.PendingCopyBytes(n.snapshots[0])
		if pc.pagedOutBytes != 0 || pending == 0 {
			t.Fatal("test requires a lazy snapshot with future copy cost")
		}
		live := mlx.ActiveMemory()
		trimRetainedHistory([]*prefixCache{pc}, []*prefixCache{pc}, int64(live+pending))
		if len(pc.root.children) != 1 {
			t.Fatal("history evicted despite sufficient headroom")
		}
		trimRetainedHistory([]*prefixCache{pc}, []*prefixCache{pc}, int64(baseline))
		mlx.DefaultStream().Synchronize()
		if len(pc.root.children) != 0 || kv.Offset() != 0 {
			t.Fatal("idle history did not yield under pressure")
		}
		if retained := mlx.ActiveMemory() - baseline; retained > 0 {
			t.Fatalf("cleared history still pins %d native bytes", retained)
		}
	})
}

func TestRetentionProtectsOnlyExecutingCache(t *testing.T) {
	generation, scoring := newTransformerEnv(), newTransformerEnv()
	defer generation.pc.close()
	defer scoring.pc.close()
	slot := &batchSlot{cache: generation.pc, scoreCache: scoring.pc}
	slot.request = &batchRequest{}
	if slot.activeCache() != generation.pc {
		t.Fatal("generation must protect its own cache")
	}
	slot.request.finished = true
	if slot.activeCache() != nil {
		t.Fatal("completed generation must not pin retained history")
	}
	slot.request = nil
	slot.evaluation = &activeEvaluation{}
	if slot.activeCache() != scoring.pc {
		t.Fatal("evaluation must protect its own cache")
	}
	slot.evaluation = nil
	slot.speech = &speechRequest{}
	s := &batchScheduler{slots: []*batchSlot{slot}}
	if !s.hasPrefixCache() {
		t.Fatal("speech hid the runner's existing conversation history")
	}
	if slot.activeCache() != nil {
		t.Fatal("speech must not pin unrelated prefix caches")
	}
}

func TestCompletedSlowReaderSurvivesHistoryEviction(t *testing.T) {
	env := newTransformerEnv()
	defer env.pc.close()
	simulateRequest(t, env.pc, []int32{1, 2, 3}, []int32{4})
	responses := make(chan CompletionResponse, 1)
	responses <- CompletionResponse{Content: "first"}
	slot := &batchSlot{cache: env.pc, request: &batchRequest{
		request:   request{Ctx: context.Background(), Responses: responses},
		finished:  true,
		responses: []CompletionResponse{{Content: "last", Done: true}},
	}}
	scheduler := &batchScheduler{r: &Runner{}, slots: []*batchSlot{slot}}
	if scheduler.step() {
		t.Fatal("full reader buffer unexpectedly advanced")
	}
	if slot.activeCache() != nil || !evictColdHistory([]*prefixCache{env.pc}, []*prefixCache{env.pc}) {
		t.Fatal("completed slow reader pinned its history")
	}
	if first := <-responses; first.Content != "first" {
		t.Fatal("queued response changed")
	}
	if !scheduler.step() || slot.request != nil {
		t.Fatal("response did not finish after eviction")
	}
	if final := <-responses; final.Content != "last" || !final.Done {
		t.Fatalf("final response changed: %+v", final)
	}
	if _, open := <-responses; open {
		t.Fatal("completion channel remains open")
	}
}

func feedNativeRetention(c *prefixCache, tokens []int32) {
	mlx.Scoped(func() {
		values := make([]float32, 2*len(tokens)*8)
		for h := range 2 {
			for i, token := range tokens {
				for d := range 8 {
					values[(h*len(tokens)+i)*8+d] = float32(token)
				}
			}
		}
		b := &batch.Batch{InputIDs: mlx.FromValues(tokens, 1, len(tokens)), SeqQueryLens: []int32{int32(len(tokens))}}
		k := mlx.FromValues(values, 1, 2, len(tokens), 8)
		v := mlx.FromValues(values, 1, 2, len(tokens), 8)
		for _, layer := range c.caches {
			layer.(cache.Attention).Update(b, k, v)
			mlx.Eval(layer.State()...)
		}
	})
	mlx.DefaultStream().Synchronize()
}

func nativeRetentionHistory() *prefixCache {
	c := newPrefixCache([]cache.Cache{cache.NewKVCache(), cache.NewRotatingKVCache(4)})
	c.ensureRoot()
	for start := 0; start < 12; start += 4 {
		tokens := []int32{int32(start + 1), int32(start + 2), int32(start + 3), int32(start + 4)}
		feedNativeRetention(c, tokens)
		n := c.advancePath(c.activePath[len(c.activePath)-1], c.key(effectiveKeyTokens(tokens, nil)), start+4)
		n.user = start < 8
		n.lastUsed = time.Unix(int64(start+1), 0)
		mlx.Scoped(func() { c.pageOut(n) })
	}
	return c
}

func TestPressureCompactsOldCheckpointsWithoutLosingLatestPrefix(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.GPUIsAvailable() {
			t.Skip("GPU allocator counters required")
		}
		c := nativeRetentionHistory()
		defer c.close()
		frontier := c.activePath[len(c.activePath)-1]
		old := c.activePath[1]
		saved := old.snapshotBytes()
		if saved == 0 {
			t.Fatal("fixture needs an independently owned old sliding state")
		}
		var pending int64
		walkNodes(c.root, func(n *trieNode) bool {
			for _, snap := range n.snapshots {
				pending += int64(cache.PendingCopyBytes(snap))
			}
			return true
		})
		before, retained := mlx.ActiveMemory(), c.pagedOutBytes
		t.Logf("before active=%d retained=%d pending=%d old=%d", before, retained, pending, saved)
		mlx.ResetPeakMemory()
		trimRetainedHistory([]*prefixCache{c}, nil, int64(before)+pending-saved)
		if got := mlx.PeakMemory(); got > before {
			t.Fatalf("pressure compaction allocated: peak %d > %d", got, before)
		}
		t.Logf("after active=%d retained=%d path=%d", mlx.ActiveMemory(), c.pagedOutBytes, len(c.activePath))
		if freed := int64(before - mlx.ActiveMemory()); freed != saved {
			t.Fatalf("native bytes released=%d, want old checkpoint %d", freed, saved)
		}
		if c.pagedOutBytes != retained-saved || len(c.activePath) != 3 || c.activePath[2] != frontier {
			t.Fatal("pressure compaction lost the latest frontier or miscounted saved state")
		}
		input := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		if session := c.beginScore(input, nil); len(session.remaining) != 0 {
			t.Fatalf("latest prefix reprocessed %d tokens", len(session.remaining))
		}
		got := c.caches[0].State()[0].Floats()
		for i, token := range input {
			if got[i*8] != float32(token) {
				t.Fatalf("retained token %d=%v, want %d", i, got[i*8], token)
			}
		}
	})
}

func TestPressureNeverMergesFrontierPastPendingCapture(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		input := []int32{1, 2, 3, 4, 21, 22, 23, 24}
		session := c.beginScore(input, nil)
		if got := c.minCacheOffset(); got != 4 {
			t.Fatalf("fixture restored %d tokens, want 4", got)
		}
		frontier := c.activePath[len(c.activePath)-1]
		frontier.lastUsed = time.Time{} // Eligible first if the frontier guard is removed.
		session.schedulePrefillSnapshots([]int{6, 8})
		if !evictColdHistory([]*prefixCache{c}, nil) {
			t.Fatal("old inactive history was not eligible")
		}
		if frontier.endOffset != 4 || c.activePath[len(c.activePath)-1] != frontier {
			t.Fatal("pressure moved the active frontier beyond its pending captures")
		}
		feedNativeRetention(c, input[4:])
		session.attachPrefillSnapshots()
		session.close()
		for _, end := range []int{6, 8} {
			_, matched := findBestMatch(c.root, c.key(effectiveKeyTokens(input[:end], nil)))
			if matched != end {
				t.Fatalf("capture at %d has only %d matching tokens", end, matched)
			}
		}
		if got := c.beginScore(input, nil); len(got.remaining) != 0 {
			t.Fatal("newly captured branch cannot be fully restored")
		}
	})
}

func TestCompressionObservationTracksChanges(t *testing.T) {
	s := &batchScheduler{}
	for i, tc := range []struct{ current, want int64 }{{10, 0}, {20, 10}, {5, 0}, {12, 7}, {12, 0}} {
		if got := s.observeCompression(tc.current); got != tc.want {
			t.Fatalf("observation %d: new compressed bytes = %d, want %d", i, got, tc.want)
		}
	}
}

func TestCompressionReclaimsOwnedHistory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		frontier := c.activePath[len(c.activePath)-1]
		old := c.activePath[1].snapshotBytes()
		if got := reclaimColdHistory([]*prefixCache{c}, 0); got != 0 || len(c.activePath) != 4 {
			t.Fatal("unchanged compressor triggered reclamation")
		}
		if got := reclaimColdHistory([]*prefixCache{c}, old); got < old {
			t.Fatalf("owned bytes released = %d, want at least %d", got, old)
		}
		if c.activePath[len(c.activePath)-1] != frontier {
			t.Fatal("pressure discarded current frontier")
		}
	})
}

func TestPhysicalHistoryLimitCreditsReleasedStorage(t *testing.T) {
	for _, tc := range []struct {
		name                              string
		immediate, before, reusable, want int64
	}{
		{"pool covered shortfall", -32, 1064, 0, 1048},
		{"pool covered part of shortfall", -128, 1032, 0, 904},
		{"pool not returned yet", -32, 1064, 64, 968},
	} {
		t.Run(tc.name, func(t *testing.T) {
			memory := discover.MemoryHeadroom{Immediate: tc.immediate}
			if got := physicalHistoryLimit(memory, tc.before, 1000, tc.reusable, 16); got != tc.want {
				t.Fatalf("history limit = %d, want %d", got, tc.want)
			}
		})
	}
}
