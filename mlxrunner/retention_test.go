package mlxrunner

import (
	"context"
	"fmt"
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
	"github.com/ollama/ollama/mlxrunner/model"
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
			if s.snapshotTimer != nil {
				s.snapshotTimer.Stop()
			}
			if s.memory != nil {
				s.memory.Close()
			}
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
			if s.snapshotTimer == nil || s.memory == nil {
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
		for evictColdHistory([]*prefixCache{pc}, nil, nil) {
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
		if evictColdHistory([]*prefixCache{pc}, nil, nil) {
			t.Fatal("a busy frontier must not be evicted")
		}
		if !evictColdHistory([]*prefixCache{pc}, []*prefixCache{pc}, nil) {
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
		for evictColdHistory([]*prefixCache{pc}, []*prefixCache{pc}, nil) {
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
	if !evictColdHistory([]*prefixCache{recent.pc, old.pc}, nil, nil) {
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
		trimHistory([]*prefixCache{pc}, []*prefixCache{pc}, nil, discover.MemoryHeadroom{AvailableRAM: int64(pending + pc.liveBytes())}, 0)
		if len(pc.root.children) != 1 {
			t.Fatal("history evicted despite sufficient headroom")
		}
		trimHistory([]*prefixCache{pc}, []*prefixCache{pc}, nil, discover.MemoryHeadroom{}, math.MaxInt64/2)
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
	if slot.activeCache() != nil || !evictColdHistory([]*prefixCache{env.pc}, []*prefixCache{env.pc}, nil) {
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
		trimHistory([]*prefixCache{c}, nil, nil, discover.MemoryHeadroom{AvailableRAM: pending}, int64(mlx.CacheMemory())+saved)
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

func TestRetainedHistoryStopsAfterMeetingLimit(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.GPUIsAvailable() {
			t.Skip("GPU allocator counters required")
		}
		c := nativeRetentionHistory()
		defer c.close()
		before := mlx.ActiveMemory()
		old := c.activePath[1].snapshotBytes()
		if old == 0 {
			t.Fatal("fixture needs owned checkpoint storage")
		}
		// One eviction releases more than this shortfall. Another checkpoint is
		// eligible, but must survive once current usage meets the fixed limit.
		storageBefore := int64(before + mlx.CacheMemory())
		memory := discover.MemoryHeadroom{AvailableRAM: pendingHistoryBytes([]*prefixCache{c})}
		reserve := int64(mlx.CacheMemory()) + 1
		trimHistory([]*prefixCache{c}, nil, nil, memory, reserve)
		if freed := int64(before - mlx.ActiveMemory()); freed < old {
			t.Fatalf("released %d bytes, want at least the first checkpoint's %d", freed, old)
		}
		if len(c.activePath) != 3 || c.activePath[1].endOffset != 8 {
			t.Fatal("eviction discarded the next checkpoint after meeting the limit")
		}
		after := mlx.ActiveMemory()
		// A later host observation credits the storage returned by the first trim.
		memory.AvailableRAM += storageBefore - int64(after+mlx.CacheMemory())
		trimHistory([]*prefixCache{c}, nil, nil, memory, reserve)
		if mlx.ActiveMemory() != after || len(c.activePath) != 3 {
			t.Fatal("rechecking the same limit discarded more history")
		}
	})
}

func TestRetentionUpdatesIdleCopyReserve(t *testing.T) {
	for _, shortfall := range []bool{false, true} {
		mlxtest.RunSubtest(t, fmt.Sprintf("physical_shortfall=%v", shortfall), func(t *mlxtest.T) {
			if !mlx.GPUIsAvailable() {
				t.Skip("GPU allocator counters required")
			}
			makeHistory := func(count int) *prefixCache {
				c := newPrefixCache([]cache.Cache{cache.NewKVCache()})
				c.ensureRoot()
				tokens := make([]int32, count)
				for i := range tokens {
					tokens[i] = int32(i + 1)
				}
				feedNativeRetention(c, tokens)
				n := c.advancePath(c.root, c.key(effectiveKeyTokens(tokens, nil)), count)
				c.pageOut(n)
				return c
			}
			before := mlx.ActiveMemory()
			large := makeHistory(1024)
			defer large.close()
			largeBytes := int64(mlx.ActiveMemory() - before)
			small := makeHistory(4)
			defer small.close()
			large.activePath[1].lastUsed = time.Unix(1, 0)
			small.activePath[1].lastUsed = time.Unix(2, 0)
			if large.liveBytes() <= small.liveBytes() {
				t.Fatal("fixture needs unequal copy reserves")
			}
			mlx.ClearCache()
			caches := []*prefixCache{large, small}
			memory := discover.MemoryHeadroom{
				AvailableRAM: int64(large.liveBytes()) + pendingHistoryBytes([]*prefixCache{small}),
			}
			// With the old copy reserve, the target remains one byte below the
			// smaller cache's usage even after all of the larger cache is freed.
			// Dropping the obsolete reserve must let that smaller history survive.
			reserve := largeBytes + 1
			if shortfall {
				// A real physical shortfall still requires the smaller cache to go;
				// lowering the copy reserve must not relax this separate constraint.
				immediate := -largeBytes - 1
				memory.FreeMemoryAboveTarget = &immediate
			}
			trimHistory(caches, caches, nil, memory, reserve)
			if large.minCacheOffset() != 0 {
				t.Fatal("oldest, larger history was not released")
			}
			if shortfall {
				if small.minCacheOffset() != 0 {
					t.Fatal("copy-reserve reduction ignored the physical shortfall")
				}
				return
			}
			if small.minCacheOffset() != 4 {
				t.Fatal("evicted the smaller history after the larger cache and its copy reserve were released")
			}
			session := small.beginScore([]int32{1, 2, 3, 4}, nil)
			defer session.close()
			if len(session.remaining) != 0 {
				t.Fatal("surviving history could not be reused")
			}
		})
	}
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
		if !evictColdHistory([]*prefixCache{c}, nil, nil) {
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

func TestCompressionReclaimsOwnedHistory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		frontier := c.activePath[len(c.activePath)-1]
		old := c.activePath[1].snapshotBytes()
		if got := reclaimColdHistory([]*prefixCache{c}, nil, 0, nil); got != 0 || len(c.activePath) != 4 {
			t.Fatal("unchanged compressor triggered reclamation")
		}
		if got := reclaimColdHistory([]*prefixCache{c}, nil, old, nil); got < old {
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
		{"queued allocations completed", 500, 600, 0, 584},
	} {
		t.Run(tc.name, func(t *testing.T) {
			memory := discover.MemoryHeadroom{FreeMemoryAboveTarget: &tc.immediate}
			if got := physicalHistoryLimit(memory, tc.before, 1000, tc.reusable, 16); got != tc.want {
				t.Fatalf("history limit = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestAdmissionPrefersMatchingHistory(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		pc := env.pc
		incoming := []int32{1, 2, 3, 4, 5, 6, 7, 8}
		unrelated := []int32{11, 12, 13, 14, 15, 16, 17, 18}
		current := []int32{21, 22, 23, 24, 25, 26, 27, 28}
		simulateRequest(t, pc, incoming, []int32{9, 10}, 4)
		simulateRequest(t, pc, unrelated, []int32{19, 20}, 4)
		simulateRequest(t, pc, current, []int32{29, 30}, 4)
		for i, child := range pc.root.children {
			walkNodes(child, func(n *trieNode) bool {
				n.lastUsed = time.Unix(int64(i+1), 0)
				return true
			})
		}
		s := &batchScheduler{slots: []*batchSlot{{cache: pc}}}
		preferred := s.matchingHistory(request{Tokens: incoming}, 0)
		before, matched := findBestMatch(pc.root, pc.key(effectiveKeyTokens(incoming, nil)))
		if !evictColdHistory([]*prefixCache{pc}, []*prefixCache{pc}, preferred) {
			t.Fatal("did not reclaim unrelated history")
		}
		after, remaining := findBestMatch(pc.root, pc.key(effectiveKeyTokens(incoming, nil)))
		if !slices.Equal(before, after) || matched != remaining {
			t.Fatalf("admission changed the arriving prefix: %d keys -> %d", matched, remaining)
		}
		session := pc.begin(incoming, nil)
		defer session.close()
		if got := len(incoming) - len(session.remaining); got < 4-pc.draftLookahead {
			t.Fatalf("retained prefix only reused %d tokens", got)
		}
		env.assertAllTokens(t, "incoming prefix", incoming[:len(incoming)-len(session.remaining)])
		checkTrieInvariants(t, pc.root)
	})
}

func TestAdmissionPrefersPeerHistory(t *testing.T) {
	for _, score := range []bool{false, true} {
		t.Run(fmt.Sprintf("score=%v", score), func(t *testing.T) {
			donor, idle := newTransformerEnv(), newTransformerEnv()
			defer donor.pc.close()
			defer idle.pc.close()
			incoming := []int32{1, 2, 3, 4, 5}
			simulateRequest(t, donor.pc, incoming, nil, 4)
			simulateRequest(t, idle.pc, []int32{1, 12, 13, 14, 15}, nil, 4)
			s := &batchScheduler{slots: []*batchSlot{{}, {}}}
			var req queuedRequest = request{Tokens: incoming}
			if score {
				s.slots[0].scoreCache, s.slots[1].scoreCache = donor.pc, idle.pc
				req = &evaluationRequest{tokens: incoming}
			} else {
				s.slots[0].cache, s.slots[1].cache = donor.pc, idle.pc
			}
			preferred := s.matchingHistory(req, 1)
			// The donor is older, but the unrelated idle frontier should go first.
			caches := []*prefixCache{donor.pc, idle.pc}
			if !evictColdHistory(caches, caches, preferred) || idle.pc.minCacheOffset() != 0 {
				t.Fatal("did not evict the unrelated idle frontier")
			}
			var session *cacheSession
			if score {
				session = idle.pc.beginScore(incoming, nil, donor.pc)
			} else {
				session = idle.pc.begin(incoming, nil, donor.pc)
			}
			if got := len(incoming) - len(session.remaining); got != 4 {
				t.Fatalf("peer reused %d tokens, want 4", got)
			}
			idle.assertAllTokens(t, "peer history", incoming[:4])
			session.close()
			// Preference must not become an exemption from severe pressure.
			for evictColdHistory(caches, caches, preferred) {
			}
			if donor.pc.minCacheOffset() != 0 || idle.pc.minCacheOffset() != 0 {
				t.Fatal("incoming prefix prevented pressure reclamation")
			}
		})
	}
}

func TestAdmissionDoesNotPreferUnrestorablePeerMatch(t *testing.T) {
	for _, tc := range []struct {
		name  string
		score bool
		busy  bool
	}{
		{"generation busy donor", false, true},
		{"scoring busy donor", true, true},
		{"scoring idle donor", true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for name, newEnv := range map[string]func() *testEnv{
				"full": newTransformerEnv, "sliding": newSlidingWindowEnv, "recurrent": newRecurrentEnv,
			} {
				t.Run(name, func(t *testing.T) {
					donor, local := newEnv(), newEnv()
					defer donor.pc.close()
					defer local.pc.close()
					incoming := []int32{1, 2, 3, 4, 5, 6, 30, 31}
					// The donor matches six tokens, but its first checkpoint is at
					// eight. The destination has a usable four-token checkpoint.
					simulateRequest(t, donor.pc, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}, nil, 8)
					simulateRequest(t, local.pc, []int32{1, 2, 3, 4, 20}, nil, 4)
					donorSlot, localSlot := &batchSlot{}, &batchSlot{}
					var req queuedRequest = request{Tokens: incoming}
					if tc.score {
						donorSlot.scoreCache, localSlot.scoreCache = donor.pc, local.pc
						req = &evaluationRequest{tokens: incoming}
					} else {
						donorSlot.cache, localSlot.cache = donor.pc, local.pc
					}
					s := &batchScheduler{slots: []*batchSlot{localSlot, donorSlot}}
					if tc.busy {
						// Still draining a response: cannot accept work, but its
						// completed cache is eligible for pressure reclamation.
						donorSlot.request = &batchRequest{finished: true}
						s.slots = []*batchSlot{donorSlot, localSlot}
					}
					destination := s.selectSlot(req)
					if s.slots[destination] != localSlot {
						t.Fatal("request did not select the available destination")
					}
					preferred := s.matchingHistory(req, destination)
					caches := []*prefixCache{donor.pc, local.pc}
					if !evictColdHistory(caches, caches, preferred) {
						t.Fatal("no idle history was evicted")
					}
					var session *cacheSession
					if tc.score {
						session = local.pc.beginScore(incoming, nil, donor.pc)
					} else {
						session = local.pc.begin(incoming, nil, donor.pc)
					}
					defer session.close()
					if reused := len(incoming) - len(session.remaining); reused != 4 {
						t.Fatalf("reused %d tokens after trim, want 4", reused)
					}
					local.assertAllTokens(t, "preserved destination", incoming[:4])
				})
			}
		})
	}
}

func TestAdmissionPeerPreferenceUsesRestoreBoundary(t *testing.T) {
	for _, tc := range []struct {
		name      string
		score     bool
		lookahead int
		items     []mediaItem
		want      int
	}{
		{"generation seed", false, 0, nil, 4},
		{"draft lookahead", false, 1, nil, 7},
		{"complete scoring row", true, 0, nil, 8},
		{"atomic media", false, 0, []mediaItem{{pos: 3, length: 5, fold: 100, item: &model.PreparedItem{}}}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			donor := newRecurrentEnv()
			defer donor.pc.close()
			donor.pc.draftLookahead = tc.lookahead
			inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8}
			session := donor.pc.begin(append(slices.Clone(inputs), 9), tc.items)
			session.schedulePrefillSnapshots([]int{4, 8})
			feedAll(donor.caches, inputs)
			session.attachPrefillSnapshots()
			session.close()
			incoming := inputs
			s := &batchScheduler{slots: []*batchSlot{{}, {request: &batchRequest{finished: true}}}}
			var req queuedRequest = request{Tokens: incoming, MediaItems: tc.items}
			if tc.score {
				s.slots[1].scoreCache = donor.pc
				req = &evaluationRequest{tokens: incoming, items: tc.items}
			} else {
				s.slots[1].cache = donor.pc
			}
			preferred := s.matchingHistory(req, s.selectSlot(req))
			longest := 0
			for _, matched := range preferred {
				longest = max(longest, matched)
			}
			if longest != tc.want {
				t.Fatalf("peer preference = %d tokens, want %d", longest, tc.want)
			}
		})
	}
}

func TestPressureReclamationWithPendingWork(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.GPUIsAvailable() {
			t.Skip("requires GPU allocator counters")
		}
		c := nativeRetentionHistory()
		defer c.close()
		var outputs []*mlx.Array
		var expected [][]float32
		for _, layer := range c.caches {
			for _, state := range layer.State() {
				want := mlx.Contiguous(state, false).Floats()
				for i := range want {
					want[i]++
				}
				expected = append(expected, want)
				outputs = append(outputs, mlx.Add(state, mlx.FromValue(float32(1))))
			}
		}
		mlx.AsyncEval(outputs...)
		// Force reclamation without stressing the host. Submitted consumers must
		// finish before all idle snapshots and live cache buffers are discarded.
		trimHistory([]*prefixCache{c}, []*prefixCache{c}, nil, discover.MemoryHeadroom{}, math.MaxInt64/2)
		if c.minCacheOffset() != 0 || c.pagedOutBytes != 0 {
			t.Fatal("pressure did not release idle history")
		}
		for i, output := range outputs {
			if got := output.Floats(); !slices.Equal(got, expected[i]) {
				t.Fatalf("submitted consumer %d changed after reclamation: got %v, want %v", i, got, expected[i])
			}
		}
	})
}

func TestAdmissionHistoryUsesMediaAndLookahead(t *testing.T) {
	for _, lookahead := range []int{0, 1} {
		c := newPrefixCache(nil)
		c.draftLookahead = lookahead
		c.ensureRoot()
		tokens := []int32{1, 2, 3, 4}
		items := []mediaItem{{pos: 1, length: 2, fold: 100, item: &model.PreparedItem{}}}
		n := c.root.appendChild(c.key(effectiveKeyTokens(tokens, items)), 4-lookahead)
		s := &batchScheduler{slots: []*batchSlot{{cache: c}}}
		if s.matchingHistory(request{Tokens: tokens, MediaItems: items}, 0)[n] == 0 {
			t.Fatalf("lookahead %d: matching media prefix was not preferred", lookahead)
		}
		items[0].fold++
		// A shared first key still earns a bounded preference at lookahead zero;
		// lookahead one includes the changed media in its very first key.
		if lookahead == 1 && s.matchingHistory(request{Tokens: tokens, MediaItems: items}, 0)[n] != 0 {
			t.Fatal("changed media incorrectly matched the draft key")
		}
	}
}

func TestCompressionPreservesIdleFrontier(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.GPUIsAvailable() {
			t.Skip("requires GPU allocator counters")
		}
		incoming, unrelated := nativeRetentionHistory(), nativeRetentionHistory()
		defer incoming.close()
		defer unrelated.close()
		for len(unrelated.activePath) > 2 {
			unrelated.evictNode(unrelated.activePath[1])
		}
		mlx.DefaultStream().Synchronize()
		preferred := make(map[*trieNode]int)
		before := slices.Clone(incoming.activePath)
		for _, n := range before {
			preferred[n] = 12
		}
		caches := []*prefixCache{incoming, unrelated}
		if freed := reclaimColdHistory(caches, caches, 1, preferred); freed != 0 {
			t.Fatalf("compression reclaimed %d bytes past the idle frontier", freed)
		}
		if !slices.Equal(before, incoming.activePath) || unrelated.minCacheOffset() == 0 {
			t.Fatal("compression discarded the frontier or skipped it to evict incoming history")
		}
		if freed := reclaimColdHistory([]*prefixCache{unrelated}, []*prefixCache{unrelated}, 1, nil); freed != 0 {
			t.Fatalf("idle compression check reclaimed the last restore point: %d bytes", freed)
		}
		memory := discover.MemoryHeadroom{AvailableRAM: pendingHistoryBytes(caches) + idleCopyBytes(caches)}
		trimHistory(caches, caches, preferred, memory, int64(mlx.CacheMemory())+1)
		if !slices.Equal(before, incoming.activePath) || unrelated.minCacheOffset() != 0 {
			t.Fatal("physical shortfall did not reclaim the unrelated idle frontier first")
		}
	})
}

func TestAdmissionPrefersRestorablePrefillWidth(t *testing.T) {
	const firstBatch, common = 2048, 2100
	for _, tc := range []struct {
		name    string
		aligned bool
		want    int
	}{
		{"ordinary", false, common},
		{"aligned", true, firstBatch},
	} {
		for _, peer := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/peer=%v", tc.name, peer), func(t *testing.T) {
				src, dst := newRecurrentEnv().pc, newRecurrentEnv().pc
				t.Cleanup(src.close)
				t.Cleanup(dst.close)
				src.alignPrefill, dst.alignPrefill = tc.aligned, tc.aligned
				tokens := slices.Repeat([]int32{1}, 4400)
				// A complete snapshot inside the second prefill batch is usable only
				// when the model's restoration contract accepts arbitrary widths.
				simulateRequest(t, src, tokens, nil, firstBatch, common, 4096)
				tokens[common] = 2
				req := request{Tokens: tokens}
				s := &batchScheduler{slots: []*batchSlot{{cache: dst}, {cache: src}}}
				wantSlot := 1
				if peer {
					s.slots[1].request = &batchRequest{finished: true}
					wantSlot = 0
				}
				dest := s.selectSlot(req)
				if dest != wantSlot {
					t.Fatalf("selected slot %d, want %d", dest, wantSlot)
				}
				longest := 0
				for _, n := range s.matchingHistory(req, dest) {
					longest = max(longest, n)
				}
				if longest != tc.want {
					t.Fatalf("preferred %d tokens under pressure, want %d", longest, tc.want)
				}
				session := s.slots[dest].cache.begin(tokens, nil, s.slots[1-dest].cache)
				defer session.close()
				if got := len(tokens) - len(session.remaining); got != longest {
					t.Fatalf("restored %d tokens, preferred %d", got, longest)
				}
			})
		}
	}
}
