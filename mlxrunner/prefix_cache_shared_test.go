package mlxrunner

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

func TestPrefixCacheSharesSavedPrefix(t *testing.T) {
	for name, newEnv := range map[string]func() *testEnv{
		"full":      newTransformerEnv,
		"sliding":   newSlidingWindowEnv,
		"recurrent": newRecurrentEnv,
		"stateless": newStatelessLayerEnv,
	} {
		t.Run(name, func(t *testing.T) {
			source, destination := newEnv(), newEnv()
			defer destination.pc.close()
			inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
			// Keep the source request active and ahead of its saved boundary.
			active := source.pc.begin(inputs, nil)
			active.schedulePrefillSnapshots([]int{4})
			feedAll(source.caches, inputs[:8])
			active.attachPrefillSnapshots()
			forkTokens := []int32{1, 2, 3, 4, 30, 31, 32}
			fork := destination.pc.begin(forkTokens, nil, source.pc)
			if got := len(forkTokens) - len(fork.remaining); got != 4 {
				t.Fatalf("reused %d tokens, want saved boundary 4", got)
			}
			destination.assertAllTokens(t, "restored fork", inputs[:4])
			source.assertAllTokens(t, "unmodified source", inputs[:8])
			// The source can finish and be destroyed without invalidating the fork.
			active.close()
			source.pc.close()
			feedAll(destination.caches, []int32{30, 31})
			destination.assertAllTokens(t, "independent fork", []int32{1, 2, 3, 4, 30, 31})
			fork.close()
			for _, snapshot := range source.tracker.all {
				if snapshot.closeCount != 1 {
					t.Fatalf("source snapshot closed %d times, want once", snapshot.closeCount)
				}
			}
		})
	}
}

func TestPrefixCacheSharedPrefixSelection(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		local, first, second int
		want                 int
	}{
		{"longest peer", 0, 4, 6, 6},
		{"local wins", 8, 4, 6, 8},
		{"peer beats local", 4, 6, 2, 6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
			local, first, second := newRecurrentEnv(), newRecurrentEnv(), newRecurrentEnv()
			for i, env := range []*testEnv{local, first, second} {
				defer env.pc.close()
				n := []int{tc.local, tc.first, tc.second}[i]
				if n > 0 {
					simulateRequest(t, env.pc, inputs[:n+1], nil, n)
				}
			}
			session := local.pc.begin(inputs, nil, first.pc, second.pc)
			if got := len(inputs) - len(session.remaining); got != tc.want {
				t.Fatalf("reused %d, want %d", got, tc.want)
			}
			local.assertAllTokens(t, "chosen prefix", inputs[:tc.want])
			session.close()
		})
	}
}

func TestPrefixCacheSharedPrefixStopsAtMissingEdge(t *testing.T) {
	source, destination := newRecurrentEnv(), newRecurrentEnv()
	defer source.pc.close()
	defer destination.pc.close()
	inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	simulateRequest(t, source.pc, inputs, nil, 4, 8)
	path, _ := findBestMatch(source.pc.root, source.pc.key(effectiveKeyTokens(inputs, nil)))
	path[1].setSnapshots(nil, &source.pc.pagedOutBytes)
	session := destination.pc.begin(inputs, nil, source.pc)
	if !slices.Equal(session.remaining, inputs) {
		t.Fatalf("reused an incomplete snapshot path: remaining %v", session.remaining)
	}
	session.close()
}

func TestSharedPrefixSurvivesIntermediateEviction(t *testing.T) {
	for name, newEnv := range map[string]func() *testEnv{
		"full": newTransformerEnv, "sliding": newSlidingWindowEnv, "recurrent": newRecurrentEnv,
	} {
		t.Run(name, func(t *testing.T) {
			source, destination := newEnv(), newEnv()
			defer source.pc.close()
			defer destination.pc.close()
			inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
			simulateRequest(t, source.pc, inputs, nil, 4, 8)
			path, _ := findBestMatch(source.pc.root, source.pc.key(effectiveKeyTokens(inputs, nil)))
			if len(path) != 3 {
				t.Fatalf("saved path has %d nodes, want root and two checkpoints", len(path))
			}
			// Eviction merges the earlier edge into its child: continuous KV
			// history must remain reusable, even though the earlier checkpoint is gone.
			source.pc.evictNode(path[1])
			path, _ = findBestMatch(source.pc.root, source.pc.key(effectiveKeyTokens(inputs, nil)))
			if len(path) != 2 || path[1].endOffset != 8 || !hasAllSnapshots(path[1], source.caches) {
				t.Fatal("eviction did not preserve a complete merged checkpoint at 8")
			}
			fork := destination.pc.begin(inputs, nil, source.pc)
			defer fork.close()
			if got := len(inputs) - len(fork.remaining); got != 8 {
				t.Fatalf("reused %d tokens after intermediate eviction, want 8", got)
			}
			destination.assertAllTokens(t, "merged peer restore", inputs[:8])
			checkTrieInvariants(t, source.pc.root)
		})
	}
}

func TestSharedPrefixWhileDonorHasPendingCaptures(t *testing.T) {
	for name, newEnv := range map[string]func() *testEnv{
		"full": newTransformerEnv, "sliding": newSlidingWindowEnv, "recurrent": newRecurrentEnv,
	} {
		t.Run(name, func(t *testing.T) {
			source, destination := newEnv(), newEnv()
			defer source.pc.close()
			defer destination.pc.close()
			old := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
			simulateRequest(t, source.pc, old, nil, 4, 8)
			// Give the fake snapshots a cost so borrowing triggers the donor's
			// normal eviction policy while its new captures are still pending.
			walkNodes(source.pc.root, func(n *trieNode) bool {
				for _, s := range n.snapshots {
					if s != nil {
						s.(*fakeSnapshot).byteSize = 32
						source.pc.pagedOutBytes += 32
					}
				}
				return true
			})
			next := []int32{1, 2, 3, 4, 30, 31, 32, 33, 34}
			active := source.pc.begin(next, nil)
			active.schedulePrefillSnapshots([]int{6, 8})
			feedAll(source.caches, next[4:6])
			source.pc.limitBytes = 1
			fork := destination.pc.begin(old, nil, source.pc)
			defer fork.close()
			if got := len(old) - len(fork.remaining); got != 8 {
				t.Fatalf("reused %d tokens from old conversation, want 8", got)
			}
			_, matched := findBestMatch(source.pc.root, source.pc.key(effectiveKeyTokens(old, nil)))
			if matched >= 8 {
				t.Fatal("donor eviction did not remove the borrowed old checkpoint")
			}
			feedAll(source.caches, next[6:8])
			active.attachPrefillSnapshots()
			active.close()
			source.assertAllTokens(t, "continued donor prefill", next[:8])
			destination.assertAllTokens(t, "independent old prefix", old[:8])
			if len(active.pendingSnapshots) != 0 {
				t.Fatal("donor did not drain its scheduled captures")
			}
			checkTrieInvariants(t, source.pc.root)
		})
	}
}

func TestSharedAttentionPrefixSurvivesSourceWrites(t *testing.T) {
	for _, heads := range []int{1, 2} {
		mlxtest.RunSubtest(t, fmt.Sprint(heads), func(t *mlxtest.T) {
			newCache := func() *prefixCache {
				return newPrefixCache([]cache.Cache{cache.NewKVCache(), cache.NewRotatingKVCache(4)})
			}
			source, destination := newCache(), newCache()
			defer destination.close()
			inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
			active := source.begin(inputs, nil)
			active.schedulePrefillSnapshots([]int{4})
			write := func(pc *prefixCache, values []float32) {
				mlx.Scoped(func() {
					x := mlx.FromValues(slices.Repeat(values, heads), 1, heads, len(values), 1)
					for _, layer := range pc.caches {
						layer.(cache.Attention).Update(nil, x, x)
						mlx.Eval(layer.State()...)
					}
				})
			}
			write(source, []float32{1, 2, 3, 4})
			active.attachPrefillSnapshots()
			write(source, []float32{5, 6, 7, 8})
			fork := destination.begin([]int32{1, 2, 3, 4, 20, 21}, nil, source)
			if got := len(fork.inputs) - len(fork.remaining); got != 4 {
				t.Fatalf("reused %d, want 4", got)
			}
			// Destroying donor snapshots and then its buffers must not change
			// the destination, including one-head views that are contiguous.
			source.close()
			for _, layer := range destination.caches {
				for _, a := range layer.State() {
					got := mlx.Contiguous(a, false).Floats()
					want := slices.Repeat([]float32{1, 2, 3, 4}, heads)
					if !slices.Equal(got, want) {
						t.Fatalf("restored %T state %v, want %v", layer, got, want)
					}
				}
			}
			write(destination, []float32{20})
			fork.close()
		})
	}
}

func TestSharedPrefixDraftLookahead(t *testing.T) {
	for _, changeNext := range []bool{false, true} {
		t.Run(fmt.Sprint(changeNext), func(t *testing.T) {
			source, destination := newHybridEnv(), newHybridEnv()
			defer source.pc.close()
			defer destination.pc.close()
			tokens := (&tokenStream{}).fresh(24)
			source.runRequest(t, tokens, []int32{50, 51})
			fork := slices.Clone(tokens)
			want := 19
			if changeNext {
				// State at19 depends on token19 through the next-token drafter.
				fork[19] = 100
				want = 15
			}
			session := destination.pc.begin(fork, nil, source.pc)
			if got := len(fork) - len(session.remaining); got != want {
				t.Fatalf("reused %d, want %d", got, want)
			}
			for _, c := range destination.pc.caches {
				assertTokens(t, "lookahead fork", c, tokens[:want])
			}
			session.close()
		})
	}
}

func TestSharedPrefixMediaIdentity(t *testing.T) {
	inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	items := []mediaItem{{pos: 4, length: 3, fold: 1 << 31, item: &model.PreparedItem{}}}
	source := newRecurrentEnv()
	defer source.pc.close()
	active := source.pc.begin(inputs, items)
	active.schedulePrefillSnapshots([]int{4, 7})
	feedAll(source.caches, inputs[:8])
	active.attachPrefillSnapshots()
	for _, tc := range []struct {
		name           string
		fold           uint32
		maxReuse, want int
	}{
		{"same image", 1 << 31, 8, 7},
		{"different image", 1<<31 | 1, 8, 4},
		{"inside atomic image", 1 << 31, 6, 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			destination := newRecurrentEnv()
			defer destination.pc.close()
			media := slices.Clone(items)
			media[0].fold = tc.fold
			session := destination.pc.beginAt(inputs, media, tc.maxReuse, source.pc)
			if got := len(inputs) - len(session.remaining); got != tc.want {
				t.Fatalf("reused %d, want %d", got, tc.want)
			}
			session.close()
		})
	}
	active.close()
}

func TestSharedPrefixSurvivesCancelledFork(t *testing.T) {
	source, destination := newRecurrentEnv(), newRecurrentEnv()
	defer destination.pc.close()
	inputs := []int32{1, 2, 3, 4, 5}
	simulateRequest(t, source.pc, inputs, nil, 4)
	fork := destination.pc.begin(inputs, nil, source.pc)
	fork.close() // Cancel before forwarding any new tokens.
	source.pc.close()
	retry := destination.pc.begin(inputs, nil)
	defer retry.close()
	if got := len(inputs) - len(retry.remaining); got != 4 {
		t.Fatalf("retry reused %d tokens, want 4 without the source", got)
	}
	destination.assertAllTokens(t, "cancelled fork", inputs[:4])
}

func TestSharedPrefixRestoreFailureRecomputes(t *testing.T) {
	source, destination := newStatelessLayerEnv(), newStatelessLayerEnv()
	defer source.pc.close()
	defer destination.pc.close()
	inputs := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	simulateRequest(t, source.pc, inputs, nil, 4, 8)
	path, _ := findBestMatch(source.pc.root, source.pc.key(effectiveKeyTokens(inputs, nil)))
	// The last attention snapshot cannot continue from the earlier edge.
	// Recurrent state restores first, then cannot rewind to the attention
	// offset after that layer rejects its snapshot. All layers must reset.
	path[len(path)-1].snapshots[2].(*fakeSnapshot).from = 7
	fork := destination.pc.begin(inputs, nil, source.pc)
	if !slices.Equal(fork.remaining, inputs) {
		t.Fatalf("incomplete restore skipped input: remaining %v", fork.remaining)
	}
	destination.assertAllTokens(t, "failed restore", nil)
	feedAll(destination.caches, inputs[:8])
	fork.close()
	destination.assertAllTokens(t, "recomputed", inputs[:8])
}
