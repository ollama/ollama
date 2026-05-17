package mlxrunner

import (
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// snapshotTracker records every fakeSnapshot created and every Close() call
// so tests can detect leaked (created but never closed) or double-closed snapshots.
type snapshotTracker struct {
	all []*fakeSnapshot
}

func (tr *snapshotTracker) track(s *fakeSnapshot) {
	if s == nil {
		return
	}
	s.tracker = tr
	tr.all = append(tr.all, s)
}

// Fake caches that store actual token sequences so tests can verify the right
// data was restored, not just the right offset.

// fakeSnapshot stores a copy of the token sub-sequence it covers.
type fakeSnapshot struct {
	tokens   []int32
	from, to int
	byteSize int // configurable for eviction tests

	tracker    *snapshotTracker
	closeCount int
}

func (s *fakeSnapshot) Size() int { return s.byteSize }
func (s *fakeSnapshot) Close() {
	s.closeCount++
}

// fakeRewindableCache tracks the full token sequence and supports
// arbitrary rewind via Restore(nil, target).
type fakeRewindableCache struct {
	tokens  []int32
	tracker *snapshotTracker
}

func (c *fakeRewindableCache) feed(tokens []int32) {
	c.tokens = append(c.tokens, tokens...)
}

func (c *fakeRewindableCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	return nil, nil
}
func (c *fakeRewindableCache) State() []*mlx.Array { return nil }
func (c *fakeRewindableCache) Offset() int         { return len(c.tokens) }

func (c *fakeRewindableCache) Free() {
	c.tokens = nil
}

func (c *fakeRewindableCache) Snapshot(fromOffset int) cache.Snapshot {
	if fromOffset >= len(c.tokens) {
		return nil
	}
	from := fromOffset
	if from < 0 {
		from = 0
	}
	s := &fakeSnapshot{
		tokens: slices.Clone(c.tokens[from:]),
		from:   from,
		to:     len(c.tokens),
	}
	c.tracker.track(s)
	return s
}

func (c *fakeRewindableCache) Restore(snapshot cache.Snapshot, target int) bool {
	if target < 0 {
		return false
	}

	if snapshot == nil {
		if target > len(c.tokens) {
			return false
		}
		c.tokens = c.tokens[:target]
		return true
	}
	s := snapshot.(*fakeSnapshot)
	if target > s.to || len(c.tokens) < s.from {
		return false
	}
	c.tokens = append(c.tokens[:s.from], s.tokens...)
	if target < len(c.tokens) {
		c.tokens = c.tokens[:target]
	}
	return true
}

func (c *fakeRewindableCache) Merge(parent, child cache.Snapshot) cache.Snapshot {
	if parent == nil || child == nil {
		if parent != nil {
			parent.Close()
		}
		if child != nil {
			child.Close()
		}
		return nil
	}
	p := parent.(*fakeSnapshot)
	ch := child.(*fakeSnapshot)
	merged := make([]int32, len(p.tokens)+len(ch.tokens))
	copy(merged, p.tokens)
	copy(merged[len(p.tokens):], ch.tokens)
	s := &fakeSnapshot{
		tokens:   merged,
		from:     p.from,
		to:       ch.to,
		byteSize: p.byteSize + ch.byteSize,
	}
	c.tracker.track(s)
	p.Close()
	ch.Close()
	return s
}

func (c *fakeRewindableCache) Split(snapshot cache.Snapshot, at int) (cache.Snapshot, cache.Snapshot) {
	if snapshot == nil {
		return nil, nil
	}
	s := snapshot.(*fakeSnapshot)
	relAt := at - s.from
	if relAt <= 0 {
		return nil, snapshot
	}
	if relAt >= len(s.tokens) {
		return snapshot, nil
	}
	p := &fakeSnapshot{
		tokens:   slices.Clone(s.tokens[:relAt]),
		from:     s.from,
		to:       at,
		byteSize: s.byteSize,
	}
	ch := &fakeSnapshot{
		tokens:   slices.Clone(s.tokens[relAt:]),
		from:     at,
		to:       s.to,
		byteSize: s.byteSize,
	}
	c.tracker.track(p)
	c.tracker.track(ch)
	s.Close()
	return p, ch
}

func TestKVCacheBeginWithFactoryLimitCapsPrefix(t *testing.T) {
	inputs := []int32{1, 2, 3, 4, 5}
	tracker := &snapshotTracker{}
	var kc kvCache

	factory := func() []cache.Cache {
		return []cache.Cache{&fakeRewindableCache{tracker: tracker}}
	}

	session := kc.beginWithFactoryLimit(inputs, factory, "test", -1, false)
	session.caches[0].(*fakeRewindableCache).feed(inputs)
	session.close()

	session = kc.beginWithFactoryLimit(inputs, factory, "test", 3, false)
	if got, want := session.caches[0].Offset(), 3; got != want {
		t.Fatalf("cache offset = %d, want %d", got, want)
	}
	if got, want := session.remaining, inputs[3:]; !slices.Equal(got, want) {
		t.Fatalf("remaining = %v, want %v", got, want)
	}
}

func TestKVCacheBeginWithFactoryLimitDoesNotKeepSeedToken(t *testing.T) {
	inputs := []int32{1, 2, 3, 4, 5}
	tracker := &snapshotTracker{}
	var kc kvCache

	factory := func() []cache.Cache {
		return []cache.Cache{&fakeRewindableCache{tracker: tracker}}
	}

	session := kc.beginWithFactoryLimit(inputs, factory, "test", -1, false)
	session.caches[0].(*fakeRewindableCache).feed(inputs)
	session.close()

	session = kc.beginWithFactoryLimit(inputs, factory, "test", len(inputs), false)
	if got, want := session.caches[0].Offset(), len(inputs); got != want {
		t.Fatalf("cache offset = %d, want %d", got, want)
	}
	if len(session.remaining) != 0 {
		t.Fatalf("remaining = %v, want empty", session.remaining)
	}
}

// fakeSlidingWindowCache models RotatingKVCache semantics: stores the full
// token sequence but only the trailing maxSize tokens are "live" in the window.
// Once the window fills, live rewind is impossible without a snapshot.
type fakeSlidingWindowCache struct {
	tokens  []int32
	maxSize int
	tracker *snapshotTracker
}

func (c *fakeSlidingWindowCache) feed(tokens []int32) {
	c.tokens = append(c.tokens, tokens...)
}

func (c *fakeSlidingWindowCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	return nil, nil
}
func (c *fakeSlidingWindowCache) State() []*mlx.Array { return nil }
func (c *fakeSlidingWindowCache) Offset() int         { return len(c.tokens) }

func (c *fakeSlidingWindowCache) Free() {
	c.tokens = nil
}

func (c *fakeSlidingWindowCache) Snapshot(fromOffset int) cache.Snapshot {
	if len(c.tokens) == 0 || len(c.tokens) <= fromOffset {
		return nil
	}
	// Snapshot captures the full window state (like RotatingKVCache.Snapshot).
	s := &fakeSnapshot{
		tokens: slices.Clone(c.tokens),
		from:   0,
		to:     len(c.tokens),
	}
	c.tracker.track(s)
	return s
}

func (c *fakeSlidingWindowCache) Restore(snapshot cache.Snapshot, target int) bool {
	if target < 0 {
		return false
	}

	if snapshot == nil {
		if target >= len(c.tokens) {
			return target == len(c.tokens)
		}
		// Live rewind only works when buffer hasn't filled (offset <= maxSize).
		if len(c.tokens) > c.maxSize {
			return false
		}
		c.tokens = c.tokens[:target]
		return true
	}
	s := snapshot.(*fakeSnapshot)
	if target > s.to {
		return false
	}
	// Reject if clamping would leave an incomplete window
	// (matches RotatingKVCache behavior).
	if target < s.to && s.to > c.maxSize {
		return false
	}
	c.tokens = slices.Clone(s.tokens)
	if target < len(c.tokens) {
		c.tokens = c.tokens[:target]
	}
	return true
}

func (c *fakeSlidingWindowCache) Merge(parent, child cache.Snapshot) cache.Snapshot {
	// Child supersedes parent for sliding window (full window state).
	if parent != nil {
		parent.Close()
	}
	return child
}

func (c *fakeSlidingWindowCache) Split(snapshot cache.Snapshot, at int) (cache.Snapshot, cache.Snapshot) {
	// Can't split a ring buffer at an arbitrary point.
	return nil, snapshot
}

// fakeRecurrentCache models RecurrentCache semantics: stores tokens
// but cannot rewind without a snapshot.
type fakeRecurrentCache struct {
	tokens  []int32
	tracker *snapshotTracker
}

func (c *fakeRecurrentCache) feed(tokens []int32) {
	c.tokens = append(c.tokens, tokens...)
}

func (c *fakeRecurrentCache) Update(keys, values *mlx.Array) (*mlx.Array, *mlx.Array) {
	return nil, nil
}
func (c *fakeRecurrentCache) State() []*mlx.Array { return nil }
func (c *fakeRecurrentCache) Offset() int         { return len(c.tokens) }

func (c *fakeRecurrentCache) Free() {
	c.tokens = nil
}

func (c *fakeRecurrentCache) Snapshot(fromOffset int) cache.Snapshot {
	// Recurrent state is cumulative; snapshot captures the full state.
	if len(c.tokens) == 0 {
		return nil
	}
	s := &fakeSnapshot{
		tokens: slices.Clone(c.tokens),
		from:   0,
		to:     len(c.tokens),
	}
	c.tracker.track(s)
	return s
}

func (c *fakeRecurrentCache) Restore(snapshot cache.Snapshot, target int) bool {
	if snapshot == nil {
		return target == len(c.tokens) // can only no-op
	}
	s := snapshot.(*fakeSnapshot)
	if target != s.to {
		return false // cumulative state requires exact match
	}
	c.tokens = slices.Clone(s.tokens)
	return true
}

func (c *fakeRecurrentCache) Merge(parent, child cache.Snapshot) cache.Snapshot {
	// Child supersedes parent for cumulative state.
	if parent != nil {
		parent.Close()
	}
	return child
}

func (c *fakeRecurrentCache) Split(snapshot cache.Snapshot, at int) (cache.Snapshot, cache.Snapshot) {
	return nil, snapshot // can't split cumulative state
}

type feedableCache interface {
	cache.Cache
	feed(tokens []int32)
}

// testEnv encapsulates a kvCache and its fake caches for a test scenario.
type testEnv struct {
	kvc        *kvCache
	caches     []cache.Cache // typed references for assertions
	tracker    *snapshotTracker
	rewindable bool // true when all caches support arbitrary Restore(nil, target)
}

// newTransformerEnv creates a test environment with a single rewindable cache
// (pure transformer model).
func newTransformerEnv() *testEnv {
	tracker := &snapshotTracker{}
	caches := []cache.Cache{&fakeRewindableCache{tracker: tracker}}
	return &testEnv{
		kvc:        &kvCache{caches: caches},
		caches:     caches,
		tracker:    tracker,
		rewindable: true,
	}
}

// newSlidingWindowEnv creates a test environment with one rewindable cache and
// one sliding window cache (Mistral-style architecture). The sliding window
// maxSize is set small enough that test sequences fill it, making
// Restore(nil, target) fail — the same behavior as production models where
// the window fills after a few turns.
func newSlidingWindowEnv() *testEnv {
	tr := &snapshotTracker{}
	rc := &fakeRewindableCache{tracker: tr}
	sw := &fakeSlidingWindowCache{maxSize: 4, tracker: tr}
	caches := []cache.Cache{rc, sw}
	return &testEnv{
		kvc:        &kvCache{caches: caches},
		caches:     caches,
		tracker:    tr,
		rewindable: false,
	}
}

// newRecurrentEnv creates a test environment with one rewindable cache and one
// non-rewindable cache (Jamba-style architecture).
func newRecurrentEnv() *testEnv {
	tr := &snapshotTracker{}
	rc := &fakeRewindableCache{tracker: tr}
	nrc := &fakeRecurrentCache{tracker: tr}
	caches := []cache.Cache{rc, nrc}
	return &testEnv{
		kvc:        &kvCache{caches: caches},
		caches:     caches,
		tracker:    tr,
		rewindable: false,
	}
}

// assertAllTokens checks that every cache in the environment contains exactly
// the expected token sequence.
func (e *testEnv) assertAllTokens(t *testing.T, label string, expected []int32) {
	t.Helper()
	for i, c := range e.caches {
		assertTokens(t, label, c, expected)
		// Verify all caches report the same offset.
		if i > 0 && c.Offset() != e.caches[0].Offset() {
			t.Errorf("%s: cache %d offset=%d != cache 0 offset=%d",
				label, i, c.Offset(), e.caches[0].Offset())
		}
	}
}

// simulateRequest mirrors the production pipeline lifecycle:
//   begin -> prefill with snapshot(false) at branch points -> generate -> close

type requestResult struct {
	remaining        []int32
	pendingSnapshots int
}

// simulateRequest runs a request through the harness. If userSnapshotAt > 0,
// a user snapshot is requested at that offset during prefill.
func simulateRequest(t *testing.T, kvc *kvCache, inputs, generated []int32, userSnapshotAt ...int) requestResult {
	t.Helper()

	session := kvc.begin(nil, inputs)
	for _, at := range userSnapshotAt {
		if at > 0 {
			session.requestSnapshot(at)
		}
	}

	result := requestResult{
		remaining:        slices.Clone(session.remaining),
		pendingSnapshots: len(session.pendingSnapshots),
	}

	assertCacheOffsetAlignment(t, kvc, "after begin")

	baseOffset := kvc.minCacheOffset()
	remaining := inputs[baseOffset:]

	// Prefill: feed tokens, pausing at each pending snapshot.
	for len(session.pendingSnapshots) > 0 {
		sp := session.pendingSnapshots[0]
		count := sp.offset - baseOffset
		if count > len(remaining) {
			break
		}
		if count > 0 {
			feedAll(kvc.caches, remaining[:count])
			remaining = remaining[count:]
			baseOffset = sp.offset
		}
		assertCacheOffsetAlignment(t, kvc, "at snapshot point")
		session.snapshot()
	}

	// Feed rest of input tokens.
	if len(remaining) > 0 {
		feedAll(kvc.caches, remaining)
	}

	assertCacheOffsetAlignment(t, kvc, "after prefill")

	// Generate tokens.
	if len(generated) > 0 {
		session.outputs = generated
		feedAll(kvc.caches, generated)
	}

	assertCacheOffsetAlignment(t, kvc, "before close")
	session.close()
	return result
}

func feedAll(caches []cache.Cache, tokens []int32) {
	for _, c := range caches {
		if fc, ok := c.(feedableCache); ok {
			fc.feed(tokens)
		}
	}
}

// assertCacheOffsetAlignment verifies all caches report the same offset.
func assertCacheOffsetAlignment(t *testing.T, kvc *kvCache, label string) {
	t.Helper()
	if len(kvc.caches) < 2 {
		return
	}
	expected := kvc.caches[0].Offset()
	for i := 1; i < len(kvc.caches); i++ {
		if got := kvc.caches[i].Offset(); got != expected {
			t.Errorf("%s: cache %d offset=%d != cache 0 offset=%d", label, i, got, expected)
		}
	}
}

// assertTokens checks that a feedable cache contains the expected token sequence.
// For sliding window caches, only the trailing maxSize tokens are checked.
func assertTokens(t *testing.T, label string, c cache.Cache, expected []int32) {
	t.Helper()
	switch fc := c.(type) {
	case *fakeRewindableCache:
		if !slices.Equal(fc.tokens, expected) {
			t.Errorf("%s: rewindable tokens = %v, want %v", label, fc.tokens, expected)
		}
	case *fakeSlidingWindowCache:
		// Sliding window stores full history but only trailing maxSize are live.
		// Verify the full token sequence matches (the window semantics are
		// enforced by Snapshot/Restore, not by the token log).
		if !slices.Equal(fc.tokens, expected) {
			t.Errorf("%s: sliding window tokens = %v, want %v", label, fc.tokens, expected)
		}
	case *fakeRecurrentCache:
		if !slices.Equal(fc.tokens, expected) {
			t.Errorf("%s: non-rewindable tokens = %v, want %v", label, fc.tokens, expected)
		}
	default:
		t.Fatalf("%s: unknown cache type %T", label, c)
	}
}

// checkTrieInvariants walks the trie and checks structural invariants.
func checkTrieInvariants(t *testing.T, root *trieNode) {
	t.Helper()
	walkNodes(root, func(n *trieNode) bool {
		if n.parent != nil {
			if n.startOffset() != n.parent.endOffset {
				t.Errorf("node [%d,%d): startOffset %d != parent endOffset %d",
					n.startOffset(), n.endOffset, n.startOffset(), n.parent.endOffset)
			}
		}
		if len(n.tokens) != n.endOffset-n.startOffset() {
			t.Errorf("node [%d,%d): token count %d != offset span %d",
				n.startOffset(), n.endOffset, len(n.tokens), n.endOffset-n.startOffset())
		}
		for _, c := range n.children {
			if c.parent != n {
				t.Errorf("child [%d,%d) parent mismatch", c.startOffset(), c.endOffset)
			}
		}
		// No two siblings should start with the same token.
		seen := make(map[int32]bool)
		for _, c := range n.children {
			if len(c.tokens) > 0 {
				first := c.tokens[0]
				if seen[first] {
					t.Errorf("node [%d,%d): duplicate sibling first token %d",
						n.startOffset(), n.endOffset, first)
				}
				seen[first] = true
			}
		}
		return true
	})
}

// checkSnapshotLeaks verifies that every tracked snapshot is either still live
// in the trie (closeCount == 0) or has been closed exactly once. It reports
// leaked snapshots (not in trie, never closed) and double-closes.
func checkSnapshotLeaks(t *testing.T, tracker *snapshotTracker, root *trieNode) {
	t.Helper()
	if tracker == nil {
		return
	}

	// Collect all live snapshots still referenced by trie nodes.
	live := make(map[*fakeSnapshot]bool)
	walkNodes(root, func(n *trieNode) bool {
		for _, s := range n.snapshots {
			if s != nil {
				if fs, ok := s.(*fakeSnapshot); ok {
					live[fs] = true
				}
			}
		}
		return true
	})

	for i, s := range tracker.all {
		if live[s] {
			if s.closeCount != 0 {
				t.Errorf("snapshot #%d [%d,%d) is still in trie but was closed %d time(s)",
					i, s.from, s.to, s.closeCount)
			}
		} else {
			if s.closeCount == 0 {
				t.Errorf("snapshot #%d [%d,%d) leaked: created but never closed and not in trie",
					i, s.from, s.to)
			} else if s.closeCount > 1 {
				t.Errorf("snapshot #%d [%d,%d) double-closed: closed %d times",
					i, s.from, s.to, s.closeCount)
			}
		}
	}
}

// forEachEnv runs fn as subtests for three realistic model configurations:
// pure transformer, transformer + sliding window (Mistral-style), and
// transformer + recurrent (Jamba-style). Leak checking runs automatically
// at the end of each subtest.
func forEachEnv(t *testing.T, fn func(t *testing.T, env *testEnv)) {
	t.Helper()
	run := func(t *testing.T, env *testEnv) {
		t.Cleanup(func() {
			checkSnapshotLeaks(t, env.tracker, env.kvc.root)
		})
		fn(t, env)
	}
	t.Run("Transformer", func(t *testing.T) { run(t, newTransformerEnv()) })
	t.Run("SlidingWindow", func(t *testing.T) { run(t, newSlidingWindowEnv()) })
	t.Run("Recurrent", func(t *testing.T) { run(t, newRecurrentEnv()) })
}

// TestBranchCreationAndReuse exercises the core multi-conversation lifecycle:
// two conversations share a prefix and diverge, creating a branch point.
// A third conversation extends the first. Verifies trie structure, cache
// hit lengths, and that semantic caches contain the correct token sequences.
func TestBranchCreationAndReuse(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc

		// Request A: [1,2,3,4,5,6,7,8] + generate [20,21] — full miss.
		resA := simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 6, 7, 8}, []int32{20, 21})
		if len(resA.remaining) != 8 {
			t.Fatalf("A: remaining = %d, want 8 (full miss)", len(resA.remaining))
		}
		env.assertAllTokens(t, "after A", []int32{1, 2, 3, 4, 5, 6, 7, 8, 20, 21})

		// Verify trie was populated by close().
		_, mA := findBestMatch(kvc.root, []int32{1, 2, 3, 4, 5, 6, 7, 8, 20, 21})
		if mA != 10 {
			t.Fatalf("A findable: expected 10 matched, got %d", mA)
		}

		// Request B: [1,2,3,4,5,10,11,12] — shares 5-token prefix with A.
		// For rewindable caches, switchToPath rewinds to the match point
		// so only the non-matching suffix needs evaluation. For non-rewindable
		// caches (RecurrentCache), the rewind fails and freeAll fires.
		resB := simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 10, 11, 12}, []int32{30, 31})
		if env.rewindable {
			if resB.pendingSnapshots != 0 {
				t.Fatalf("B: pendingSnapshots = %d, want 0 (rewind succeeded)", resB.pendingSnapshots)
			}
			if len(resB.remaining) != 3 {
				t.Fatalf("B: remaining = %d, want 3 (rewind to match point)", len(resB.remaining))
			}
		} else {
			if resB.pendingSnapshots != 1 {
				t.Fatalf("B: pendingSnapshots = %d, want 1", resB.pendingSnapshots)
			}
			if len(resB.remaining) != 8 {
				t.Fatalf("B: remaining = %d, want 8 (freeAll fallback)", len(resB.remaining))
			}
		}
		env.assertAllTokens(t, "after B", []int32{1, 2, 3, 4, 5, 10, 11, 12, 30, 31})

		// Both A and B should be findable in the trie.
		_, mA2 := findBestMatch(kvc.root, []int32{1, 2, 3, 4, 5, 6, 7, 8, 20, 21})
		if mA2 < 5 {
			t.Fatalf("A still findable: expected >= 5 matched, got %d", mA2)
		}
		_, mB := findBestMatch(kvc.root, []int32{1, 2, 3, 4, 5, 10, 11, 12, 30, 31})
		if mB < 5 {
			t.Fatalf("B findable: expected >= 5 matched, got %d", mB)
		}

		// Request C: [1,2,3,4,5,6,7,8,40,41] — extends A's prefix.
		// Should get a cache hit for the shared prefix.
		resC := simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 6, 7, 8, 40, 41}, nil)
		if len(resC.remaining) >= 10 {
			t.Fatalf("C: remaining = %d, want < 10 (should get cache hit)", len(resC.remaining))
		}
		env.assertAllTokens(t, "after C", []int32{1, 2, 3, 4, 5, 6, 7, 8, 40, 41})

		checkTrieInvariants(t, kvc.root)
	})
}

// TestExactMatchSeedBehavior verifies the holdback mechanism: when the exact
// same prompt is requested twice, the cache does not overclaim cached work.
// The last token must be re-evaluated to seed generation.
func TestExactMatchSeedBehavior(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc

		// Request A: first time.
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5}, []int32{10, 11})

		// Request B: identical prompt. Holdback means matched=4, partial in
		// the 5-token edge. For rewindable caches, switchToPath rewinds to
		// offset 4, so only the held-back token needs re-evaluation. For
		// non-rewindable caches, the rewind fails and freeAll fires.
		resB := simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5}, []int32{20, 21})
		if env.rewindable {
			if len(resB.remaining) != 1 {
				t.Fatalf("B: remaining = %d, want 1 (rewind to holdback point)", len(resB.remaining))
			}
			if resB.pendingSnapshots != 0 {
				t.Fatalf("B: pendingSnapshots = %d, want 0 (rewind succeeded)", resB.pendingSnapshots)
			}
		} else {
			if len(resB.remaining) != 5 {
				t.Fatalf("B: remaining = %d, want 5 (freeAll fallback)", len(resB.remaining))
			}
			if resB.pendingSnapshots != 1 {
				t.Fatalf("B: pendingSnapshots = %d, want 1", resB.pendingSnapshots)
			}
		}
		env.assertAllTokens(t, "after B", []int32{1, 2, 3, 4, 5, 20, 21})

		checkTrieInvariants(t, kvc.root)
	})
}

// TestConversationResumption tests the most common pattern: user sends a message,
// gets a response, then sends a follow-up. The follow-up should reuse the cached
// prefix (system prompt + first turn + assistant response).
func TestConversationResumption(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc

		// Turn 1: system prompt + user message, assistant generates response.
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5}, []int32{10, 11, 12})
		env.assertAllTokens(t, "turn 1", []int32{1, 2, 3, 4, 5, 10, 11, 12})

		// Turn 2: full history + new user message. Should get a cache hit on
		// the prefix [1,2,3,4,5,10,11,12] and only need to evaluate [20,21].
		resB := simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 10, 11, 12, 20, 21}, []int32{30})
		if len(resB.remaining) > 5 {
			t.Fatalf("turn 2: remaining = %d, want <= 5 (should reuse most of history)", len(resB.remaining))
		}
		env.assertAllTokens(t, "turn 2", []int32{1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30})

		// Turn 3: even longer history.
		resC := simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 40, 41}, nil)
		if len(resC.remaining) > 5 {
			t.Fatalf("turn 3: remaining = %d, want <= 5", len(resC.remaining))
		}
		env.assertAllTokens(t, "turn 3", []int32{1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 40, 41})

		checkTrieInvariants(t, kvc.root)
	})
}

// TestEvictionPreservesActiveConversations creates multiple conversations sharing
// a system prompt, triggers eviction via large snapshot sizes, and verifies the
// active path and shared prefix survive while memory stays bounded.
func TestEvictionPreservesActiveConversations(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc
		systemPrompt := []int32{1, 2, 3, 4, 5}

		// Create 5 conversations with unique suffixes.
		for i := range 5 {
			suffix := []int32{int32(100 + i*10), int32(101 + i*10), int32(102 + i*10)}
			inputs := append(slices.Clone(systemPrompt), suffix...)
			simulateRequest(t, kvc, inputs, []int32{int32(200 + i)})
		}

		// Inflate snapshot sizes to trigger eviction.
		walkNodes(kvc.root, func(n *trieNode) bool {
			if !n.hasSnapshots() {
				return true
			}
			snaps := make([]cache.Snapshot, len(n.snapshots))
			for i, s := range n.snapshots {
				if s != nil {
					snaps[i] = &fakeSnapshot{byteSize: 2 * 1024 * 1024 * 1024} // 2 GiB per snapshot
				}
			}
			n.setSnapshots(snaps, &kvc.pagedOutBytes)
			return true
		})

		// Run eviction.
		kvc.enforceEvictionPolicy()

		// Memory should be within limits.
		if kvc.pagedOutBytes > maxPagedOutBytes {
			t.Fatalf("pagedOutBytes = %d, want <= %d", kvc.pagedOutBytes, maxPagedOutBytes)
		}

		// Active path should be untouched.
		if len(kvc.activePath) < 2 {
			t.Fatalf("activePath should have >= 2 nodes, got %d", len(kvc.activePath))
		}

		// System prompt prefix should still be findable (multi-child
		// branch points are protected from eviction entirely).
		_, matched := findBestMatch(kvc.root, systemPrompt)
		if matched < len(systemPrompt) {
			t.Fatalf("system prompt match = %d, want %d", matched, len(systemPrompt))
		}

		checkTrieInvariants(t, kvc.root)
	})
}

// TestUserSnapshotPreservesRestorePoint verifies that user-created snapshots
// (snapshot(true)) resist structural changes that would destroy them:
//   - A user node forces new tokens into a child instead of extending in-place
//   - The snapshot remains restorable after other branches are added
func TestUserSnapshotPreservesRestorePoint(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc

		// Request A: user snapshot at offset 5, then generate.
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5}, []int32{10, 11}, 5)

		assertUserNodeExists(t, kvc, "after A")

		// Request B: extends A's prefix. The user node at offset 5 should
		// force tokens into a child rather than extending in-place.
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 10, 11, 20, 21}, nil)
		env.assertAllTokens(t, "after B", []int32{1, 2, 3, 4, 5, 10, 11, 20, 21})
		assertUserNodeExists(t, kvc, "after B")

		// Request C: diverge from the user node.
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 30, 31}, []int32{40})

		// Request D: switch back to A's branch — user snapshot still restorable.
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 10, 11, 20, 21, 50}, nil)
		env.assertAllTokens(t, "back to A", []int32{1, 2, 3, 4, 5, 10, 11, 20, 21, 50})

		checkTrieInvariants(t, kvc.root)
	})
}

// TestUserSnapshotResistsAutoMerge verifies that when a sibling leaf is evicted,
// a user-marked parent node is not auto-merged with its remaining single child.
func TestUserSnapshotResistsAutoMerge(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc

		// Request A: user snapshot at offset 3, then continue to offset 5.
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5}, []int32{10}, 3)

		// Request B: diverges at the user node, creating a second child.
		simulateRequest(t, kvc, []int32{1, 2, 3, 6, 7}, []int32{20})

		userNode := findUserNode(t, kvc)
		if len(userNode.children) != 2 {
			t.Fatalf("user node children = %d, want 2", len(userNode.children))
		}

		// Inflate snapshot sizes and evict. The non-active branch should be
		// evicted, leaving the user node with one child.
		walkNodes(kvc.root, func(n *trieNode) bool {
			if !n.hasSnapshots() {
				return true
			}
			snaps := make([]cache.Snapshot, len(n.snapshots))
			for i, s := range n.snapshots {
				if s != nil {
					snaps[i] = &fakeSnapshot{byteSize: 5 * 1024 * 1024 * 1024}
				}
			}
			n.setSnapshots(snaps, &kvc.pagedOutBytes)
			return true
		})
		kvc.enforceEvictionPolicy()

		// The user node should still exist (not auto-merged) even with one child.
		assertUserNodeExists(t, kvc, "after eviction")

		checkTrieInvariants(t, kvc.root)
	})
}

func findUserNode(t *testing.T, kvc *kvCache) *trieNode {
	t.Helper()
	var found *trieNode
	walkNodes(kvc.root, func(n *trieNode) bool {
		if n.user {
			found = n
		}
		return true
	})
	if found == nil {
		t.Fatal("no user-marked node found")
	}
	return found
}

func assertUserNodeExists(t *testing.T, kvc *kvCache, label string) {
	t.Helper()
	var exists bool
	walkNodes(kvc.root, func(n *trieNode) bool {
		if n.user {
			exists = true
		}
		return true
	})
	if !exists {
		t.Fatalf("%s: no user-marked node found", label)
	}
}

// TestBranchSwitchRestoresCorrectState exercises switching back to an older
// branch after working on a different one, verifying that the restored cache
// state contains the correct token sequence for both rewindable and
// non-rewindable caches.
func TestBranchSwitchRestoresCorrectState(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc

		// Request A: [1,2,3,4,5] + generate [10,11]
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5}, []int32{10, 11})
		env.assertAllTokens(t, "after A", []int32{1, 2, 3, 4, 5, 10, 11})

		// Request B: [1,2,3,6,7] — diverges at token 4
		simulateRequest(t, kvc, []int32{1, 2, 3, 6, 7}, []int32{12, 13})
		env.assertAllTokens(t, "after B", []int32{1, 2, 3, 6, 7, 12, 13})

		// Request C: switch back to A's branch [1,2,3,4,5,10,11,20]
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5, 10, 11, 20}, nil)
		env.assertAllTokens(t, "after C (back to A)", []int32{1, 2, 3, 4, 5, 10, 11, 20})

		checkTrieInvariants(t, kvc.root)
	})
}

// TestLRUOnlyUpdatesUsedNodes verifies that intermediate nodes on the active
// path whose snapshots were not actually restored don't get their lastUsed
// refreshed, allowing them to age out and collapse.
func TestLRUOnlyUpdatesUsedNodes(t *testing.T) {
	forEachEnv(t, func(t *testing.T, env *testEnv) {
		kvc := env.kvc

		// Request A: creates path [1,2,3,4,5] + generate [10,11]
		simulateRequest(t, kvc, []int32{1, 2, 3, 4, 5}, []int32{10, 11})

		// Request B: diverges at token 4, creating a branch point at offset 3
		// with a split snapshot.
		simulateRequest(t, kvc, []int32{1, 2, 3, 6, 7}, []int32{20, 21})

		// Set all lastUsed to a known old time.
		oldTime := time.Now().Add(-1 * time.Hour)
		walkNodes(kvc.root, func(n *trieNode) bool {
			n.lastUsed = oldTime
			return true
		})

		// Request C: continue on B's branch. This will match B's path
		// and extend it. The branch point's snapshot may be paged in
		// for some cache types but not others.
		beforeRequest := time.Now()
		simulateRequest(t, kvc, []int32{1, 2, 3, 6, 7, 20, 21, 30}, nil)

		// The path must have enough depth to exercise intermediate nodes.
		if len(kvc.activePath) < 3 {
			t.Fatalf("activePath too short to test intermediate nodes: got %d nodes", len(kvc.activePath))
		}

		// The frontier (deepest node on the active path) must be updated.
		frontier := kvc.activePath[len(kvc.activePath)-1]
		if frontier.lastUsed.Before(beforeRequest) {
			t.Errorf("frontier lastUsed was not updated: got %v, want >= %v",
				frontier.lastUsed, beforeRequest)
		}

		// Every non-frontier node on the active path (including root)
		// should retain its old lastUsed — only the frontier gets refreshed.
		for i, node := range kvc.activePath[:len(kvc.activePath)-1] {
			if !node.lastUsed.Before(beforeRequest) {
				t.Errorf("activePath[%d] (endOffset=%d) lastUsed was refreshed: got %v, want < %v",
					i, node.endOffset, node.lastUsed, beforeRequest)
			}
		}

		checkTrieInvariants(t, kvc.root)
	})
}
