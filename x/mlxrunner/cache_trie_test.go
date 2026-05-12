package mlxrunner

import (
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

func newTestTrie(tokens []int32) *trieNode {
	root := &trieNode{lastUsed: time.Now()}
	if len(tokens) > 0 {
		child := &trieNode{
			tokens:    slices.Clone(tokens),
			endOffset: len(tokens),
			parent:    root,
			lastUsed:  time.Now(),
		}
		root.children = []*trieNode{child}
	}
	return root
}

func TestFindBestMatchMultipleBranches(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}

	branch1 := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	branch2 := &trieNode{
		tokens:    []int32{4, 5, 6},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{branch1, branch2}

	// Match branch 1.
	path, matched := findBestMatch(root, []int32{1, 2, 3, 7})
	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}
	if len(path) != 2 || path[1] != branch1 {
		t.Fatal("expected to match branch1")
	}

	// Match branch 2.
	path, matched = findBestMatch(root, []int32{4, 5, 6, 8})
	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}
	if len(path) != 2 || path[1] != branch2 {
		t.Fatal("expected to match branch2")
	}

	// Match neither.
	_, matched = findBestMatch(root, []int32{7, 8, 9})
	if matched != 0 {
		t.Fatalf("expected 0 matched, got %d", matched)
	}
}

func TestFindBestMatchPrefersFullEdge(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}

	shared := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{shared}

	longer := &trieNode{
		tokens:    []int32{10, 11, 12, 13, 14},
		endOffset: 8,
		parent:    shared,
		lastUsed:  time.Now(),
	}
	shorter := &trieNode{
		tokens:    []int32{10, 11, 12},
		endOffset: 6,
		parent:    shared,
		lastUsed:  time.Now(),
	}
	// Put longer first so naive first-match would pick it.
	shared.children = []*trieNode{longer, shorter}

	input := []int32{1, 2, 3, 10, 11, 12, 99, 100}
	path, matched := findBestMatch(root, input)

	if matched != 6 {
		t.Fatalf("expected 6 matched, got %d", matched)
	}
	if len(path) != 3 {
		t.Fatalf("expected 3 nodes in path, got %d", len(path))
	}
	if path[2] != shorter {
		t.Fatal("expected findBestMatch to pick shorter (full edge match), not longer (partial)")
	}
}

func TestFindBestMatchPrefersLongerPartial(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}

	child1 := &trieNode{
		tokens:    []int32{1, 2, 3, 4, 5},
		endOffset: 5,
		parent:    root,
		lastUsed:  time.Now(),
	}
	child2 := &trieNode{
		tokens:    []int32{1, 2, 9},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{child2, child1}

	input := []int32{1, 2, 3, 7, 8}
	path, matched := findBestMatch(root, input)

	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}
	if path[1] != child1 {
		t.Fatal("expected findBestMatch to pick child1 (longer partial match)")
	}
}

func TestSplitNodeWithSnapshots(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3, 4, 5})
	child := root.children[0]

	rc := &fakeRewindableCache{tracker: &snapshotTracker{}, tokens: []int32{1, 2, 3, 4, 5}}
	child.snapshots = []cache.Snapshot{rc.Snapshot(0)}
	child.user = true

	caches := []cache.Cache{rc}

	newParent := splitNode(child, 3, caches, nil)

	if !newParent.hasSnapshots() {
		t.Fatal("newParent should have snapshots after split")
	}
	if newParent.user {
		t.Fatal("newParent should not be a user snapshot after splitNode")
	}
	if !child.hasSnapshots() {
		t.Fatal("child should have snapshots after split")
	}
	if !child.user {
		t.Fatal("child should remain a user snapshot")
	}
}

func TestFindSplitAppendSequence(t *testing.T) {
	root := newTestTrie([]int32{1, 2, 3, 4, 5})

	path, matched := findBestMatch(root, []int32{1, 2, 3, 6, 7})
	if matched != 3 {
		t.Fatalf("expected 3 matched, got %d", matched)
	}

	lastNode := path[len(path)-1]
	matchedInEdge := matched - lastNode.startOffset()
	split := splitNode(lastNode, matchedInEdge, nil, nil)

	split.appendTokens(root, []int32{6, 7}, 5)

	if len(root.children) != 1 {
		t.Fatalf("root should have 1 child, got %d", len(root.children))
	}
	shared := root.children[0]
	if !slices.Equal(shared.tokens, []int32{1, 2, 3}) {
		t.Fatalf("shared tokens = %v, want [1,2,3]", shared.tokens)
	}
	if len(shared.children) != 2 {
		t.Fatalf("shared should have 2 children, got %d", len(shared.children))
	}

	_, m1 := findBestMatch(root, []int32{1, 2, 3, 4, 5})
	if m1 != 5 {
		t.Fatalf("original branch: expected 5 matched, got %d", m1)
	}
	_, m2 := findBestMatch(root, []int32{1, 2, 3, 6, 7})
	if m2 != 5 {
		t.Fatalf("new branch: expected 5 matched, got %d", m2)
	}
	_, m3 := findBestMatch(root, []int32{1, 2, 3, 9, 9})
	if m3 != 3 {
		t.Fatalf("unrelated input: expected 3 matched, got %d", m3)
	}
}

func TestRepeatedBranching(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}

	root.appendTokens(root, []int32{1, 2, 3, 4, 5}, 5)

	_, matchedB := findBestMatch(root, []int32{1, 2, 3, 6, 7})
	if matchedB != 3 {
		t.Fatalf("B: expected 3 matched, got %d", matchedB)
	}
	nodeA := root.children[0]
	split1 := splitNode(nodeA, 3, nil, nil)
	split1.appendTokens(root, []int32{6, 7}, 5)

	_, matchedC := findBestMatch(root, []int32{1, 2, 8, 9})
	if matchedC != 2 {
		t.Fatalf("C: expected 2 matched, got %d", matchedC)
	}
	split2 := splitNode(split1, 2, nil, nil)
	split2.appendTokens(root, []int32{8, 9}, 4)

	_, mA := findBestMatch(root, []int32{1, 2, 3, 4, 5})
	if mA != 5 {
		t.Fatalf("A: expected 5 matched, got %d", mA)
	}
	_, mB := findBestMatch(root, []int32{1, 2, 3, 6, 7})
	if mB != 5 {
		t.Fatalf("B: expected 5 matched, got %d", mB)
	}
	_, mC := findBestMatch(root, []int32{1, 2, 8, 9})
	if mC != 4 {
		t.Fatalf("C: expected 4 matched, got %d", mC)
	}

	checkTrieInvariants(t, root)
}

func TestMergeWithChild(t *testing.T) {
	t.Run("Basic", func(t *testing.T) {
		// root -> A[1,2,3] -> B[4,5] -> {C[6], D[7]}
		now := time.Now()
		root := &trieNode{lastUsed: now}
		a := &trieNode{
			tokens:    []int32{1, 2, 3},
			endOffset: 3,
			parent:    root,
			lastUsed:  now,
			snapshots: []cache.Snapshot{&fakeSnapshot{tokens: []int32{1, 2, 3}, from: 0, to: 3}},
		}
		b := &trieNode{
			tokens:    []int32{4, 5},
			endOffset: 5,
			parent:    a,
			lastUsed:  now,
			snapshots: []cache.Snapshot{&fakeSnapshot{tokens: []int32{4, 5}, from: 3, to: 5}},
		}
		c := &trieNode{tokens: []int32{6}, endOffset: 6, parent: b, lastUsed: now}
		d := &trieNode{tokens: []int32{7}, endOffset: 6, parent: b, lastUsed: now}
		root.children = []*trieNode{a}
		a.children = []*trieNode{b}
		b.children = []*trieNode{c, d}

		mc := &fakeRewindableCache{tracker: &snapshotTracker{}, tokens: []int32{1, 2, 3, 4, 5}}
		mergeWithChild(a, []cache.Cache{mc}, nil)

		// Tokens concatenated.
		if !slices.Equal(a.tokens, []int32{1, 2, 3, 4, 5}) {
			t.Fatalf("merged tokens = %v, want [1,2,3,4,5]", a.tokens)
		}
		if a.endOffset != 5 {
			t.Fatalf("merged endOffset = %d, want 5", a.endOffset)
		}
		// Grandchildren reparented.
		if len(a.children) != 2 {
			t.Fatalf("merged children count = %d, want 2", len(a.children))
		}
		if c.parent != a || d.parent != a {
			t.Fatal("grandchildren should be reparented to merged node")
		}
		// B detached.
		if b.parent != nil || b.children != nil || b.snapshots != nil {
			t.Fatal("child B should be fully detached after merge")
		}
		// Merged snapshot should cover [0,5).
		if !a.hasSnapshots() {
			t.Fatal("merged node should have snapshots")
		}
		ms := a.snapshots[0].(*fakeSnapshot)
		if ms.from != 0 || ms.to != 5 {
			t.Fatalf("merged snapshot = [%d,%d), want [0,5)", ms.from, ms.to)
		}

		checkTrieInvariants(t, root)
	})

	t.Run("UserFlag", func(t *testing.T) {
		root := &trieNode{lastUsed: time.Now()}
		parent := &trieNode{
			tokens: []int32{1, 2}, endOffset: 2, parent: root,
			lastUsed: time.Now(), user: false,
		}
		child := &trieNode{
			tokens: []int32{3, 4}, endOffset: 4, parent: parent,
			lastUsed: time.Now(), user: true,
		}
		root.children = []*trieNode{parent}
		parent.children = []*trieNode{child}

		mergeWithChild(parent, nil, nil)

		if !parent.user {
			t.Fatal("merged node should inherit user=true from child")
		}
	})

	t.Run("LastUsed", func(t *testing.T) {
		now := time.Now()
		root := &trieNode{lastUsed: now}
		parent := &trieNode{
			tokens: []int32{1}, endOffset: 1, parent: root,
			lastUsed: now.Add(-1 * time.Hour),
		}
		child := &trieNode{
			tokens: []int32{2}, endOffset: 2, parent: parent,
			lastUsed: now.Add(1 * time.Hour),
		}
		root.children = []*trieNode{parent}
		parent.children = []*trieNode{child}

		mergeWithChild(parent, nil, nil)

		if !parent.lastUsed.Equal(now.Add(1 * time.Hour)) {
			t.Fatal("merged node should pick the more recent lastUsed")
		}
	})

	t.Run("PanicOnMultipleChildren", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic on node with 2 children")
			}
		}()
		root := &trieNode{lastUsed: time.Now()}
		node := &trieNode{
			tokens: []int32{1}, endOffset: 1, parent: root, lastUsed: time.Now(),
			children: []*trieNode{
				{tokens: []int32{2}, endOffset: 2, lastUsed: time.Now()},
				{tokens: []int32{3}, endOffset: 2, lastUsed: time.Now()},
			},
		}
		root.children = []*trieNode{node}
		mergeWithChild(node, nil, nil)
	})
}

func TestSplitMergeRoundTrip(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}
	leaf := &trieNode{
		tokens:    []int32{1, 2, 3, 4, 5},
		endOffset: 5,
		parent:    root,
		lastUsed:  time.Now(),
		snapshots: []cache.Snapshot{&fakeSnapshot{tokens: []int32{1, 2, 3, 4, 5}, from: 0, to: 5}},
	}
	root.children = []*trieNode{leaf}

	mc := &fakeRewindableCache{tracker: &snapshotTracker{}, tokens: []int32{1, 2, 3, 4, 5}}
	caches := []cache.Cache{mc}

	// Split at 3: [1,2,3] -> [4,5]
	newParent := splitNode(leaf, 3, caches, nil)
	if !slices.Equal(newParent.tokens, []int32{1, 2, 3}) {
		t.Fatalf("after split: parent tokens = %v, want [1,2,3]", newParent.tokens)
	}
	if !slices.Equal(leaf.tokens, []int32{4, 5}) {
		t.Fatalf("after split: child tokens = %v, want [4,5]", leaf.tokens)
	}
	checkTrieInvariants(t, root)

	// Merge back: should restore [1,2,3,4,5]
	mergeWithChild(newParent, caches, nil)
	if !slices.Equal(newParent.tokens, []int32{1, 2, 3, 4, 5}) {
		t.Fatalf("after merge: tokens = %v, want [1,2,3,4,5]", newParent.tokens)
	}
	if newParent.endOffset != 5 {
		t.Fatalf("after merge: endOffset = %d, want 5", newParent.endOffset)
	}
	if len(newParent.children) != 0 {
		t.Fatalf("after merge: children count = %d, want 0", len(newParent.children))
	}
	// Merged snapshot should cover [0,5).
	if !newParent.hasSnapshots() {
		t.Fatal("after merge: should have snapshots")
	}
	ms := newParent.snapshots[0].(*fakeSnapshot)
	if ms.from != 0 || ms.to != 5 {
		t.Fatalf("after merge: snapshot = [%d,%d), want [0,5)", ms.from, ms.to)
	}

	checkTrieInvariants(t, root)
}

func TestRemoveNode(t *testing.T) {
	t.Run("Leaf", func(t *testing.T) {
		root := &trieNode{lastUsed: time.Now()}
		shared := &trieNode{
			tokens: []int32{1, 2, 3}, endOffset: 3, parent: root, lastUsed: time.Now(),
		}
		leafA := &trieNode{
			tokens: []int32{4, 5}, endOffset: 5, parent: shared, lastUsed: time.Now(),
			snapshots: []cache.Snapshot{&fakeSnapshot{from: 3, to: 5}},
		}
		leafB := &trieNode{
			tokens: []int32{6, 7}, endOffset: 5, parent: shared, lastUsed: time.Now(),
			snapshots: []cache.Snapshot{&fakeSnapshot{from: 3, to: 5}},
		}
		root.children = []*trieNode{shared}
		shared.children = []*trieNode{leafA, leafB}

		removeNode(leafA, nil)

		if len(shared.children) != 1 {
			t.Fatalf("parent should have 1 child, got %d", len(shared.children))
		}
		if shared.children[0] != leafB {
			t.Fatal("remaining child should be leafB")
		}
		if leafA.parent != nil {
			t.Fatal("removed node parent should be nil")
		}
		if leafA.snapshots != nil {
			t.Fatal("removed node snapshots should be nil")
		}

		checkTrieInvariants(t, root)
	})

	t.Run("PanicOnRoot", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic when removing root")
			}
		}()
		removeNode(&trieNode{}, nil)
	})

	t.Run("PanicOnNonLeaf", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic when removing non-leaf")
			}
		}()
		parent := &trieNode{parent: &trieNode{}}
		parent.children = []*trieNode{{}}
		removeNode(parent, nil)
	})
}
