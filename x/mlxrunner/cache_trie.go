package mlxrunner

import (
	"fmt"
	"slices"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

// trieNode represents a node in the compressed prefix trie for KV cache branching.
// Each node stores a compressed edge (multiple tokens) and optional paged-out
// snapshot data per cache layer.
type trieNode struct {
	tokens    []int32 // compressed edge — multiple tokens per node
	endOffset int     // cumulative tokens from root to end of this node
	parent    *trieNode
	children  []*trieNode
	lastUsed  time.Time        // for LRU eviction
	snapshots []cache.Snapshot // per-layer paged-out snapshot data (nil if not paged out)
	user      bool             // true = explicit restore point (resist auto-merge)
}

// startOffset returns the cumulative token offset at the start of this node's edge.
func (n *trieNode) startOffset() int {
	return n.endOffset - len(n.tokens)
}

// snapshotBytes returns the total bytes of paged-out snapshots on this node.
func (n *trieNode) snapshotBytes() int64 {
	var total int64
	for _, s := range n.snapshots {
		if s != nil {
			total += int64(s.Size())
		}
	}
	return total
}

// setSnapshots replaces this node's snapshots with snaps and closes the old ones.
// If counter is non-nil, the net byte delta is applied to it.
func (n *trieNode) setSnapshots(snaps []cache.Snapshot, counter *int64) {
	old := n.swapSnapshots(snaps, counter)
	for _, s := range old {
		if s != nil {
			s.Close()
		}
	}
}

// swapSnapshots is like setSnapshots but returns the previous snapshots
// without closing them. Use this when the old snapshots will be consumed
// (e.g. by Split/Merge).
func (n *trieNode) swapSnapshots(snaps []cache.Snapshot, counter *int64) []cache.Snapshot {
	old := n.snapshots
	if counter != nil {
		*counter -= n.snapshotBytes()
	}
	n.snapshots = snaps
	if counter != nil {
		*counter += n.snapshotBytes()
	}
	return old
}

// hasSnapshots returns true if any layer has snapshot data.
func (n *trieNode) hasSnapshots() bool {
	return slices.ContainsFunc(n.snapshots, func(s cache.Snapshot) bool { return s != nil })
}

// hasAllSnapshots returns true if every layer has snapshot data.
func (n *trieNode) hasAllSnapshots() bool {
	return len(n.snapshots) > 0 && !slices.Contains(n.snapshots, nil)
}

// findBestMatch walks the trie matching input tokens, returning the path of
// nodes traversed and the total number of tokens matched.
func findBestMatch(root *trieNode, tokens []int32) (path []*trieNode, matched int) {
	if root == nil {
		return nil, 0
	}

	path = []*trieNode{root}
	pos := 0

	node := root
	for pos < len(tokens) {
		// When multiple children share the same first token (e.g. after
		// a split), prefer the child whose full edge matches over one
		// that only partially matches. This is just being defensive - it
		// shouldn't actually happen.
		var best *trieNode
		bestMatched := 0
		bestFull := false
		for _, child := range node.children {
			edge := child.tokens
			if len(edge) == 0 {
				continue
			}
			if edge[0] != tokens[pos] {
				continue
			}
			// Count matching tokens in this child's edge.
			j := 0
			for j < len(edge) && pos+j < len(tokens) && edge[j] == tokens[pos+j] {
				j++
			}
			full := j == len(edge)
			// Prefer full edge matches; among same type, prefer longer.
			if best == nil || (full && !bestFull) || (full == bestFull && j > bestMatched) {
				best = child
				bestMatched = j
				bestFull = full
			}
		}
		if best == nil {
			break
		}

		pos += bestMatched
		path = append(path, best)

		if !bestFull {
			// Partial match within this edge
			break
		}
		node = best
	}

	return path, pos
}

// appendTokens either creates a new child node or extends the leaf in place,
// returning the node that now holds the tokens.
func (n *trieNode) appendTokens(root *trieNode, tokens []int32, endOffset int) *trieNode {
	if n == root || len(n.children) > 0 || n.hasSnapshots() {
		child := &trieNode{
			tokens:    make([]int32, len(tokens)),
			endOffset: endOffset,
			parent:    n,
			lastUsed:  n.lastUsed,
		}
		copy(child.tokens, tokens)
		n.children = append(n.children, child)
		return child
	}
	n.tokens = append(n.tokens, tokens...)
	n.endOffset = endOffset
	return n
}

// removeNode removes a leaf node from the trie.
func removeNode(node *trieNode, counter *int64) {
	if node.parent == nil {
		panic("removeNode called on root")
	}
	if len(node.children) != 0 {
		panic("removeNode called on non-leaf node")
	}
	p := node.parent
	for i, child := range p.children {
		if child == node {
			p.children = append(p.children[:i], p.children[i+1:]...)
			break
		}
	}
	node.parent = nil
	node.setSnapshots(nil, counter)
}

// splitNode splits a node at the given token offset within its edge,
// creating a new parent node. Returns the new parent.
// `at` is relative to the node's edge (0-based index into node.tokens).
// If caches are provided, snapshots are split between parent and child
// using Cache.Split; otherwise snapshots are invalidated.
func splitNode(node *trieNode, at int, caches []cache.Cache, counter *int64) *trieNode {
	if at <= 0 || at >= len(node.tokens) {
		panic(fmt.Sprintf("splitNode: invalid split offset %d for node with %d tokens", at, len(node.tokens)))
	}

	// Create new parent with the prefix of the edge.
	newParent := &trieNode{
		tokens:    make([]int32, at),
		endOffset: node.startOffset() + at,
		parent:    node.parent,
		children:  []*trieNode{node},
		lastUsed:  node.lastUsed,
	}
	copy(newParent.tokens, node.tokens[:at])

	// Update the original node to have only the suffix.
	node.tokens = node.tokens[at:]
	// endOffset stays the same for the original node.

	// Split snapshots between parent and child using Cache.Split.
	// Split consumes the old snapshots, so we remove them first (adjusting
	// the counter), then assign the split halves (adjusting it back).
	if node.hasSnapshots() {
		oldSnaps := node.swapSnapshots(nil, counter)
		parentSnaps := make([]cache.Snapshot, len(oldSnaps))
		childSnaps := make([]cache.Snapshot, len(oldSnaps))
		for i, snap := range oldSnaps {
			if snap != nil {
				parentSnaps[i], childSnaps[i] = caches[i].Split(snap, newParent.endOffset)
			}
		}
		newParent.setSnapshots(parentSnaps, counter)
		node.setSnapshots(childSnaps, counter)
	}

	// Reparent: replace node with newParent in the old parent's children.
	if node.parent != nil {
		for i, child := range node.parent.children {
			if child == node {
				node.parent.children[i] = newParent
				break
			}
		}
	}
	node.parent = newParent

	return newParent
}

// mergeWithChild merges a node with its single child: concatenates tokens,
// merges snapshot data via Cache.Merge, and removes the child.
func mergeWithChild(node *trieNode, caches []cache.Cache, counter *int64) {
	if len(node.children) != 1 {
		panic(fmt.Sprintf("mergeWithChild called on node with %d children", len(node.children)))
	}

	child := node.children[0]

	// Concatenate tokens.
	node.tokens = append(node.tokens, child.tokens...)
	node.endOffset = child.endOffset

	// Merge snapshots per layer. Merge consumes the old snapshots, so we
	// remove them first (adjusting the counter), then assign the merged
	// result (adjusting it back).
	if len(node.snapshots) > 0 || len(child.snapshots) > 0 {
		nodeSnaps := node.swapSnapshots(nil, counter)
		childSnaps := child.swapSnapshots(nil, counter)
		merged := make([]cache.Snapshot, len(caches))
		for i := range caches {
			var ps, cs cache.Snapshot
			if nodeSnaps != nil {
				ps = nodeSnaps[i]
			}
			if childSnaps != nil {
				cs = childSnaps[i]
			}

			merged[i] = caches[i].Merge(ps, cs)
		}
		node.setSnapshots(merged, counter)
	}

	// Adopt grandchildren.
	node.children = child.children
	for _, gc := range node.children {
		gc.parent = node
	}

	// Inherit user flag from child if child was a user-created snapshot node.
	node.user = child.user

	// Update lastUsed to the more recent of the two.
	if child.lastUsed.After(node.lastUsed) {
		node.lastUsed = child.lastUsed
	}

	child.parent = nil
	child.children = nil
}

// walkNodes calls fn for every node in the trie (depth-first).
// If fn returns false, the walk stops.
func walkNodes(root *trieNode, fn func(*trieNode) bool) {
	if root == nil {
		return
	}
	var walk func(*trieNode) bool
	walk = func(n *trieNode) bool {
		if !fn(n) {
			return false
		}
		for _, child := range n.children {
			if !walk(child) {
				return false
			}
		}
		return true
	}
	walk(root)
}
