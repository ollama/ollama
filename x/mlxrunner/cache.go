// cache.go manages a shared KV cache across conversations using a compressed
// prefix trie. Each trie node stores a token sequence (edge) and optional
// per-layer snapshots that can be paged in/out of the live MLX cache arrays.
//
// Key properties:
//   - Only one path through the trie is "active" (backed by live MLX arrays)
//     at a time. Switching paths pages out the frontier node and pages in the
//     new path.
//   - Snapshots are only captured at the frontier (end) of the active path.
//     Intermediate node snapshots come from split prefill.
//   - All cache layers must stay at the same token offset.
//   - Sibling edges must not share a common token prefix (compressed trie
//     invariant).
//   - begin() always re-evaluates at least one token so the pipeline can seed
//     generation, even on a full prefix match.

package mlxrunner

import (
	"cmp"
	"fmt"
	"log/slog"
	"slices"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

const maxPagedOutBytes int64 = 8 << 30 // 8 GiB eviction threshold for paged-out snapshot memory

type kvCache struct {
	root          *trieNode   // root of the prefix trie
	activePath    []*trieNode // current root→leaf path with live MLX arrays
	caches        []cache.Cache
	pagedOutBytes int64 // total bytes in paged-out snapshots across the trie
}

// pendingSnapshot is a snapshot scheduled to be taken during prefill.
type pendingSnapshot struct {
	offset int
	user   bool
}

// cacheSession manages caches for a single pipeline run.
// Callers should append generated tokens to outputs and
// defer close to save the cache state.
type cacheSession struct {
	cache   *kvCache
	inputs  []int32
	outputs []int32

	caches    []cache.Cache
	remaining []int32

	// pendingSnapshots lists offsets where snapshots should be captured
	// during prefill, sorted by offset. Entries are scheduled on the caches
	// before prefill and drained or discarded after.
	pendingSnapshots []pendingSnapshot
}

func (c *kvCache) ensureCaches(m base.Model) {
	if len(c.caches) != 0 {
		return
	}
	if cacheFactory, ok := m.(interface{ NewCaches() []cache.Cache }); ok {
		c.caches = cacheFactory.NewCaches()
		return
	}
	c.caches = make([]cache.Cache, m.NumLayers())
	for i := range c.caches {
		c.caches[i] = cache.NewKVCache()
	}
}

func (c *kvCache) ensureRoot() {
	if c.root == nil {
		c.root = &trieNode{
			lastUsed: time.Now(),
		}
		c.activePath = []*trieNode{c.root}
	}
}

// begin prepares caches for a new request. It finds the nearest
// matching cache or creates new caches if none match.
func (c *kvCache) begin(m base.Model, inputs []int32) *cacheSession {
	c.ensureCaches(m)
	c.ensureRoot()

	matchPath, matched := findBestMatch(c.root, inputs)
	originalMatched := matched

	// Always keep at least one token to re-evaluate so the
	// pipeline can seed token generation from it.
	if matched == len(inputs) && matched > 0 {
		matchPath, matched = findBestMatch(c.root, inputs[:len(inputs)-1])
	}

	// Switch to the matched path, paging in/out as needed.
	c.switchToPath(matchPath, matched)

	// switchToPath aligns caches to a common offset
	prefix := c.minCacheOffset()
	remaining := inputs[prefix:]

	session := &cacheSession{
		cache:     c,
		inputs:    inputs,
		caches:    c.caches,
		remaining: remaining,
	}

	// Schedule a snapshot at the branch point during prefill so future
	// requests diverging here can restore instead of re-evaluating.
	if prefix < matched {
		session.pendingSnapshots = append(session.pendingSnapshots, pendingSnapshot{offset: matched, user: false})
	}

	msg := "cache hit"
	if prefix == 0 {
		msg = "cache miss"
	}
	slog.Info(msg, "total", len(inputs), "matched", originalMatched, "cached", prefix, "left", len(remaining))

	return session
}

// switchToPath transitions from the current active path to a new path,
// paging out diverging segments and paging in the new path.
func (c *kvCache) switchToPath(newPath []*trieNode, matched int) {
	defer c.enforceEvictionPolicy()

	// Find common ancestor index.
	commonLen := 0
	for commonLen < len(c.activePath) && commonLen < len(newPath) {
		if c.activePath[commonLen] != newPath[commonLen] {
			break
		}
		commonLen++
	}

	ancestorOffset := 0
	if commonLen > 0 {
		ancestorOffset = c.activePath[commonLen-1].endOffset
	}

	var pageOutCount, pageInCount int

	// Page out the leaf of the old path. Only the leaf's live cache
	// state is correct — intermediate nodes already have snapshots
	// captured during their creation (splitNode + prefill). Snapshotting
	// non-leaf nodes here would produce wrong results for non-rewindable
	// caches (e.g. RecurrentCache) whose state reflects the leaf, not
	// the intermediate boundary.
	leaf := len(c.activePath) - 1
	leafDiverges := leaf >= commonLen
	leafNeedsRewind := matched < c.activePath[leaf].endOffset
	if leafDiverges || leafNeedsRewind {
		node := c.activePath[leaf]
		if !node.hasAllSnapshots() {
			fromOffset := node.startOffset()
			snaps := make([]cache.Snapshot, len(c.caches))
			for j, kv := range c.caches {
				if kv == nil {
					continue
				}
				snaps[j] = kv.Snapshot(fromOffset)
			}
			node.setSnapshots(snaps, &c.pagedOutBytes)
			pageOutCount++
			logutil.Trace(fmt.Sprintf("page out: [%d, %d)", fromOffset, node.endOffset))
		}
	}

	// Rewind each cache to the target offset or free it. When matched
	// falls within the ancestor's range (same-path case), we rewind
	// directly to the match point. Otherwise we rewind to the ancestor
	// and let page-in bring us forward to matched.
	rewindTarget := min(ancestorOffset, matched)
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if !kv.Restore(nil, rewindTarget) {
			kv.Free()
		}
	}

	// Page in — walk the full new path, restoring from snapshots.
	// Freed caches naturally pick up the first available snapshot.
	// Caches already past a node skip it via offset check.
pageIn:
	for _, node := range newPath {
		if !node.hasSnapshots() {
			continue
		}
		nodeTarget := min(node.endOffset, matched)
		for j, kv := range c.caches {
			if kv == nil {
				continue
			}
			if j >= len(node.snapshots) || node.snapshots[j] == nil {
				continue
			}
			if kv.Offset() >= nodeTarget {
				continue
			}
			if !kv.Restore(node.snapshots[j], nodeTarget) {
				// Restore failed — stop page-in and let alignment
				// bring all caches to a consistent offset.
				break pageIn
			}
		}
		if node.endOffset > ancestorOffset {
			pageInCount++
			logutil.Trace(fmt.Sprintf("page in: [%d, %d)", node.startOffset(), nodeTarget))
		}
	}

	// Align all caches to the minimum offset.
	c.activePath = newPath
	minOff := c.minCacheOffset()
	for _, kv := range c.caches {
		if kv != nil && kv.Offset() != minOff {
			if !kv.Restore(nil, minOff) {
				slog.Warn("failed to restore cache, freeing all caches", "offset", minOff)
				c.freeAll()
				break
			}
		}
	}
	for i := len(c.activePath) - 1; i >= 0; i-- {
		if c.activePath[i].endOffset <= minOff {
			c.activePath = c.activePath[:i+1]
			break
		}
	}

	// Update last-used time on only the final used node. For recurrent
	// caches we don't need the intermediate snapshots and for KV caches
	// we can reslice the data out of merged edges.
	if len(c.activePath) > 0 {
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}

	if pageOutCount > 0 || pageInCount > 0 {
		slog.Debug("switching cache path", "page_out", pageOutCount, "page_in", pageInCount)
	}
}

// schedulePrefillSnapshots schedules every cache to capture snapshots as the
// forward pass crosses the given absolute token offsets, so a single full-size
// prefill records interior states without the caller breaking the batch. The
// passed offsets are user-requested restore points; they are merged with any
// snapshots begin already scheduled (e.g. a branch point), with coinciding
// offsets upgraded to user so eviction preserves them.
//
// Offsets at or before the current cache position, or past the end of the
// prompt, are dropped: callers only request offsets ahead of the prefill base,
// so this is a defensive guard.
func (s *cacheSession) schedulePrefillSnapshots(offsets []int) {
	c := s.cache
	base := c.minCacheOffset()
	for _, offset := range offsets {
		if offset <= base || offset > len(s.inputs) {
			continue
		}
		// Deduplicate: if this offset already exists, upgrade to user.
		found := false
		for i := range s.pendingSnapshots {
			if s.pendingSnapshots[i].offset == offset {
				s.pendingSnapshots[i].user = true
				found = true
				break
			}
		}
		if !found {
			s.pendingSnapshots = append(s.pendingSnapshots, pendingSnapshot{offset: offset, user: true})
		}
	}
	slices.SortFunc(s.pendingSnapshots, func(a, b pendingSnapshot) int {
		return a.offset - b.offset
	})

	if len(s.pendingSnapshots) == 0 {
		return
	}

	prepared := make([]int, len(s.pendingSnapshots))
	for i, p := range s.pendingSnapshots {
		prepared[i] = p.offset
	}
	for _, kv := range c.caches {
		if kv != nil {
			kv.PrepareSnapshots(prepared)
		}
	}
}

// discardPrefillSnapshots drains and closes the snapshots scheduled by
// schedulePrefillSnapshots without attaching them to the trie, releasing their
// pinned/lazy state. It is a no-op once attachPrefillSnapshots has drained the
// schedule, so close can call it unconditionally to clean up an abandoned
// prefill.
func (s *cacheSession) discardPrefillSnapshots() {
	if len(s.pendingSnapshots) == 0 {
		return
	}
	s.pendingSnapshots = nil

	for _, kv := range s.cache.caches {
		if kv == nil {
			continue
		}
		for _, snap := range kv.TakeSnapshots() {
			if snap != nil {
				snap.Close()
			}
		}
	}
}

// attachPrefillSnapshots collects the snapshots captured during prefill and
// attaches them to the trie, materializing a node at each requested offset.
// Pending offsets are ascending and were scheduled in the same order, so the
// snapshots each cache returns line up with them. The trie frontier is
// advanced to each offset in turn, so its node edges [prev, offset) match the
// edge-local ranges the caches captured.
func (s *cacheSession) attachPrefillSnapshots() {
	if len(s.pendingSnapshots) == 0 {
		return
	}

	c := s.cache
	pending := s.pendingSnapshots
	s.pendingSnapshots = nil

	// Drain each cache's captures (one per pending offset, in order) into
	// per-offset rows.
	rows := make([][]cache.Snapshot, len(pending))
	for i := range rows {
		rows[i] = make([]cache.Snapshot, len(c.caches))
	}
	for j, kv := range c.caches {
		if kv == nil {
			continue
		}
		taken := kv.TakeSnapshots()
		for i := range pending {
			if i < len(taken) {
				rows[i][j] = taken[i]
			}
		}
	}

	// Prefill leaves one token unprocessed for decode seeding, so an offset
	// at or past the live cache position was never crossed by a write and has
	// no captured state. Skip it rather than materialize a node whose edge
	// claims tokens the cache never wrote. Closing its (nil) row is a no-op.
	reached := c.minCacheOffset()
	stored := append(s.inputs, s.outputs...)
	for i, p := range pending {
		if p.offset > reached {
			// Never crossed by a write, so the row is nil; close any entry
			// defensively in case a cache captured one anyway.
			for _, snap := range rows[i] {
				if snap != nil {
					snap.Close()
				}
			}
			continue
		}
		frontier := c.activePath[len(c.activePath)-1]
		if frontier.endOffset < p.offset {
			edgeTokens := stored[frontier.endOffset:p.offset]
			frontier = c.advancePath(frontier, edgeTokens, p.offset)
		}
		if p.user {
			frontier.user = true
		}
		s.attachCapturedSnapshots(frontier, rows[i])
	}
}

// attachCapturedSnapshots stores pre-captured snapshots on a trie node. Unlike
// taking a fresh Snapshot from the live cache, this works for an interior node
// whose offset the live cache has already advanced past: the snapshots come
// from the capture scheduled earlier, not from the cache's current state. The
// node takes ownership of the snapshots (TakeSnapshots already transferred it).
func (s *cacheSession) attachCapturedSnapshots(node *trieNode, snaps []cache.Snapshot) {
	c := s.cache
	node.setSnapshots(snaps, &c.pagedOutBytes)
	node.lastUsed = time.Now()
	slog.Debug("created snapshot", "offset", node.endOffset)
	c.enforceEvictionPolicy()
}

// advancePath advances the active path from the current frontier by matching
// tokens against existing trie children, splitting partial matches, and
// appending any remaining tokens as new nodes. Returns the new frontier.
func (c *kvCache) advancePath(frontier *trieNode, tokens []int32, endOffset int) *trieNode {
	// Check if existing children already cover some or all of tokens.
	// tokens may span multiple trie nodes when extending a previous run's
	// leaf and this snapshot now overlaps that same range.
	matchPath, matched := findBestMatch(frontier, tokens)
	// matchPath[0] is frontier itself; the rest are newly traversed nodes.
	remaining := tokens[matched:]

	// Check for a partial match within the last node's edge — if so, split it.
	if len(matchPath) > 1 {
		lastNode := matchPath[len(matchPath)-1]
		matchedInEdge := frontier.endOffset + matched - lastNode.startOffset()
		if matchedInEdge > 0 && matchedInEdge < len(lastNode.tokens) {
			matchPath[len(matchPath)-1] = splitNode(lastNode, matchedInEdge, c.caches, &c.pagedOutBytes)
		}
	}

	// Append traversed nodes (excluding frontier) to the active path.
	c.activePath = append(c.activePath, matchPath[1:]...)
	dest := matchPath[len(matchPath)-1]

	if len(remaining) > 0 {
		// Drop non-user snapshots so appendTokens can extend in-place
		// rather than creating a new child node.
		if len(dest.children) == 0 && !dest.user {
			dest.setSnapshots(nil, &c.pagedOutBytes)
		}
		newDest := dest.appendTokens(c.root, remaining, endOffset)
		if newDest != dest {
			c.activePath = append(c.activePath, newDest)
		}
		dest = newDest
	}
	return dest
}

// freeAll releases all cache layers.
func (c *kvCache) freeAll() {
	for _, kv := range c.caches {
		if kv != nil {
			kv.Free()
		}
	}
}

func (c *kvCache) minCacheOffset() int {
	offset := 0
	found := false
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if off := kv.Offset(); !found || off < offset {
			offset = off
			found = true
		}
	}
	return offset
}

// close saves the token state if the forward pass ran.
func (s *cacheSession) close() {
	// Release any prefill snapshots the session scheduled but never attached to
	// the trie. A successful prefill drains them in attachPrefillSnapshots (so
	// this is a no-op then); an abandoned one (e.g. cancellation between
	// schedule and attach) leaves them in the caches, where the next request's
	// PrepareSnapshots would overwrite the schedule without closing them,
	// leaking the pinned/lazy snapshots and their VRAM.
	s.discardPrefillSnapshots()

	offset := s.cache.minCacheOffset()
	if offset <= 0 {
		return
	}

	arrays := make([]*mlx.Array, 0, 2*len(s.caches))
	for _, kv := range s.caches {
		if kv == nil {
			continue
		}
		arrays = append(arrays, kv.State()...)
	}

	// Ensure that if we have run the forward pass and set the metadata
	// that we also actually have the data.
	mlx.AsyncEval(arrays...)

	// Advance the trie frontier with any newly generated tokens.
	c := s.cache
	if len(c.activePath) > 0 {
		frontier := c.activePath[len(c.activePath)-1]
		stored := append(s.inputs, s.outputs...)

		if offset > frontier.endOffset {
			newTokens := stored[frontier.endOffset:offset]
			c.advancePath(frontier, newTokens, offset)
		}
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}
}

// enforceEvictionPolicy evicts eligible nodes until paged-out memory is within limits.
func (c *kvCache) enforceEvictionPolicy() {
	if c.pagedOutBytes <= maxPagedOutBytes {
		return
	}

	activeSet := make(map[*trieNode]bool, len(c.activePath))
	for _, n := range c.activePath {
		activeSet[n] = true
	}

	for c.pagedOutBytes > maxPagedOutBytes {
		var best *trieNode
		walkNodes(c.root, func(n *trieNode) bool {
			if n == c.root || activeSet[n] || len(n.children) > 1 {
				return true
			}
			// Evict: oldest, then deepest, then largest.
			if best == nil || cmp.Or(
				n.lastUsed.Compare(best.lastUsed),
				cmp.Compare(best.endOffset, n.endOffset),
				cmp.Compare(best.snapshotBytes(), n.snapshotBytes()),
			) < 0 {
				best = n
			}
			return true
		})
		if best == nil {
			break
		}
		c.evictNode(best)
	}
}

// evictNode evicts a single node from the trie, freeing its snapshot memory.
func (c *kvCache) evictNode(node *trieNode) {
	if len(node.children) == 0 {
		// Leaf: remove entirely.
		slog.Debug("evicting leaf", "offset", node.startOffset(), "tokens", len(node.tokens), "freed", mlx.PrettyBytes(int(node.snapshotBytes())))
		removeNode(node, &c.pagedOutBytes)
	} else if len(node.children) == 1 {
		// Interior node with one child: merge with child.
		before := c.pagedOutBytes
		tokens := len(node.tokens)
		mergeWithChild(node, c.caches, &c.pagedOutBytes)
		slog.Debug("evicting interior node", "offset", node.startOffset(), "tokens", tokens, "freed", mlx.PrettyBytes(int(before-c.pagedOutBytes)))
	} else {
		panic("evictNode called on multi-child branch point")
	}
}

func (c *kvCache) dumpTree() {
	// Summary stats
	var cacheBytes int
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		for _, a := range kv.State() {
			if a != nil {
				cacheBytes += a.NumBytes()
			}
		}
	}

	// Build active path set for marking.
	active := make(map[*trieNode]bool, len(c.activePath))
	for _, n := range c.activePath {
		active[n] = true
	}

	var nodeCount, snapshotCount int
	var pagedBytes int64
	var lines []string
	var dump func(n *trieNode, prefix string, isLast bool)
	dump = func(n *trieNode, prefix string, isLast bool) {
		if n == nil {
			return
		}
		nodeCount++

		// Build connector
		var connector string
		if n.parent == nil {
			connector = ""
		} else if isLast {
			connector = prefix + "`-- "
		} else {
			connector = prefix + "|-- "
		}

		// Node label
		nodeBytes := n.snapshotBytes()
		pagedBytes += nodeBytes

		label := fmt.Sprintf("[%d,%d) %dt", n.startOffset(), n.endOffset, len(n.tokens))
		if nodeBytes > 0 {
			label += " " + mlx.PrettyBytes(int(nodeBytes)).String()
		}
		if !n.lastUsed.IsZero() {
			label += fmt.Sprintf(" %s ago", time.Since(n.lastUsed).Truncate(time.Millisecond))
		}
		var flags []string
		if n.user {
			flags = append(flags, "user")
		}
		if n.hasAllSnapshots() {
			snapshotCount++
			flags = append(flags, "snap")
		}
		if active[n] {
			flags = append(flags, "active")
		}
		if len(flags) > 0 {
			label += " (" + flags[0]
			for _, f := range flags[1:] {
				label += ", " + f
			}
			label += ")"
		}
		lines = append(lines, connector+label)

		// Recurse children
		childPrefix := prefix
		if n.parent != nil {
			if isLast {
				childPrefix += "    "
			} else {
				childPrefix += "|   "
			}
		}
		for i, child := range n.children {
			dump(child, childPrefix, i == len(n.children)-1)
		}
	}
	dump(c.root, "", true)

	offset := c.minCacheOffset()
	logutil.Trace(fmt.Sprintf("kv cache active_tokens: %d, active_size: %s, paged_out: %s, trie: nodes=%d, snapshots=%d",
		offset, mlx.PrettyBytes(cacheBytes), mlx.PrettyBytes(int(pagedBytes)), nodeCount, snapshotCount))
	for i, l := range lines {
		if i == 0 {
			logutil.Trace("cache trie: " + l)
		} else {
			logutil.Trace("  " + l)
		}
	}
}
