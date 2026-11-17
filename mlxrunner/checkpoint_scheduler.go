// Completed requests queue checkpoints for background saving. Between inference
// steps, the MLX thread stages bounded, owned chunks; a file worker writes and
// syncs them before publishing a complete checkpoint. Saved idle state can then
// be released from RAM. Restore uses read-only mappings to copy checked tensor
// bytes into independent MLX arrays, yielding between layers. Missing or invalid
// checkpoints fall back to prompt processing.
package mlxrunner

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
)

// runnerCheckpoints shares one disk index per cache layout across all slots.
// The scheduler is its sole owner; only bounded file operations run elsewhere.
type runnerCheckpoints struct {
	manifest  *manifest.Manifest
	layout    cacheLayout
	dir       string
	ready     chan struct{}
	stores    map[string]*checkpointCache
	pending   map[*prefixCache]*pendingCheckpoint
	idleAge   time.Duration
	diskLimit int64
}

type pendingCheckpoint struct {
	store      *checkpointCache
	cache      *prefixCache
	epoch      uint64
	nodes      []*trieNode
	media      []checkpointMedia
	used       time.Time
	writeAfter time.Time
	retry      time.Time
	saved      string // file covering the complete live state, if one was saved
}

func newRunnerCheckpoints(m *manifest.Manifest) *runnerCheckpoints {
	policy, enabled, err := loadCachePolicy()
	if err != nil {
		slog.Warn("Persistent KV cache unavailable", "error", err)
		return nil
	}
	if !enabled {
		return nil
	}
	return &runnerCheckpoints{
		manifest:  m,
		dir:       policy.dir,
		ready:     make(chan struct{}, 1),
		stores:    make(map[string]*checkpointCache),
		pending:   make(map[*prefixCache]*pendingCheckpoint),
		idleAge:   policy.idleAge,
		diskLimit: policy.diskLimit,
	}
}

func (p *runnerCheckpoints) open(c *prefixCache) *checkpointCache {
	if p == nil || c == nil || len(c.caches) == 0 {
		return nil
	}
	var layout strings.Builder
	fmt.Fprintln(&layout, c.draftLookahead)
	p.layout.writeTo(&layout)
	for _, layer := range c.caches {
		fmt.Fprintf(&layout, "%T\n", layer)
	}
	key := layout.String()
	if d, exists := p.stores[key]; exists {
		return d
	}
	p.stores[key] = nil // An unavailable store never fails the model load.
	runtimeID, modelID, err := cacheIdentity(p.manifest, c, p.layout)
	if err != nil {
		slog.Warn("Persistent KV identity unavailable", "error", err)
		return nil
	}
	identity := runtimeID + "-" + modelID
	ctx, cancel := context.WithTimeout(context.Background(), checkpointOpenTimeout)
	defer cancel()
	store, err := openCacheDiskStore(ctx, p.dir, identity)
	if err != nil {
		slog.Warn("Persistent KV store unavailable", "error", err)
		return nil
	}
	if p.diskLimit > 0 {
		store.limit = p.diskLimit
	}
	if err := store.purgeIncompatible(ctx); err != nil {
		slog.Debug("KV cleanup deferred", "error", err)
	}
	d, err := openCheckpointCache(ctx, store, identity, c.caches, p.ready)
	if err != nil {
		store.close()
		slog.Warn("Persistent KV recovery unavailable", "error", err)
		return nil
	}
	p.stores[key] = d
	logutil.Trace("KV disk cache opened", "directory", store.dir, "checkpoints", len(d.saved), "bytes", d.bytes, "shared_limit", store.limit, "free_reserve", store.reserve)
	return d
}

func (p *runnerCheckpoints) completed(s *cacheSession) {
	if p == nil || s == nil || len(s.cache.activePath) < 2 {
		return
	}
	c := s.cache
	d := p.open(c)
	if d == nil {
		return
	}
	// Keep a prompt restore point as well as the latest reply. Recurrent and
	// wrapped sliding state cannot rewind a reply checkpoint to the prompt.
	var nodes []*trieNode
	for i := len(c.activePath) - 2; i > 0; i-- {
		if c.activePath[i].user {
			nodes = append(nodes, c.activePath[i])
			break
		}
	}
	nodes = append(nodes, c.activePath[len(c.activePath)-1])
	q := &pendingCheckpoint{store: d, cache: c, epoch: c.epoch, nodes: nodes, used: time.Now()}
	q.writeAfter = q.used.Add(checkpointWriteDelay)
	for _, item := range s.items {
		q.media = append(q.media, checkpointMedia{item.pos, item.length, item.digest})
	}
	p.pending[c] = q
	select {
	case p.ready <- struct{}{}:
	default:
	}
}

// maintain advances local writes and selects idle slots whose durable state can
// leave RAM. The caller applies that decision separately; peers must not repeat
// the local timer or disk-existence checks.
func (p *runnerCheckpoints) maintain(slots []*prefixCache) []int {
	if p == nil {
		return nil
	}
	var release []int
	for _, d := range p.stores {
		if d == nil || d.job == nil {
			continue
		}
		job := d.job
		_, err := d.step()
		if d.job != nil {
			continue
		}
		q := p.pending[job.cache]
		if q != nil && q.epoch == job.epoch {
			if err != nil && !errors.Is(err, context.Canceled) {
				q.nodes = append(q.nodes, job.node)
				q.retry = time.Now().Add(checkpointRetryInterval)
				slog.Debug("KV checkpoint write deferred", "error", err)
			} else if err == nil && len(job.record.Keys) >= q.cache.minCacheOffset() {
				q.saved = job.file.Path()
			}
			// An idle slot can be compacted while its file is being written
			// or published. Save the surviving reply without a disk-error delay.
			compacted := job.node.endOffset != len(job.record.Keys) || !slices.Contains(q.cache.activePath, job.node)
			if compacted && (err == nil || errors.Is(err, context.Canceled)) && len(q.cache.activePath) > 1 {
				q.nodes = append(q.nodes, q.cache.activePath[len(q.cache.activePath)-1])
			}
		}
	}
	for c, q := range p.pending {
		if c.epoch != q.epoch {
			delete(p.pending, c)
			continue
		}
		if q.store.job != nil || time.Now().Before(q.retry) {
			continue
		}
		// Compaction can remove an earlier prompt node while the final reply
		// remains eligible. Skip it now: no I/O will arrive to wake us for it.
		for len(q.nodes) > 0 && !slices.Contains(c.activePath, q.nodes[0]) {
			q.nodes = q.nodes[1:]
		}
		if len(q.nodes) == 0 {
			if index := slices.Index(slots, c); index >= 0 && p.idleAge >= 0 && time.Since(q.used) >= p.idleAge {
				if q.canRelease() {
					release = append(release, index)
				} else {
					delete(p.pending, c)
				}
			}
			continue
		}
		node := q.nodes[0]
		var media []checkpointMedia
		for _, m := range q.media {
			if m.Position < node.endOffset+c.draftLookahead {
				media = append(media, m)
			}
		}
		alreadySaved := false
		for _, saved := range q.store.saved {
			if len(saved.record.Keys) != node.endOffset || !slices.Equal(saved.record.Media, media) {
				continue
			}
			matches := true
			for n := node; n.parent != nil && matches; n = n.parent {
				matches = slices.Equal(saved.record.Keys[n.startOffset():n.endOffset], n.tokens)
			}
			if !matches {
				continue
			}
			if _, err := os.Stat(saved.file.Path()); err == nil {
				saved.touch()
				if node.endOffset >= c.minCacheOffset() {
					q.saved = saved.file.Path()
				}
				alreadySaved = true
				break
			}
		}
		if alreadySaved {
			q.nodes = q.nodes[1:]
			// Continue scheduling without waiting for an I/O completion that
			// this already-durable checkpoint will not produce.
			select {
			case p.ready <- struct{}{}:
			default:
			}
			continue
		}
		if time.Now().Before(q.writeAfter) {
			continue
		}
		q.nodes = q.nodes[1:]
		if err := q.store.start(c, node, media); err != nil {
			q.nodes = append(q.nodes, node)
			q.retry = time.Now().Add(checkpointRetryInterval)
			slog.Debug("KV checkpoint admission deferred", "error", err)
		}
	}
	slices.Sort(release)
	return release
}

func (q *pendingCheckpoint) canRelease() bool {
	if q.cache.epoch != q.epoch || q.saved == "" {
		return false
	}
	// Disk eviction may have removed the saved state since the write completed.
	// Keep the resident copy in that case; rewriting it would evict a more
	// recently used conversation. A peer can still evict it after this check,
	// which is an ordinary cache miss, never permission to use partial state.
	if _, err := os.Stat(q.saved); err != nil {
		return false
	}
	return true
}

func (p *runnerCheckpoints) release(slots []*prefixCache, indexes []int) {
	for _, index := range indexes {
		c := slots[index]
		c.close()
		delete(p.pending, c)
		logutil.Trace("Released persisted idle KV state", "slot", index)
	}
}

// reclaimSaved gives pressure relief a cheap first choice: inactive state
// which can already be restored from disk, oldest conversation first. The
// caller supplies idle slots; a saved checkpoint never makes active KV evictable.
func (p *runnerCheckpoints) reclaimSaved(slots, idle []*prefixCache, preferred map[*trieNode]int, bytes int64) int64 {
	if p == nil || bytes <= 0 {
		return 0
	}
	before := mlx.ActiveMemory()
	for _, index := range p.savedVictims(slots, idle, preferred) {
		if int64(before-mlx.ActiveMemory()) >= bytes {
			break
		}
		p.release(slots, []int{index})
	}
	released := int64(before - mlx.ActiveMemory())
	if released > 0 {
		mlx.ClearCache()
	}
	return released
}

// savedVictims is the only step that consults local checkpoint availability.
// Applying its slot indexes needs neither a pending write nor a local file.
func (p *runnerCheckpoints) savedVictims(slots, idle []*prefixCache, preferred map[*trieNode]int) []int {
	if p == nil {
		return nil
	}
	// A disk copy is cheaper to discard only among equally useful resident
	// prefixes. Never evict the arriving request's match ahead of colder history.
	_, cold, _ := oldestColdHistory(slots, idle, preferred)
	var victims []int
	for i, c := range slots {
		if c == nil || cold != nil && preferred[c.root] > preferred[cold] {
			continue
		}
		if q := p.pending[c]; q != nil && slices.Contains(idle, c) && q.canRelease() {
			victims = append(victims, i)
		}
	}
	slices.SortStableFunc(victims, func(a, b int) int {
		return cmp.Or(cmp.Compare(preferred[slots[a].root], preferred[slots[b].root]), p.pending[slots[a]].used.Compare(p.pending[slots[b]].used))
	})
	return victims
}

// finishPending gives an already queued checkpoint a bounded chance to finish
// before pressure forces its resident state to be discarded. The caller selects
// and releases each saved victim, recalculating its budget after every release.
func (p *runnerCheckpoints) finishPending(ctx context.Context, slots, idle []*prefixCache, preferred map[*trieNode]int) bool {
	if p == nil {
		return false
	}
	for ctx.Err() == nil {
		pending := false
		for _, c := range idle {
			q := p.pending[c]
			if q == nil || q.epoch != c.epoch || time.Now().Before(q.retry) {
				continue
			}
			q.writeAfter = time.Time{}
			pending = pending || len(q.nodes) > 0 || q.store.job != nil && q.store.job.cache == c
		}
		if !pending {
			return false
		}
		mlx.Scoped(func() { p.maintain(nil) })
		if len(p.savedVictims(slots, idle, preferred)) > 0 {
			return true
		}
		select {
		case <-ctx.Done():
		case <-p.wake():
		}
	}
	return false
}

func (p *runnerCheckpoints) wake() <-chan struct{} {
	if p == nil {
		return nil
	}
	return p.ready
}

// I/O completions wake the scheduler themselves. A timer is needed only for
// delayed writes, retries, or releasing saved idle state under the chosen policy.
func (p *runnerCheckpoints) timer() <-chan time.Time {
	if p == nil {
		return nil
	}
	var next time.Time
	for _, q := range p.pending {
		if q.store.job != nil {
			continue
		}
		var due time.Time
		if len(q.nodes) > 0 {
			due = q.writeAfter
			if q.retry.After(due) {
				due = q.retry
			}
		} else if p.idleAge >= 0 {
			due = q.used.Add(p.idleAge)
		} else {
			continue
		}
		if due.IsZero() {
			return time.After(0)
		}
		if next.IsZero() || due.Before(next) {
			next = due
		}
	}
	if next.IsZero() {
		return nil
	}
	return time.After(time.Until(next))
}

func (p *runnerCheckpoints) close() {
	if p == nil {
		return
	}
	for _, d := range p.stores {
		d.close()
	}
}

// flushCheckpoints finishes completed requests' writes before a normal model
// unload. The caller bounds the wait; a slow or unavailable disk must not keep
// the runner resident indefinitely. All cache access stays on the MLX worker.
func (r *Runner) flushCheckpoints(ctx context.Context) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		pending := false
		err := r.onMLX(ctx, func() error {
			p := r.checkpoints
			if p == nil {
				return nil
			}
			for _, q := range p.pending {
				q.writeAfter = time.Time{}
			}
			mlx.Scoped(func() { p.maintain(nil) })
			for _, d := range p.stores {
				pending = pending || d != nil && d.job != nil
			}
			for _, q := range p.pending {
				if len(q.nodes) > 0 {
					if time.Now().Before(q.retry) {
						return fmt.Errorf("checkpoint write deferred after disk error")
					}
					pending = true
				}
			}
			return nil
		})
		if err != nil || !pending {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-r.checkpoints.wake():
		}
	}
}
