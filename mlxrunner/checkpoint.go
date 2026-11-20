package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/cache"
)

// A checkpoint is self-contained: a disk eviction cannot strand the suffix of
// another conversation without its prefix. Files contain exact cache state,
// token keys, and strong media identities, never model weights or MLX handles.
type checkpointRecord struct {
	Identity  string
	Lookahead int
	User      bool
	Keys      []trieKey
	Media     []checkpointMedia
	Layers    []checkpointLayer
}

type checkpointMedia struct {
	Position, Length int
	Digest           [32]byte
}

type checkpointLayer struct {
	Kind    string
	State   cache.SnapshotData
	Tensors []mlx.TensorExtent
}

type savedCheckpoint struct {
	file     *mlx.TensorFile
	record   checkpointRecord
	used     time.Time
	touched  time.Time
	pins     int
	size     int64
	metadata int
}

func (s *savedCheckpoint) touch() {
	s.used = time.Now()
	if s.used.Sub(s.touched) >= checkpointTouchInterval {
		if err := os.Chtimes(s.file.Path(), s.used, s.used); err == nil {
			s.touched = s.used
		}
	}
}

// checkpointCache owns only disk metadata. Request slots keep their existing
// resident tries and share this index, so increasing parallelism does not
// multiply recovery metadata or disk quotas.
type checkpointCache struct {
	store    *cacheDiskStore
	identity string
	saved    []*savedCheckpoint
	bytes    int64
	metadata int
	limit    int64
	ready    chan struct{}
	job      *checkpointWrite
}

func openCheckpointCache(ctx context.Context, store *cacheDiskStore, identity string, caches []cache.Cache, ready chan struct{}) (*checkpointCache, error) {
	guard, err := store.lock(ctx)
	if err != nil {
		return nil, err
	}
	defer guard.Close()
	d := &checkpointCache{store: store, identity: identity, limit: min(store.limit, checkpointInitialLimit), ready: ready}
	entries, err := os.ReadDir(store.dir)
	if err != nil {
		return nil, err
	}
	var metadataBytes int
	type checkpointFile struct {
		name string
		info os.FileInfo
	}
	var files []checkpointFile
	for _, entry := range entries {
		if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 || filepath.Ext(entry.Name()) != ".kv" {
			continue
		}
		if info, err := entry.Info(); err == nil {
			files = append(files, checkpointFile{entry.Name(), info})
		}
	}
	slices.SortFunc(files, func(a, b checkpointFile) int { return b.info.ModTime().Compare(a.info.ModTime()) })
	for _, entry := range files {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		path := filepath.Join(store.dir, entry.name)
		file, metadata, err := mlx.OpenTensorFile(path)
		if err != nil {
			_ = os.Remove(path)
			continue
		}
		var record checkpointRecord
		err = json.Unmarshal(metadata, &record)
		if err == nil {
			err = record.validate(identity, caches, file.Size())
		}
		if err != nil {
			file.Close()
			_ = os.Remove(path)
			continue
		}
		metadataBytes += len(metadata)
		if len(d.saved) >= maxCheckpointFiles || metadataBytes > maxCheckpointMetadata {
			file.Close()
			_ = os.Remove(path)
			continue
		}
		d.saved = append(d.saved, &savedCheckpoint{file: file, record: record, used: entry.info.ModTime(), size: entry.info.Size(), metadata: len(metadata)})
		d.bytes += entry.info.Size()
		d.metadata += len(metadata)
		d.grow(file.Size())
	}
	return d, nil
}

func (r checkpointRecord) validate(identity string, caches []cache.Cache, size int64) error {
	if r.Identity != identity || len(r.Keys) == 0 || len(r.Keys) > maxCheckpointTokens || len(r.Layers) != len(caches) || r.Lookahead < 0 || r.Lookahead > 1 {
		return fmt.Errorf("incompatible checkpoint")
	}
	end := len(r.Keys)
	tensors := 0
	for i, layer := range r.Layers {
		if caches[i] == nil {
			if layer.Kind != "" || len(layer.Tensors) != 0 {
				return fmt.Errorf("state for stateless cache layer %d", i)
			}
			continue
		}
		if _, ok := caches[i].(cache.SnapshotCodec); !ok || layer.Kind != fmt.Sprintf("%T", caches[i]) || len(layer.Tensors) == 0 || len(layer.Tensors) > 8 {
			return fmt.Errorf("incompatible checkpoint layer %d", i)
		}
		state := layer.State
		if state.End != end || state.Start < 0 || state.Start >= end || state.SequenceAxis < -1 || state.SequenceAxis > 7 || state.SequenceAxis >= 0 && state.Start != 0 || state.Cursor < 0 {
			return fmt.Errorf("invalid checkpoint bounds")
		}
		for _, e := range layer.Tensors {
			if err := e.Validate(size); err != nil {
				return err
			}
			tensors++
		}
	}
	if tensors == 0 {
		return fmt.Errorf("empty checkpoint")
	}
	last := 0
	for _, m := range r.Media {
		if m.Position < last || m.Length <= 0 || m.Position > end || m.Length > maxCheckpointTokens-m.Position || m.Digest == ([32]byte{}) {
			return fmt.Errorf("invalid checkpoint media identity")
		}
		last = m.Position + m.Length
	}
	media := 0
	for pos := range end + r.Lookahead {
		var token uint32
		if r.Lookahead == 0 {
			token = uint32(r.Keys[pos])
		} else if pos < end {
			token = uint32(uint64(r.Keys[pos]) >> 32)
		} else {
			token = uint32(r.Keys[pos-1])
		}
		for media < len(r.Media) && r.Media[media].Position+r.Media[media].Length <= pos {
			media++
		}
		covered := media < len(r.Media) && r.Media[media].Position <= pos
		if (token>>31 != 0) != covered {
			return fmt.Errorf("checkpoint token lacks matching media identity")
		}
	}
	return nil
}

// grow expands the initial 1 GiB allowance to hold roughly four conversations
// with separate prompt and reply checkpoints. The shared disk quota still caps it.
func (d *checkpointCache) grow(size int64) {
	if size > 0 && size <= math.MaxInt64/checkpointSizeMultiple {
		d.limit = min(d.store.limit, max(d.limit, checkpointSizeMultiple*size))
	}
}

// trim removes whole old checkpoints. A restore in this runner pins its file
// across scheduler rounds. A peer deleting a file is treated as an ordinary
// cache miss; partial state is never used for generation.
func (d *checkpointCache) trim(extra int64) bool {
	slices.SortStableFunc(d.saved, func(a, b *savedCheckpoint) int { return a.used.Compare(b.used) })
	for _, s := range slices.Clone(d.saved) {
		if d.bytes <= d.limit-extra && len(d.saved) < maxCheckpointFiles && d.metadata <= maxCheckpointMetadata-mlx.TensorFileMetadataLimit {
			return true
		}
		if s.pins == 0 {
			d.remove(s)
		}
	}
	return d.bytes <= d.limit-extra && len(d.saved) < maxCheckpointFiles && d.metadata <= maxCheckpointMetadata-mlx.TensorFileMetadataLimit
}

func (d *checkpointCache) remove(s *savedCheckpoint) {
	if err := os.Remove(s.file.Path()); err != nil && !os.IsNotExist(err) {
		return // Windows readers can defer deletion; keep it charged.
	}
	s.file.Close()
	if i := slices.Index(d.saved, s); i >= 0 {
		d.bytes -= s.size
		d.metadata -= s.metadata
		d.saved = slices.Delete(d.saved, i, i+1)
	}
}

// checkpointState borrows a complete prefix from an idle slot. Sliceable state
// comes from the live prefix; whole-state layers must have a capture at end.
// No extra snapshot owner is created, so resuming the slot need not copy the
// whole cache merely to keep a background write alive.
func checkpointState(c *prefixCache, node *trieNode) ([]cache.SnapshotData, error) {
	states := make([]cache.SnapshotData, len(c.caches))
	for i := range c.caches {
		data, err := checkpointLayerState(c, node, i)
		if err != nil {
			return nil, err
		}
		states[i] = data
	}
	return states, nil
}

func checkpointLayerState(c *prefixCache, node *trieNode, i int) (cache.SnapshotData, error) {
	if node == nil || !slices.Contains(c.activePath, node) || node.endOffset <= 0 {
		return cache.SnapshotData{}, fmt.Errorf("checkpoint is no longer on the live path")
	}
	layer := c.caches[i]
	if layer == nil {
		return cache.SnapshotData{}, nil
	}
	codec, ok := layer.(cache.SnapshotCodec)
	if !ok {
		return cache.SnapshotData{}, fmt.Errorf("cache layer %T has no persistence codec", layer)
	}
	data := codec.SnapshotData(nil)
	if data.SequenceAxis >= 0 {
		if data.Start != 0 || data.End < node.endOffset {
			return cache.SnapshotData{}, fmt.Errorf("incomplete live prefix")
		}
		for j, a := range data.Arrays {
			start := make([]int32, a.NumDims())
			stop := make([]int32, a.NumDims())
			for k, dim := range a.Dims() {
				stop[k] = int32(dim)
			}
			stop[data.SequenceAxis] = int32(node.endOffset)
			data.Arrays[j] = mlx.SliceStartStop(a, start, stop)
		}
		data.End = node.endOffset
	} else if data.End != node.endOffset {
		if i >= len(node.snapshots) || node.snapshots[i] == nil {
			return cache.SnapshotData{}, fmt.Errorf("missing whole-state checkpoint")
		}
		data = codec.SnapshotData(node.snapshots[i])
	}
	if data.End != node.endOffset || len(data.Arrays) == 0 || slices.Contains(data.Arrays, nil) {
		return cache.SnapshotData{}, fmt.Errorf("incomplete checkpoint layer")
	}
	return data, nil
}

type checkpointWrite struct {
	cache       *prefixCache
	node        *trieNode
	epoch       uint64
	record      checkpointRecord
	file        *mlx.TensorFile
	reservation *cacheReservation
	writer      *mlx.TensorWriter
	layer, part int
	counts      []int
	ctx         context.Context //nolint:containedctx // A queued write owns cancellation until its background I/O has drained.
	cancel      context.CancelFunc
	done        chan error
	published   bool
	metadata    int
	started     time.Time
	staging     time.Duration
}

func (d *checkpointCache) start(c *prefixCache, node *trieNode, media []checkpointMedia) error {
	if d.job != nil {
		return fmt.Errorf("checkpoint writer busy")
	}
	if node == nil || node.endOffset > maxCheckpointTokens {
		return fmt.Errorf("checkpoint metadata exceeds token limit")
	}
	states, err := checkpointState(c, node)
	if err != nil {
		return err
	}
	record := checkpointRecord{Identity: d.identity, Lookahead: c.draftLookahead, User: node.user, Keys: make([]trieKey, node.endOffset), Media: media, Layers: make([]checkpointLayer, len(states))}
	for n := node; n.parent != nil; n = n.parent {
		copy(record.Keys[n.startOffset():n.endOffset], n.tokens)
	}
	var bytes int64 = mlx.TensorFileMetadataLimit + 64
	for i, data := range states {
		if len(data.Arrays) == 0 {
			continue
		}
		for _, a := range data.Arrays {
			n := int64(a.NumBytes())
			if n <= 0 || n > math.MaxInt64-bytes-mlx.TensorFileAlignment {
				return fmt.Errorf("invalid checkpoint size")
			}
			bytes += (n + mlx.TensorFileAlignment - 1) / mlx.TensorFileAlignment * mlx.TensorFileAlignment
		}
		data.Arrays = nil
		record.Layers[i] = checkpointLayer{Kind: fmt.Sprintf("%T", c.caches[i]), State: data}
	}
	d.grow(bytes)
	if bytes > d.limit || !d.trim(bytes) {
		return fmt.Errorf("checkpoint exceeds model disk allowance")
	}
	ctx, cancel := context.WithCancel(context.Background())
	j := &checkpointWrite{cache: c, node: node, epoch: c.epoch, record: record, ctx: ctx, cancel: cancel, started: time.Now()}
	j.counts = make([]int, len(states))
	for i, state := range states {
		j.counts[i] = len(state.Arrays)
	}
	d.job = j
	d.io(func() (err error) {
		j.file, err = mlx.NewTensorFile(d.store.dir)
		if err == nil {
			j.reservation, err = d.store.reserveWrite(ctx, j.file.Path(), bytes)
		}
		return err
	})
	return nil
}

// io transfers ownership of the file and writer to one background operation.
// The MLX worker observes them again only after receiving its completion.
func (d *checkpointCache) io(fn func() error) {
	j := d.job
	j.done = make(chan error, 1)
	go func() {
		j.done <- fn()
		select {
		case d.ready <- struct{}{}:
		default:
		}
	}()
}

// step stages at most 8 MiB. The caller runs on MLX and must leave the source
// slot idle until the write completes, or invalidate it by beginning a request.
func (d *checkpointCache) step() (bool, error) {
	j := d.job
	if j == nil {
		return false, nil
	}
	if j.done != nil {
		select {
		case err := <-j.done:
			j.done = nil
			if err != nil {
				d.abort()
				return true, err
			}
		default:
			return false, nil
		}
	}
	if j.published {
		info, err := os.Stat(j.file.Path())
		if err != nil {
			d.abort()
			return true, err
		}
		d.saved = append(d.saved, &savedCheckpoint{file: j.file, record: j.record, used: time.Now(), size: info.Size(), metadata: j.metadata})
		d.bytes += info.Size()
		d.metadata += j.metadata
		slog.Debug("KV checkpoint saved", "tokens", len(j.record.Keys), "bytes", info.Size(), "elapsed", time.Since(j.started), "staging", j.staging)
		j.cancel()
		d.job = nil
		return true, nil
	}
	// Compaction can merge another node into this one without changing the
	// request epoch. Its new state no longer belongs to the captured metadata.
	if j.epoch != j.cache.epoch || j.node.endOffset != len(j.record.Keys) || !slices.Contains(j.cache.activePath, j.node) {
		d.abort()
		return true, context.Canceled
	}
	for j.layer < len(j.counts) && j.part == j.counts[j.layer] {
		j.layer++
		j.part = 0
	}
	if j.layer == len(j.counts) {
		if err := j.record.validate(d.identity, j.cache.caches, j.file.Size()); err != nil {
			d.abort()
			return true, err
		}
		d.io(func() error {
			if err := j.ctx.Err(); err != nil {
				return err
			}
			metadata, err := json.Marshal(j.record)
			if err != nil {
				return err
			}
			j.metadata = len(metadata)
			if err := j.file.Seal(metadata); err != nil {
				return err
			}
			guard, err := d.store.lock(j.ctx)
			if err != nil {
				return err
			}
			defer guard.Close()
			if err := j.file.Publish(); err != nil {
				return err
			}
			j.published = true
			return j.reservation.Close()
		})
		return true, nil
	}
	dataState, err := checkpointLayerState(j.cache, j.node, j.layer)
	if err != nil {
		d.abort()
		return true, err
	}
	a := dataState.Arrays[j.part]
	if j.writer == nil {
		j.writer, err = j.file.BeginTensor(a)
		if err != nil {
			d.abort()
			return true, err
		}
	}
	start := time.Now()
	data, err := j.writer.Stage(j.ctx, a)
	j.staging += time.Since(start)
	if err != nil {
		d.abort()
		return true, err
	}
	d.io(func() error {
		if err := j.writer.Write(j.ctx, data); err != nil {
			return err
		}
		if j.writer.Done() {
			layer := &j.record.Layers[j.layer]
			layer.Tensors = append(layer.Tensors, j.writer.Extent())
			j.part++
			j.writer = nil
		}
		return nil
	})
	return true, nil
}

func (d *checkpointCache) abort() {
	j := d.job
	if j == nil {
		return
	}
	j.cancel()
	if j.done != nil {
		<-j.done
	}
	if j.file != nil {
		logutil.Trace("KV checkpoint abandoned", "tokens", len(j.record.Keys), "bytes", j.file.Size(), "elapsed", time.Since(j.started), "staging", j.staging)
		j.file.Close()
	}
	j.reservation.Close()
	d.job = nil
}

func (d *checkpointCache) close() {
	if d == nil {
		return
	}
	d.abort()
	for _, s := range d.saved {
		s.file.Close()
	}
	d.store.close()
}

type checkpointRestore struct {
	store   *checkpointCache
	saved   *savedCheckpoint
	cache   *prefixCache
	layer   int
	started time.Time
	reading time.Duration
	done    bool
	err     error
}

func (d *checkpointCache) findRestore(c *prefixCache, inputs []int32, items []mediaItem, maxReuse, resident int) *savedCheckpoint {
	if d == nil || resident >= maxReuse {
		return nil
	}
	var keys []trieKey
	var best *savedCheckpoint
	for _, s := range d.saved {
		n := len(s.record.Keys)
		if n <= resident || n > maxReuse || n > len(inputs)-c.draftLookahead {
			continue
		}
		if keys == nil {
			keys = c.key(effectiveKeyTokens(inputs, items))
		}
		if !slices.Equal(s.record.Keys, keys[:n]) {
			continue
		}
		if !s.record.matchesMedia(items, n+c.draftLookahead) {
			continue
		}
		if insideAtomicItem(items, n) != nil {
			continue
		}
		if best == nil || n > len(best.record.Keys) || n == len(best.record.Keys) && s.used.After(best.used) {
			best = s
		}
	}
	if best == nil {
		logutil.Trace("KV disk lookup", "hit", false, "resident", resident, "checkpoints", len(d.saved))
		return nil
	}
	return best
}

// restore starts the best eligible prefix. Scheduler replay restricts this to
// one exact prefix length, then waits for a common completion decision.
func (d *checkpointCache) restore(c *prefixCache, inputs []int32, items []mediaItem, maxReuse, resident int) *checkpointRestore {
	best := d.findRestore(c, inputs, items, maxReuse, resident)
	if best == nil {
		return nil
	}
	c.ensureRoot()
	best.pins++
	logutil.Trace("KV disk lookup", "hit", true, "tokens", len(best.record.Keys), "bytes", best.size, "resident", resident)
	return &checkpointRestore{store: d, saved: best, cache: c, started: time.Now()}
}

func (r checkpointRecord) matchesMedia(items []mediaItem, end int) bool {
	i := 0
	for _, item := range items {
		if item.pos >= end {
			break
		}
		if i == len(r.Media) || item.digest == ([32]byte{}) || r.Media[i] != (checkpointMedia{item.pos, item.length, item.digest}) {
			return false
		}
		i++
	}
	return i == len(r.Media)
}

// step restores one layer, then yields to the other clients. The destination
// cannot execute a forward until every layer is at the same checkpoint.
func (p *checkpointRestore) step(ctx context.Context) (bool, error) {
	if p.done {
		return true, p.err
	}
	if err := ctx.Err(); err != nil {
		p.fail(err)
		return true, err
	}
	if p.layer == 0 {
		p.cache.freeAll()
		p.cache.activePath = []*trieNode{p.cache.root}
	}
	for p.layer < len(p.cache.caches) && p.cache.caches[p.layer] == nil {
		p.layer++
	}
	if p.layer < len(p.cache.caches) {
		start := time.Now()
		defer func() { p.reading += time.Since(start) }()
		layer := p.saved.record.Layers[p.layer]
		state := layer.State
		for _, e := range layer.Tensors {
			a, err := p.saved.file.Read(ctx, e)
			if err != nil {
				p.fail(err)
				return true, err
			}
			state.Arrays = append(state.Arrays, a)
		}
		c := p.cache.caches[p.layer]
		if !c.(cache.SnapshotCodec).RestoreData(state, len(p.saved.record.Keys)) {
			err := fmt.Errorf("checkpoint state does not match cache layer")
			p.fail(err)
			return true, err
		}
		mlx.Eval(c.State()...)
		p.layer++
		return false, nil
	}
	p.done = true
	return true, nil
}

// commit publishes restored layers only after the scheduler accepts completion.
// Until then, a peer's failed restore can discard these layers without adding
// a restore point which a fallback would immediately pick up again.
func (p *checkpointRestore) commit() error {
	if p == nil || !p.done || p.err != nil || p.saved == nil {
		return fmt.Errorf("checkpoint restore is not ready")
	}
	n := p.cache.advancePath(p.cache.root, p.saved.record.Keys, len(p.saved.record.Keys))
	n.user = p.saved.record.User
	p.cache.pageOut(n)
	p.saved.touch()
	logutil.Trace("KV checkpoint restored", "tokens", len(p.saved.record.Keys), "bytes", p.saved.size, "elapsed", time.Since(p.started), "reading", p.reading)
	p.close(true)
	return nil
}

// stepAll is useful when ranks own different layer counts. They each finish
// their local work before a shared completion decision; ordinary scheduling
// continues to use step and yield after one layer.
func (p *checkpointRestore) stepAll(ctx context.Context) (bool, error) {
	for {
		if done, err := p.step(ctx); done {
			return true, err
		}
	}
}

func (p *checkpointRestore) fail(err error) {
	p.done, p.err = true, err
	saved := p.saved
	p.close(false)
	if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		p.store.remove(saved)
	}
}

func (p *checkpointRestore) close(success bool) {
	if p == nil || p.saved == nil {
		return
	}
	p.saved.pins--
	if !success {
		p.cache.freeAll()
		p.cache.activePath = []*trieNode{p.cache.root}
	}
	p.saved = nil
}
