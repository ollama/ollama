package structured

import (
	"bytes"
	mathbits "math/bits"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
)

// Vocab indexes a tokenizer's decoded pieces for mask computation. It is
// built once per loaded model and shared across requests; Mask is safe for
// concurrent use.
type Vocab struct {
	ids    []int32  // token ids sorted by piece bytes
	pieces [][]byte // piece per sorted index
	eos    []int32
	size   int32 // one past the largest token id

	mu    sync.Mutex
	cache map[uint64]*Mask
}

// maskCacheLimit bounds the per-vocab mask cache. States recur heavily
// (in-string, in-whitespace), so a small cache captures most steps; on
// overflow the cache is simply reset.
const maskCacheLimit = 256

// NewVocab builds the index. pieces is indexed by token id; nil or empty
// pieces (special tokens, padding) are never allowed. eos lists the token
// ids that are allowed exactly when the grammar can complete.
func NewVocab(pieces [][]byte, eos []int32) *Vocab {
	v := &Vocab{
		size:  int32(len(pieces)),
		eos:   append([]int32(nil), eos...),
		cache: make(map[uint64]*Mask),
	}
	for id, piece := range pieces {
		if len(piece) == 0 {
			continue
		}
		v.ids = append(v.ids, int32(id))
	}
	sort.Slice(v.ids, func(i, j int) bool {
		return bytes.Compare(pieces[v.ids[i]], pieces[v.ids[j]]) < 0
	})
	v.pieces = make([][]byte, len(v.ids))
	for i, id := range v.ids {
		v.pieces[i] = pieces[id]
	}
	return v
}

// Mask is the set of token ids legal in some matcher state.
type Mask struct {
	bits []uint64
}

// Allowed reports whether token id may be sampled.
func (m *Mask) Allowed(id int32) bool {
	if id < 0 || int(id) >= len(m.bits)*64 {
		return false
	}
	return m.bits[id>>6]&(1<<(id&63)) != 0
}

// ForEach calls f for every allowed token id in ascending order.
func (m *Mask) ForEach(f func(id int32)) {
	for wi, w := range m.bits {
		for w != 0 {
			b := mathbits.TrailingZeros64(w)
			f(int32(wi*64 + b))
			w &^= 1 << b
		}
	}
}

func (m *Mask) set(id int32) {
	m.bits[id>>6] |= 1 << (id & 63)
}

func (m *Mask) or(other *Mask) {
	for i, w := range other.bits {
		m.bits[i] |= w
	}
}

// Mask computes (with memoization) the set of tokens legal in the
// matcher's current state: every token whose full piece is a valid
// continuation, plus the EOS tokens when the grammar can complete.
func (v *Vocab) Mask(m *Matcher) *Mask {
	key := mix64(m.g.id*0x9e3779b97f4a7c15) ^ m.StateKey()

	v.mu.Lock()
	if cached, ok := v.cache[key]; ok {
		v.mu.Unlock()
		return cached
	}
	v.mu.Unlock()

	mask := v.computeMask(m)
	if m.CanComplete() {
		for _, id := range v.eos {
			if id >= 0 && id < v.size {
				mask.set(id)
			}
		}
	}

	v.mu.Lock()
	if len(v.cache) >= maskCacheLimit {
		v.cache = make(map[uint64]*Mask)
	}
	// Keep the first computed instance so equal states return identical
	// masks even under concurrent computation.
	if cached, ok := v.cache[key]; ok {
		mask = cached
	} else {
		v.cache[key] = mask
	}
	v.mu.Unlock()
	return mask
}

// computeMask fans the radix walk out over the top-level byte groups.
// Matcher frames are immutable, so workers may share them read-only; each
// worker owns a slab, scratch buffers, and a private mask that is OR-merged
// at the end.
func (v *Vocab) computeMask(m *Matcher) *Mask {
	words := (int(v.size) + 63) / 64
	mask := &Mask{bits: make([]uint64, words)}
	if len(v.ids) == 0 {
		return mask
	}

	type span struct {
		lo, hi int
		b      byte
	}
	var spans []span
	for i := 0; i < len(v.ids); {
		b := v.pieces[i][0]
		j := i + 1
		for j < len(v.ids) && v.pieces[j][0] == b {
			j++
		}
		spans = append(spans, span{lo: i, hi: j, b: b})
		i = j
	}

	workers := min(runtime.GOMAXPROCS(0), len(spans))
	if workers <= 1 {
		w := &walker{v: v, g: m.g, mask: mask}
		for _, sp := range spans {
			w.step(sp.lo, sp.hi, 0, m.stacks, sp.b)
		}
		return mask
	}

	var cursor atomic.Int64
	var wg sync.WaitGroup
	partial := make([]*Mask, workers)
	for wi := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := &walker{v: v, g: m.g, mask: &Mask{bits: make([]uint64, words)}}
			partial[wi] = w.mask
			for {
				i := int(cursor.Add(1)) - 1
				if i >= len(spans) {
					return
				}
				sp := spans[i]
				w.step(sp.lo, sp.hi, 0, m.stacks, sp.b)
			}
		}()
	}
	wg.Wait()
	for _, p := range partial {
		mask.or(p)
	}
	return mask
}

// frameSlab bump-allocates frames with stack-discipline reclamation: a
// mark taken before an advance releases every frame the advance and its
// subtree created. Live frames are only ever those along the current DFS
// path, so one walk allocates a handful of chunks total.
type frameSlab struct {
	chunks [][]frame
	ci     int
	used   int
}

const slabChunk = 4096

type slabMark struct {
	ci   int
	used int
}

func (s *frameSlab) alloc(parent *frame, ruleIdx, alt, idx int32) *frame {
	if s.ci < len(s.chunks) && s.used == len(s.chunks[s.ci]) {
		s.ci++
		s.used = 0
	}
	if s.ci == len(s.chunks) {
		s.chunks = append(s.chunks, make([]frame, slabChunk))
		s.used = 0
	}
	f := &s.chunks[s.ci][s.used]
	s.used++
	*f = frameValue(parent, ruleIdx, alt, idx)
	return f
}

func (s *frameSlab) mark() slabMark { return slabMark{ci: s.ci, used: s.used} }

func (s *frameSlab) release(m slabMark) { s.ci, s.used = m.ci, m.used }

type walker struct {
	v       *Vocab
	g       *Grammar
	mask    *Mask
	slab    frameSlab
	scratch [][]*frame // per-depth reusable state buffers
}

// step advances stacks over b into the depth's scratch buffer and, when b
// is legal, walks the piece range [lo,hi) at depth+1 under the advanced
// state. Slab frames and the buffer are reclaimed once the subtree
// completes; the self-loop case (states like string content that map to
// themselves) reuses the input state and keeps nothing.
func (w *walker) step(lo, hi, depth int, stacks []*frame, b byte) {
	for depth >= len(w.scratch) {
		w.scratch = append(w.scratch, nil)
	}
	mark := w.slab.mark()
	n := normalizer{g: w.g, out: w.scratch[depth][:0], alloc: w.slab.alloc}
	ok := n.advanceStacks(stacks, b)
	switch {
	case !ok:
		w.scratch[depth] = n.out[:0]
		w.slab.release(mark)
	case stackSetEqual(n.out, stacks):
		w.scratch[depth] = n.out[:0]
		w.slab.release(mark)
		w.walk(lo, hi, depth+1, stacks)
	default:
		next := n.out
		w.scratch[depth] = nil // in use by the subtree
		w.walk(lo, hi, depth+1, next)
		w.slab.release(mark)
		w.scratch[depth] = next[:0]
	}
}

// walk radix-descends the sorted pieces: [lo,hi) share the prefix
// pieces[lo][:depth], already consumed. Pieces that end here are allowed;
// longer pieces recurse per next byte.
func (w *walker) walk(lo, hi, depth int, stacks []*frame) {
	// Distinct ids may decode to identical bytes; every piece fully
	// consumed at this depth is allowed.
	for lo < hi && len(w.v.pieces[lo]) == depth {
		w.mask.set(w.v.ids[lo])
		lo++
	}
	for i := lo; i < hi; {
		b := w.v.pieces[i][depth]
		j := i + 1
		for j < hi && w.v.pieces[j][depth] == b {
			j++
		}
		w.step(i, j, depth, stacks, b)
		i = j
	}
}

func stackSetEqual(a, b []*frame) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if stackHash(a[i]) != stackHash(b[i]) || !frameEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}
