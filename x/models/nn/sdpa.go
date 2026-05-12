package nn

import (
	"encoding/binary"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// SDPAOption configures a call to ScaledDotProductAttention.
type SDPAOption func(*sdpaConfig)

type sdpaConfig struct {
	// Exactly one of (k,v,kLens) or history supplies keys/values.
	k, v    *mlx.Array
	kLens   []int32
	history *KVHistory

	// Optional model-supplied logical mask.
	mask AttentionMask
}

// WithKVHistory supplies a cache's per-layer view of K and V. The
// cache hides any storage layout (sliding window, ring buffer,
// k-padding) behind the history.
func WithKVHistory(h *KVHistory) SDPAOption {
	return func(c *sdpaConfig) { c.history = h }
}

// WithMLAHistory supplies a cache's per-layer view for absorbed MLA
// attention, where V is the first valueDim positions of K.
func WithMLAHistory(h *KVHistory, valueDim int) SDPAOption {
	v := h.K().Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(0, valueDim))
	return WithKVHistory(&KVHistory{k: h.K(), v: v, applier: h.applier})
}

// WithKV supplies explicit K/V tensors for the no-cache path. kLens
// gives per-row real key extents — pass b.SeqQueryLens for self-
// attention, or the caller's own extents for cross-attention.
func WithKV(k, v *mlx.Array, kLens []int32) SDPAOption {
	return func(c *sdpaConfig) { c.k = k; c.v = v; c.kLens = kLens }
}

// WithMask supplies the model's logical-coordinate mask.
func WithMask(m AttentionMask) SDPAOption {
	return func(c *sdpaConfig) { c.mask = m }
}

// ScaledDotProductAttention runs the fast SDPA kernel against q and
// the keys/values supplied via exactly one of WithKV or
// WithKVHistory. Automatically applies any Q/K padding masking required
// for padded batches.
func ScaledDotProductAttention(b *batch.Batch, q *mlx.Array, scale float32, opts ...SDPAOption) *mlx.Array {
	var cfg sdpaConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	haveKV := cfg.k != nil || cfg.v != nil
	haveHistory := cfg.history != nil
	if haveKV && haveHistory {
		panic("nn.ScaledDotProductAttention: WithKV and WithKVHistory are mutually exclusive")
	}
	if !haveKV && !haveHistory {
		panic("nn.ScaledDotProductAttention: no keys/values supplied (use WithKV or WithKVHistory)")
	}

	k, v := cfg.k, cfg.v
	var applier MaskApplier
	if cfg.history != nil {
		k = cfg.history.K()
		v = cfg.history.V()
		applier = cfg.history.applier
	}

	inputs := dispatchInputs{
		batch:   b,
		mask:    cfg.mask,
		applier: applier,
		K:       k.Dim(2),
		dtype:   k.DType(),
		kLens:   newKLensKey(cfg.kLens),
	}

	if cached, ok := b.Memo.Get(inputs); ok {
		d := cached.(sdpaDispatch)
		return mlx.FastScaledDotProductAttention(q, k, v, scale, d.mode, d.arr)
	}

	d := inputs.resolve()
	b.Memo.Put(inputs, d)
	return mlx.FastScaledDotProductAttention(q, k, v, scale, d.mode, d.arr)
}

// sdpaDispatch is the resolved kernel call for a given SDPA key —
// either a flag-mode fast path (mode "" or "causal", arr nil) or an
// array-mode call with a materialized tensor. Memoized on b.Memo so
// sibling layers skip applier composition, padding build, and AsArray.
type sdpaDispatch struct {
	mode string
	arr  *mlx.Array
}

// dispatchInputs bundles every value resolve reads and doubles as
// the Memo map key. All fields are comparable: batch is a
// *batch.Batch pointer, the applier interface is comparable when
// its concrete type is, and kLens is a kLensKey string that hashes
// by content.
//
// Making resolve a method on this struct is the enforcement — any
// new dependency must be added as a field, which automatically
// participates in the map key.
//
// applier and kLens are mutually exclusive by construction:
// WithKVHistory sets applier (which owns any K-padding in its output
// space) and leaves kLens ""; WithKV sets kLens and leaves applier nil.
type dispatchInputs struct {
	batch   *batch.Batch
	mask    AttentionMask
	applier MaskApplier
	K       int
	dtype   mlx.DType
	kLens   kLensKey
}

// kLensKey is a comparable encoding of an int32 slice (four bytes
// per element, native endian) so it can live in a struct used as a
// map key. Decode back via Int32s.
type kLensKey string

func newKLensKey(vs []int32) kLensKey {
	if len(vs) == 0 {
		return ""
	}
	buf := make([]byte, len(vs)*4)
	for i, v := range vs {
		binary.NativeEndian.PutUint32(buf[i*4:], uint32(v))
	}
	return kLensKey(buf)
}

// Int32s decodes the key back to a fresh []int32.
func (k kLensKey) Int32s() []int32 {
	if len(k) == 0 {
		return nil
	}
	b := []byte(k)
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(binary.NativeEndian.Uint32(b[i*4:]))
	}
	return out
}

// resolve composes model + padding + storage contributions and
// returns the kernel dispatch decision. Reads only from inputs; any
// new input must be added to dispatchInputs.
//
// Order matters: QPaddingMask is added in logical Q-space before the
// applier runs, so an applier that remaps coordinates receives the
// full logical mask. The applier and KPaddingMask branches are
// mutually exclusive — on the applier path the output may be in a
// remapped K space, so the applier owns any K-padding; on the
// WithKV path kLens describes the direct K tensor, which shares
// logical K space with QPaddingMask.
func (inputs dispatchInputs) resolve() sdpaDispatch {
	mask := inputs.mask.Intersect(QPaddingMask(inputs.batch, inputs.dtype))

	if inputs.applier != nil {
		mask = inputs.applier.ApplyMask(mask)
	} else if inputs.kLens != "" {
		mask = mask.Intersect(KPaddingMask(inputs.batch, inputs.K, inputs.kLens.Int32s(), inputs.dtype))
	}

	switch {
	case mask.IsZero():
		return sdpaDispatch{mode: ""}
	case mask.IsCausal():
		if inputs.batch.InputIDs.Dim(1) == 1 {
			// At L=1 the causal "k > q" constraint is redundant -
			// drop it so the kernel dispatches to the no-mask fast path.
			return sdpaDispatch{mode: ""}
		} else {
			return sdpaDispatch{mode: "causal"}
		}
	default:
		return sdpaDispatch{mode: "array", arr: mask.AsArray(inputs.batch, inputs.K, inputs.dtype)}
	}
}

// MaskApplier composes a cache's storage-mask contribution onto a
// fully-composed logical mask. The returned mask may live in the
// applier's own coordinate system (e.g. a rotated or compacted K layout),
// so any addition in logical K space must happen before the applier runs.
// SDPA does not add KPaddingMask on this path — the applier owns any
// K-padding its output needs.
//
// Implementations must be comparable struct values whose fields
// capture everything the composition depends on (no slice, map, or
// func fields); the value doubles as the applier's identity in
// SDPA's dispatch-cache key, where a non-comparable concrete type
// would panic at map insertion. A nil MaskApplier means "no storage
// contribution".
type MaskApplier interface {
	ApplyMask(logical AttentionMask) AttentionMask
}

// KVHistory is the per-forward view a KV cache hands to SDPA:
// post-Update K and V plus an optional MaskApplier that composes
// the cache's storage mask onto the caller's model mask.
type KVHistory struct {
	k, v    *mlx.Array
	applier MaskApplier
}

// NewKVHistory constructs a KVHistory. Intended for
// cache implementations across packages; model code uses
// WithKVHistory / WithKV instead.
func NewKVHistory(k, v *mlx.Array, applier MaskApplier) *KVHistory {
	return &KVHistory{k: k, v: v, applier: applier}
}

// K returns the post-Update keys tensor.
//
// Last-resort escape hatch for custom attention paths — may force a
// slow materialization to canonical form depending on the cache's
// internal storage. Prefer ScaledDotProductAttention via
// WithKVHistory.
func (h *KVHistory) K() *mlx.Array { return h.k }

// V returns the post-Update values tensor.
//
// Last-resort escape hatch for custom attention paths — may force a
// slow materialization to canonical form depending on the cache's
// internal storage. Prefer ScaledDotProductAttention via
// WithKVHistory.
func (h *KVHistory) V() *mlx.Array { return h.v }

// Mask returns the final AttentionMask for this layer's SDPA —
// cache storage restrictions composed onto the caller's fully-
// composed logical mask.
//
// Last-resort escape hatch for custom attention paths — may force a
// slow materialization to canonical form depending on the cache's
// internal storage. Prefer ScaledDotProductAttention via
// WithKVHistory.
func (h *KVHistory) Mask(logical AttentionMask) AttentionMask {
	if h.applier == nil {
		return logical
	}
	return h.applier.ApplyMask(logical)
}

// AttentionMask describes an attention mask in four states:
//   - zero value: no mask.
//   - flag-form causal (causal=true only): dispatches to the MLX
//     kernel's mask_mode="causal" fast path.
//   - causal with relaxation rectangles: a causal mask with
//     bidirectional attention rectangles, such as for images.
//   - additive tensor (array!=nil): broadcast-compatible with
//     [B, 1, L, K]; contributed by a custom mask, helpers such as
//     QPaddingMask, KPaddingMask, or cache appliers and accumulated
//     via Intersect.
//
// The mask is a pure logical description — it carries no batch and
// exists independent of cache storage layout.
//
// All fields are comparable, so AttentionMask values compare with ==
// by full identity — SDPA uses this directly as a dispatch-cache key.
type AttentionMask struct {
	causal      bool
	relaxations *relaxNode
	array       *mlx.Array
}

type relaxRect struct {
	seq, qLo, qHi, kLo, kHi int
}

// relaxNode is a singly-linked list node holding relaxation
// rectangles. Each AttentionMask must have a fresh set of
// nodes to avoid false sharing between masks.
type relaxNode struct {
	rect relaxRect
	next *relaxNode
}

// CausalMask returns a flag-form causal mask. The mask stays
// tensor-free — hitting the kernel's mask_mode="causal" fast path —
// until something composes a relaxation, padding, or applier tensor
// onto it; then SDPA materializes via AsArray.
func CausalMask() AttentionMask {
	return AttentionMask{causal: true}
}

// ArrayMask wraps an explicit additive tensor broadcast-compatible
// with [B, 1, L, K].
func ArrayMask(a *mlx.Array) AttentionMask {
	return AttentionMask{array: a}
}

// IsZero reports whether the mask is the zero value (no mask at all).
func (m AttentionMask) IsZero() bool {
	return !m.causal && m.array == nil && m.relaxations == nil
}

// IsCausal reports whether the mask is pure flag-form causal — no
// relaxations and no accumulated array. SDPA dispatches to the
// kernel's "causal" fast path on this; any padding, applier
// contribution, or relaxation falls to the array path.
func (m AttentionMask) IsCausal() bool {
	return m.causal && m.relaxations == nil && m.array == nil
}

// Relax records a relaxation rectangle for batch sequence seq —
// positions (q, k) with q in [qLo, qHi) and k in [kLo, kHi) become
// freely attendable regardless of the causal base. Coordinates are
// absolute sequence positions on both axes, matching how causal is
// defined (k <= q). Multiple calls compose as a union per sequence.
//
// Rectangles that cannot change any cell — empty or already fully
// inside causal (kHi-1 <= qLo) — are dropped so IsCausal stays true
// and the mask remains on the kernel's fast path.
//
// Panics on pure ArrayMask (the caller owns the tensor and should
// modify it directly) or on the zero mask (nothing to relax).
func (m AttentionMask) Relax(seq, qLo, qHi, kLo, kHi int) AttentionMask {
	if !m.causal {
		if m.array != nil {
			panic("AttentionMask.Relax: cannot relax a pure ArrayMask; modify the tensor directly")
		}
		panic("AttentionMask.Relax: cannot relax a zero mask")
	}
	if qLo >= qHi || kLo >= kHi {
		return m
	}
	if kHi-1 <= qLo {
		return m
	}
	m.relaxations = &relaxNode{
		rect: relaxRect{seq: seq, qLo: qLo, qHi: qHi, kLo: kLo, kHi: kHi},
		next: m.relaxations,
	}
	return m
}

// Intersect returns the element-wise sum of this mask and other. Masks are
// additive and apply before softmax, so this is intersection
// semantics — a position is valid only if both sides have 0 there.
//
// At AsArray time a causal+Relax+array mask materializes as: causal
// writes -inf into the upper triangle, Relax overwrites its
// rectangles back to 0, then array is added on top — restricting 0
// cells further or no-op'ing on -inf cells.
func (m AttentionMask) Intersect(other AttentionMask) AttentionMask {
	if m.IsZero() {
		return other
	}
	if other.IsZero() {
		return m
	}

	result := AttentionMask{
		causal: m.causal || other.causal,
	}

	// Relax requires causal, so relaxations != nil implies causal.
	switch {
	case m.relaxations != nil && other.relaxations != nil:
		// Both sides causal+Relax: pairwise rect intersection per sequence.
		var list *relaxNode
		for a := m.relaxations; a != nil; a = a.next {
			for b := other.relaxations; b != nil; b = b.next {
				if a.rect.seq != b.rect.seq {
					continue
				}
				qLo := max(a.rect.qLo, b.rect.qLo)
				qHi := min(a.rect.qHi, b.rect.qHi)
				kLo := max(a.rect.kLo, b.rect.kLo)
				kHi := min(a.rect.kHi, b.rect.kHi)
				if qHi <= qLo || kHi <= kLo || kHi-1 <= qLo {
					continue
				}
				list = &relaxNode{
					rect: relaxRect{seq: a.rect.seq, qLo: qLo, qHi: qHi, kLo: kLo, kHi: kHi},
					next: list,
				}
			}
		}
		result.relaxations = list
	case m.relaxations != nil && !other.causal:
		result.relaxations = m.relaxations
	case other.relaxations != nil && !m.causal:
		result.relaxations = other.relaxations
	default:
		// Implicit: one side causal+Relax, the other plain causal
		// (no relaxations). Plain causal blocks every cell Relax
		// tried to release, so intersection with its empty release
		// set leaves nothing — result.relaxations stays nil and
		// collapses to pure causal.
	}

	switch {
	case m.array != nil && other.array != nil:
		result.array = mlx.Add(m.array, other.array)
	case m.array != nil:
		result.array = m.array
	case other.array != nil:
		result.array = other.array
	}

	return result
}

// AsArray materializes the mask as a [B, 1, L, K] additive tensor
// (0 where valid, -inf where blocked). B and L come from b; K and
// dtype come from the caller.
//
// Composition order:
//  1. Start from zero.
//  2. If m.causal: -inf where oldestPos+k > SeqOffsets[b] + q per row.
//  3. Apply m.relaxations (qLo/qHi and kLo/kHi are absolute positions).
//  4. Add m.array if present.
func (m AttentionMask) AsArray(b *batch.Batch, K int, dtype mlx.DType) *mlx.Array {
	// Pure ArrayMask: caller owns the tensor, nothing to compose.
	if !m.causal && m.relaxations == nil && m.array != nil {
		if m.array.DType() == dtype {
			return m.array
		}
		return m.array.AsType(dtype)
	}

	B := len(b.SeqOffsets)
	L := b.InputIDs.Dim(1)

	negInf := float32(math.Inf(-1))
	vals := make([]float32, B*L*K)
	if m.causal {
		for i := range B {
			off := int(b.SeqOffsets[i])
			oldestPos := max(0, off+L-K)
			base := i * L * K
			for q := range L {
				absQ := off + q
				row := base + q*K
				for k := range K {
					if oldestPos+k > absQ {
						vals[row+k] = negInf
					}
				}
			}
		}
	}

	for n := m.relaxations; n != nil; n = n.next {
		r := n.rect
		if r.seq < 0 || r.seq >= B {
			continue
		}
		off := int(b.SeqOffsets[r.seq])
		oldestPos := max(0, off+L-K)
		qLo := min(max(r.qLo-off, 0), L)
		qHi := min(max(r.qHi-off, 0), L)
		kLo := min(max(r.kLo-oldestPos, 0), K)
		kHi := min(max(r.kHi-oldestPos, 0), K)
		base := r.seq * L * K
		for q := qLo; q < qHi; q++ {
			row := base + q*K
			for k := kLo; k < kHi; k++ {
				vals[row+k] = 0
			}
		}
	}

	out := mlx.FromValues(vals, B, 1, L, K)
	if m.array != nil {
		out = mlx.Add(out, m.array)
	}
	if dtype != mlx.DTypeFloat32 {
		out = out.AsType(dtype)
	}
	return out
}

// QPaddingMask returns an additive [B, 1, L, 1] mask that blocks
// padded query rows (q >= b.SeqQueryLens[i]) across all keys. It is
// logical — independent of whatever layout the cache uses for K.
// Returns the zero mask when every row is full.
func QPaddingMask(b *batch.Batch, dtype mlx.DType) AttentionMask {
	return padTailMask(len(b.SeqOffsets), b.InputIDs.Dim(1), 2, b.SeqQueryLens, dtype)
}

// KPaddingMask returns an additive [B, 1, 1, K] mask that blocks
// padded key columns (k >= kLens[i]) across all queries. Storage-
// dependent: kLens describes where real content ends in physical K,
// so this is typically used without a cache where the caller knows
// the actual layout. Returns the zero mask when every row is full.
func KPaddingMask(b *batch.Batch, K int, kLens []int32, dtype mlx.DType) AttentionMask {
	return padTailMask(len(b.SeqOffsets), K, 3, kLens, dtype)
}

// SlidingWindowMask returns an additive [B, 1, L, K] mask blocking
// keys outside a per-row window of size `window`: any key whose
// absolute position p < absQ - window + 1 is blocked. Returns the
// zero mask when window <= 0 or no row needs blocking.
//
// Defined in logical position space — the K axis is position-ordered
// with column 0 at oldestPos = max(0, b.SeqOffsets[i]+L-K).
func SlidingWindowMask(b *batch.Batch, K, window int, dtype mlx.DType) AttentionMask {
	if window <= 0 {
		return AttentionMask{}
	}
	B := len(b.SeqOffsets)
	L := b.InputIDs.Dim(1)
	negInf := float32(math.Inf(-1))
	vals := make([]float32, B*L*K)
	needed := false
	for i := range B {
		off := int(b.SeqOffsets[i])
		oldestPos := max(0, off+L-K)
		base := i * L * K
		for q := range L {
			absQ := off + q
			lo := absQ - window + 1
			maskCount := lo - oldestPos
			if maskCount <= 0 {
				continue
			}
			if maskCount > K {
				maskCount = K
			}
			row := base + q*K
			for k := range maskCount {
				vals[row+k] = negInf
				needed = true
			}
		}
	}
	if !needed {
		return AttentionMask{}
	}
	out := mlx.FromValues(vals, B, 1, L, K)
	if dtype != mlx.DTypeFloat32 {
		out = out.AsType(dtype)
	}
	return ArrayMask(out)
}

func padTailMask(B, total, axis int, lens []int32, dtype mlx.DType) AttentionMask {
	needed := false
	for i := range B {
		if int(lens[i]) < total {
			needed = true
			break
		}
	}
	if !needed {
		return AttentionMask{}
	}

	negInf := float32(math.Inf(-1))
	vals := make([]float32, B*total)
	for i := range B {
		n := int(lens[i])
		base := i * total
		for j := n; j < total; j++ {
			vals[base+j] = negInf
		}
	}
	shape := [4]int{B, 1, 1, 1}
	shape[axis] = total
	out := mlx.FromValues(vals, shape[0], shape[1], shape[2], shape[3])
	if dtype != mlx.DTypeFloat32 {
		out = out.AsType(dtype)
	}
	return ArrayMask(out)
}
