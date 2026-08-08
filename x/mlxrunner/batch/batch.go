package batch

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// Batch is the per-forward-pass input handed to a model.
type Batch struct {
	// InputIDs is the input token IDs for this forward pass, shape (B, L).
	InputIDs *mlx.Array

	// SeqOffsets gives each row's current position within its sequence —
	// where the chunk in InputIDs starts. Length equals the batch dimension
	// of InputIDs.
	SeqOffsets []int32

	// SeqQueryLens is each row's real query length in this forward. Values
	// less than L mean the row's tail is padding that must be masked out.
	// Length equals the batch dimension of InputIDs.
	SeqQueryLens []int32

	// Hidden is the draft-conditioning state for a draft model's forward.
	// It is nil for ordinary forward passes.
	Hidden *mlx.Array

	// Media lists a row's media items, on prefill forwards and draft
	// forwards that embed prompt tokens; items outside the query range
	// ride featureless. Nil at decode and for text-only requests.
	Media []MediaItem

	// Layout carries each row's opaque layout state from PrepareMedia,
	// identical on every forward of the request; the runner never reads
	// it. Nil entries derive nothing from layout.
	Layout []any

	// Memo is per-forward memoization used to cache results, such as masks,
	// which are often the same across layers.
	Memo Memo
}

// MediaItem is one media occurrence in a row's sequence. The runner
// knows only where the expansion was spliced; which positions bear
// features is the model's, derived from Pos and Opaque.
type MediaItem struct {
	// Seq is the batch row the item belongs to.
	Seq int

	// Pos is the absolute sequence position of the expansion's first token.
	Pos int

	// Features is the item's whole feature-row array, attached only while
	// the item's token range overlaps this forward's query range.
	Features *mlx.Array

	// Opaque is the item's PreparedMedia.Opaque, round-tripped untouched.
	Opaque any
}

type Memo struct {
	entries map[any]any
}

// Get returns the memoized value for key and true if present, or nil
// and false otherwise.
func (m *Memo) Get(key any) (any, bool) {
	v, ok := m.entries[key]
	return v, ok
}

// Put stores value under key, allocating on first use.
func (m *Memo) Put(key, value any) {
	if m.entries == nil {
		m.entries = map[any]any{}
	}
	m.entries[key] = value
}
