package base

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Segment is one run of the prompt in stream order: either a tokenized text
// run (Tokens set) or a single media item (Kind and Data set).
type Segment struct {
	Tokens []int32
	Kind   string
	Data   []byte
}

// PreparedItem describes one media occurrence in the prepared stream: where
// its expansion sits, the preprocessed encoder input, and the model-private
// state that travels with it. A model chooses the item granularity — one
// item per media segment, or several (e.g. one per tile) when parts of a
// segment can be encoded and evaluated independently.
type PreparedItem struct {
	// Range is the expansion's token range [start, end) in
	// PreparedRequest.Tokens. It must be non-empty: media identity enters
	// the prefix-cache keys through the positions the expansion occupies.
	Range [2]int

	// Source is the index of the segment this item was prepared from; the
	// item's prefix-cache identity is keyed on that segment's bytes.
	Source int

	// MediaData is the preprocessed encoder input, uploaded by the runner
	// with shape Dims. Dims also enters the prefix-cache keys — it pins the
	// preprocessing geometry, which changes features under identical bytes.
	MediaData []float32
	Dims      []int

	// Opaque carries model-private preprocessing state (geometry, grids)
	// to EncodeMedia and, via batch.MediaItem, to Forward.
	Opaque any

	// Causal marks an expansion whose feature tokens attend causally, so a
	// chunk may end inside it. Unset, the first evaluation covers the whole
	// expansion in one forward: a bidirectional run's early rows attend its
	// later keys.
	Causal bool
}

// PreparedRequest is the CPU-phase product of preparing a prompt's media:
// the full input stream with every media segment's expansion spliced in
// place, and the items in stream order.
type PreparedRequest struct {
	Tokens []int32
	Items  []PreparedItem

	// Layout is an opaque request-scoped value derived from the media
	// layout (position tables, mask precursors), computed here because
	// this is the one pass that sees every splice position. Immutable; the
	// runner carries it unread to every forward via Batch.Layout. Nil when
	// the model derives nothing from layout.
	Layout any
}

// MediaModel is implemented by models that accept media inputs.
type MediaModel interface {
	// PrepareMedia runs once per request on the request goroutine with the
	// prompt's segments in stream order: CPU only, no mlx. It returns the
	// expanded stream, splicing each media segment's placeholder expansion
	// in place of the segment. It must be deterministic for given segments
	// within a process — prefix-cache restores splice cached state with
	// recomputed state under keys derived from the source bytes.
	PrepareMedia(segments []Segment) (*PreparedRequest, error)

	// EncodeMedia builds the lazy feature graph for one prepared item from
	// its uploaded MediaData. Runs on the MLX thread and must not evaluate —
	// the consuming forward's evaluation pulls the encoder.
	EncodeMedia(item *PreparedItem, data *mlx.Array) *mlx.Array
}
