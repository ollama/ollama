package base

import (
	// Every model's PrepareMedia decodes through image.Decode; the decoder
	// set is registered once here so all models accept the same formats.
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Segment is one run of the prompt in stream order: either a tokenized text
// run (Tokens set) or a single media item (Kind and Data set).
type Segment struct {
	Tokens []int32
	Kind   string
	Data   []byte
}

// PreparedItem describes one media occurrence in the prepared stream. A
// model chooses item granularity: one per media segment, or several when
// parts encode and evaluate independently (e.g. per tile).
type PreparedItem struct {
	// Range is the expansion's token range [start, end) in Tokens;
	// non-empty, since cache identity enters through these positions.
	Range [2]int

	// Source is the index of the segment this item was prepared from; the
	// item's prefix-cache identity is keyed on that segment's bytes.
	Source int

	// MediaData is the preprocessed encoder input with shape Dims. Dims
	// enters the cache keys too: geometry changes features under
	// identical bytes.
	MediaData []float32
	Dims      []int

	// Opaque carries model-private preprocessing state to EncodeMedia
	// and Forward.
	Opaque any

	// Causal marks an expansion whose tokens attend causally, so chunks
	// may split it. Unset, the first evaluation covers the whole
	// expansion in one forward, as bidirectional runs require.
	Causal bool
}

// PreparedRequest is the expanded input stream, every media segment's
// expansion spliced in place, with the items in stream order.
type PreparedRequest struct {
	Tokens []int32
	Items  []PreparedItem

	// Layout is an opaque request-scoped value computed in the one pass
	// that sees every splice position; immutable, carried unread by the
	// runner to every forward. Delivered only when Items is non-empty.
	// Nil when the model derives nothing from it.
	Layout any
}

// MediaModel is implemented by models that accept media inputs.
type MediaModel interface {
	// PrepareMedia runs once per request on the request goroutine, CPU
	// only, and returns the expanded stream. It must be deterministic for
	// given segments: prefix-cache restores splice cached state with
	// recomputed state.
	PrepareMedia(segments []Segment) (*PreparedRequest, error)

	// EncodeMedia builds one item's lazy feature graph on the MLX thread;
	// it must not evaluate — the consuming forward's evaluation pulls it.
	// Read the pixels from data: the runner frees the item's MediaData
	// once its expansion is evaluated.
	EncodeMedia(item *PreparedItem, data *mlx.Array) *mlx.Array
}
