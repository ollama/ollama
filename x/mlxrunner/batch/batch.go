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

	// Memo is per-forward memoization used to cache results, such as masks,
	// which are often the same across layers.
	Memo Memo
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
