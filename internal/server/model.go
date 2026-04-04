package model

import "math"

type Model struct {
	// ... other fields ...
	KVHeads      int // number of key/value heads
	HeadDim      int // dimension of each head
	// ...
}

// ComputeKVCacheSize returns the memory (in bytes) required for the KV cache
// for a given context length using the model's head dimension and number of KV heads.
// The calculation assumes fp16 storage (2 bytes per element).
// Previously the code used 4 bytes per element (fp32), causing ~2× memory usage.
func (m *Model) ComputeKVCacheSize(contextLength int) int64 {
	// Fixed: use 2 bytes per element (fp16) instead of 4 bytes (fp32)
	elements := contextLength * m.KVHeads
	return int64(elements * m.HeadDim * 2)
}