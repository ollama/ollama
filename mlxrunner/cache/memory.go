package cache

import "math"

// EstimateMemory budgets one request's live cache, copies and checkpoints using
// tensor shapes from a fresh cache's first forward. It does not evaluate arrays
// or allocate storage.
// Unknown or uninitialized caches cannot safely increase automatic parallelism.
func EstimateMemory(c Cache, tokens, checkpoints int) (int64, bool) {
	if c == nil {
		return 0, true // Some layers share another layer's cache.
	}
	if tokens <= 0 || checkpoints < 0 {
		return 0, false
	}
	var bytes int64
	for _, a := range c.State() {
		if a != nil {
			bytes += int64(a.NumBytes())
		}
	}
	if bytes <= 0 || c.Offset() <= 0 {
		return 0, false
	}
	var count int64
	switch c := c.(type) {
	case *KVCache:
		bytes /= int64(c.Offset())
		// Full-attention snapshots share the live prefix until it is copied.
		count = 2 * ((int64(tokens) + int64(c.step) - 1) / int64(c.step) * int64(c.step))
	case *RotatingKVCache:
		bytes /= int64(min(c.Offset(), c.maxSize))
		count = int64(min(tokens, c.maxSize)) * (2 + int64(checkpoints))
	case *RecurrentCache:
		// Bounded state is retained at each prompt checkpoint, plus live state
		// and a copy when the next request resumes or rewinds it.
		count = 2 + int64(checkpoints)
	default:
		return 0, false
	}
	if count <= 0 || bytes > math.MaxInt64/count {
		return 0, false
	}
	return bytes * count, true
}
