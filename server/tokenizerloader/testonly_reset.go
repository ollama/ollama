//go:build test

package tokenizerloader

// ResetForTest clears any package-level cache/state so tests start clean.
func ResetForTest() {
	initCacheForTest()
}

// If your loader already has an initCache or similar, call it here.
// Otherwise implement a tiny function in loader.go (non-test) that reinitializes the LRU.
