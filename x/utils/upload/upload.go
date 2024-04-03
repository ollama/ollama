package upload

import (
	"iter"

	"golang.org/x/exp/constraints"
)

type Chunk[I constraints.Integer] struct {
	Offset I
	N      I
}

// Chunks yields a sequence of a part number and a Chunk. The Chunk is the offset
// and size of the chunk. The last chunk may be smaller than chunkSize if size is
// not a multiple of chunkSize.
//
// The first part number is 1 and increases monotonically.
func Chunks[I constraints.Integer](size, chunkSize I) iter.Seq2[int, Chunk[I]] {
	return func(yield func(int, Chunk[I]) bool) {
		var n int
		for off := I(0); off < size; off += chunkSize {
			n++
			if !yield(n, Chunk[I]{off, min(chunkSize, size-off)}) {
				return
			}
		}
	}
}
