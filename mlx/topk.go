package mlx

// topKChunk fits in one block of MLX's GPU sort; longer rows take a
// multi-pass merge sort.
const topKChunk = 2048

// TopKIndices returns the int32 indices of the k largest values along the last
// axis, in argpartition's order. Argpartition sorts the whole row, so long rows
// are sorted in chunks and only each chunk's top k is sorted again.
func TopKIndices(a *Array, k int) *Array {
	dims := a.Dims()
	n := dims[len(dims)-1]
	if k >= topKChunk || n <= topKChunk {
		return topKSorted(a, k)
	}

	rows := a.Size() / n
	x := a.Reshape(rows, n)
	chunks := n / topKChunk
	split := chunks * topKChunk

	local := topKSorted(x.Slice(Slice(), Slice(0, split)).Reshape(rows, chunks, topKChunk), k)
	offsets := Arange(0, float64(split), topKChunk, DTypeInt32).Reshape(1, chunks, 1)
	candidates := local.Add(offsets).Reshape(rows, chunks*k)
	if split < n {
		tail := Arange(float64(split), float64(n), 1, DTypeInt32).Reshape(1, n-split)
		candidates = Concatenate([]*Array{candidates, BroadcastTo(tail, int32(rows), int32(n-split))}, -1)
	}

	best := topKSorted(x.TakeAlongAxis(candidates, -1), k)
	return candidates.TakeAlongAxis(best, -1).Reshape(append(dims[:len(dims)-1:len(dims)-1], k)...)
}

// topKSorted is TopKIndices in a single sort.
func topKSorted(a *Array, k int) *Array {
	slices := make([]slice, a.NumDims())
	for i := range slices {
		slices[i] = Slice()
	}
	slices[len(slices)-1] = Slice(0, k)
	return a.Negative().ArgpartitionAxis(k-1, -1).Slice(slices...).AsType(DTypeInt32)
}
