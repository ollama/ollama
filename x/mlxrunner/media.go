package mlxrunner

import (
	"encoding/binary"
	"hash/fnv"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// mediaItem is one media occurrence in a request's token stream.
type mediaItem struct {
	pos    int
	length int
	fold   uint32
	item   *base.PreparedItem
}

// foldValue derives an item's trie-key substitute: a hash of the raw
// bytes and preprocessing dims, with bit 31 forced so it can never equal
// a token ID.
func foldValue(data []byte, dims []int) uint32 {
	h := fnv.New64a()
	h.Write(data)
	var b [8]byte
	for _, d := range dims {
		binary.LittleEndian.PutUint64(b[:], uint64(d))
		h.Write(b[:])
	}
	sum := h.Sum64()
	return (uint32(sum>>32) ^ uint32(sum)) | 1<<31
}
