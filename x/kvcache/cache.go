package kvcache

import (
	"errors"

	"github.com/ollama/ollama/x/ml"
	"github.com/ollama/ollama/x/model/input"
)

var (
	ErrKvCacheFull  = errors.New("could not find a kv cache slot")
	ErrNotSupported = errors.New("model does not support operation")
)

type Cache interface {
	// ** used by model implementations **

	// SetLayer sets the active layer of the cache
	SetLayer(layer int)

	// Get returns the history of key and value tensors plus a mask
	//
	// The shape of the tensors is documented in the specific
	// cache implementation used.
	Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor)

	// Put stores a batch of key and value in the cache
	//
	// The shape of the tensors is documented in the specific
	// cache implementation used.
	Put(ctx ml.Context, key, value ml.Tensor)

	// SetConfig controls optimizations (mostly backend-specific) that may transform
	// the output of the cache to work better with specific kernels. If not called,
	// the backend settings will be used. This works well when calling Attention.
	//
	// The config can be overridden by models, especially if they require vanilla
	// output when implementing their own version of attention. To do this, pass
	// an empty ml.CacheConfig.
	//
	// Most models will not need to use this.
	SetConfig(ml.CacheConfig)

	// ** cache management **

	// Init sets up runtime parameters.
	// backend: Used to allocate cache data storage and execute management operations (such as defrag)
	// dtype: The data type for storing cache entries
	// maxSequences: The maximum number of sequences stored in the cache - across all batches
	// capacity: The number of cache entries to store, per sequence
	// maxBatch: The maximum number of tokens that can occur in a single batch
	Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int)

	// Close closes the cache and frees resources associated with it
	Close()

	// StartForward is called before the start of the model's forward pass.
	// For each token in the coming batch, there must be a corresponding
	// entry in positions and seqs. reserve is to preallocate memory
	// without actually storing data in the cache.
	StartForward(ctx ml.Context, batch input.Batch, reserve bool) error

	// CopyPrefix copies tokens in the range [0, len) from srcSeq to dstSeq
	CopyPrefix(srcSeq, dstSeq int, len int32)

	// CanResume returns true if the cache can continue with the next token at
	// the given position and sequence. Assumes that the caller has already
	// verified the contents of the cache.
	CanResume(seq int, pos int32) bool

	// Remove deletes tokens in the range [beginIndex, endIndex) from seq. Set
	// endIndex to math.MaxInt32 to remove everything starting at beginIndex.
	//
	// If an error occurs, the entire context for the sequence should be
	// removed by calling Remove(seq, 0, math.MaxInt32)
	Remove(seq int, beginIndex, endIndex int32) error
}
