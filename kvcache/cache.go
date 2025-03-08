package kvcache

import (
	"errors"

	"github.com/ollama/ollama/ml"
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

	// Init sets up runtime parameters
	Init(backend ml.Backend, dtype ml.DType, capacity int32)

	// Close closes the cache and frees resources associated with it
	Close()

	// StartForward is called before the start of the model's forward pass.
	// For each token in the coming batch, there must be a corresponding
	// entry in positions and seqs.
	StartForward(ctx ml.Context, positions []int32, seqs []int) error

	// CopyPrefix copies tokens in the range [0, len) from srcSeq to dstSeq
	CopyPrefix(srcSeq, dstSeq int, len int32)

	// Remove deletes tokens in the range [beginIndex, endIndex) from seq. Set
	// endIndex to math.MaxInt32 to remove everything starting at beginIndex.
	//
	// If an error occurs, the entire context for the sequence should be
	// removed by calling Remove(seq, 0, math.MaxInt32)
	Remove(seq int, beginIndex, endIndex int32) error
}
