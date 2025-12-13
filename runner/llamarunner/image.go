package llamarunner

import (
	"errors"
	"fmt"
	"hash/maphash"
	"log/slog"
	"sync"
	"time"

	"github.com/ollama/ollama/llama"
)

const imageCacheSize = 4

type ImageContext struct {
	// mu is required to be held when generating embeddings or accessing the cache
	mu sync.Mutex

	mtmd *llama.MtmdContext

	// cache of images to embeddings
	images    []imageCache
	imageHash maphash.Hash
}

func NewImageContext(llamaContext *llama.Context, modelPath string) (*ImageContext, error) {
	arch, err := llama.GetModelArch(modelPath)
	if err != nil {
		return nil, fmt.Errorf("unable to determine vision architecture: %w (%s)", err, modelPath)
	}

	var c ImageContext
	if arch == "clip" {
		c.mtmd, err = llama.NewMtmdContext(llamaContext, modelPath)
	} else {
		return nil, fmt.Errorf("unknown vision model architecture: %s", arch)
	}

	if err != nil {
		return nil, err
	}

	c.images = make([]imageCache, imageCacheSize)

	return &c, nil
}

func (c *ImageContext) Free(modelPath string) {
	if c == nil {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clear image cache to prevent stale data
	c.ClearCacheUnsafe()

	if c.mtmd != nil {
		c.mtmd.Free()
		c.mtmd = nil
	}
}

// ClearCache clears all cached image embeddings (thread-safe)
func (c *ImageContext) ClearCache() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ClearCacheUnsafe()
}

// ClearCacheUnsafe clears cache without locking (caller must hold mu)
func (c *ImageContext) ClearCacheUnsafe() {
	for i := range c.images {
		c.images[i].key = 0
		c.images[i].val = nil
		c.images[i].lastUsed = time.Time{}
	}
}

func (c *ImageContext) MultimodalTokenize(llamaContext *llama.Context, data []byte) ([]llama.MtmdChunk, error) {
	if c == nil {
		return nil, nil
	}

	if len(data) <= 0 {
		return nil, errors.New("received zero length image")
	}

	hash := c.hashImage(data)

	c.mu.Lock()
	defer c.mu.Unlock()

	// Try to find cached embeddings first
	chunks, err := c.findImage(hash)
	if err != nil {
		if c.mtmd != nil {
			chunks, err = c.mtmd.MultimodalTokenize(llamaContext, data)
			if err != nil {
				return nil, err
			}
		} else {
			return nil, errors.New("received image but vision model not loaded")
		}

		c.addImage(hash, chunks)
	}

	return chunks, nil
}

func (c *ImageContext) BatchSize(configuredBatchSize int) int {
	// If images are not supported, we don't need to allocate embedding batches
	if c == nil {
		return 0
	}

	// For M-RoPE models (Qwen2-VL, Qwen3-VL), we need a larger batch to fit
	// entire images. The max image size is typically 2048x2048 pixels, which
	// with patch_size=16 and merge=2 gives (2048/16/2)^2 = 4096 tokens for
	// a square image. Non-square images can be larger (e.g., 53x76 = 4028).
	// Use 8192 to be safe for large images.
	if c.UsesMRoPE() {
		const mropeBatchSize = 8192
		if configuredBatchSize < mropeBatchSize {
			slog.Debug("M-RoPE batch size increased for large images", "configured", configuredBatchSize, "actual", mropeBatchSize)
			return mropeBatchSize
		}
	}

	return configuredBatchSize
}

func (c *ImageContext) EmbedSize(llamaContext *llama.Context) int {
	// For multimodal models, use NEmbdInp() which returns the vision projector
	// embedding dimension (e.g., 8192 for qwen3vl) instead of NEmbd() which
	// returns the text model dimension (e.g., 2048 for qwen3vl)
	return llamaContext.Model().NEmbdInp()
}

// UsesMRoPE returns true if the vision model requires M-RoPE (Qwen2-VL, Qwen3-VL)
func (c *ImageContext) UsesMRoPE() bool {
	if c == nil || c.mtmd == nil {
		return false
	}
	return c.mtmd.UsesMRoPE()
}

type imageCache struct {
	key      uint64
	val      []llama.MtmdChunk
	lastUsed time.Time
}

func (c *ImageContext) hashImage(image []byte) uint64 {
	c.imageHash.Reset()
	_, _ = c.imageHash.Write(image)
	return c.imageHash.Sum64()
}

var errImageNotFound = errors.New("image not found in cache")

func (c *ImageContext) findImage(hash uint64) ([]llama.MtmdChunk, error) {
	for i := range c.images {
		if c.images[i].key == hash {
			slog.Debug("image cache HIT", "entry", i, "hash", hash)
			c.images[i].lastUsed = time.Now()
			return c.images[i].val, nil
		}
	}

	return nil, errImageNotFound
}

func (c *ImageContext) addImage(hash uint64, embed []llama.MtmdChunk) {
	best := time.Now()
	var bestImage int

	for i := range c.images {
		if c.images[i].key == hash {
			bestImage = i
			break
		}

		if c.images[i].lastUsed.Compare(best) < 0 {
			best = c.images[i].lastUsed
			bestImage = i
		}
	}

	slog.Debug("image cache MISS - encoding and storing", "entry", bestImage, "hash", hash)
	c.images[bestImage].key = hash
	c.images[bestImage].val = embed
	c.images[bestImage].lastUsed = time.Now()
}
