package llm

import (
	"log/slog"
)

// Interface for GGML functionality needed by validation
type GGMLModel interface {
	KV() KV
}

// kvCacheBytesPerElement returns the number of bytes per element for a given KV cache type
func kvCacheBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "", "f16", "fp16":
		return 2 // fp16 is the default
	case "q8_0":
		return 1 // 1/2 of fp16
	case "q4_0":
		return 0.5 // 1/4 of fp16
	default:
		// Only log a warning if an explicit type was provided but not recognized
		if cacheType != "" {
			slog.Warn("unknown cache type, defaulting to fp16", "type", cacheType)
		}
		return 2
	}
}

// estimateKvCacheSize determines the total memory required for KV cache based on the quantization type
func estimateKvCacheSize(cacheType string, numCtx uint64, ggml *GGML) uint64 {
	bytesPerElement := kvCacheBytesPerElement(cacheType)

	kv := ggml.KV()
	estimate := uint64(float64(numCtx*kv.BlockCount()*(kv.EmbeddingHeadCountK()+kv.EmbeddingHeadCountV())*kv.HeadCountKV()) * bytesPerElement)
	// round up to the nearest multiple of 64 bytes
	return ((estimate + 63) / 64) * 64
}
