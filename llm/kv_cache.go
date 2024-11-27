package llm

import (
	"log/slog"
	"slices"
)

// Interface for GGML functionality needed by validation
type GGMLModel interface {
	KV() KV
}

// kVCacheQuantization determines the appropriate KV cache quantization type based on
// model characteristics, GPU capabilities, and environment settings
func kVCacheQuantization(kvct string, ggml GGMLModel) string {
	if kvct == "f16" || kvct == "f32" {
		return kvct
	}

	// First check if the requested type is valid
	validKVCacheTypes := []string{"f32", "f16", "q8_0", "q4_0"}
	if !slices.Contains(validKVCacheTypes, kvct) {
		slog.Warn("unsupported cache type, defaulting to f16", "type", kvct)
		return "f16"
	}

	// Check flash attention support
	flashAttnEnabled, _ := checkFlashAttentionSupport(ggml)
	if !flashAttnEnabled && kvct != "f16" && kvct != "f32" {
		slog.Info("quantized KV cache requires flash attention, defaulting to f16",
			"requested_type", kvct)
		return "f16"
	}

	// Flash attention is enabled and type is valid, use requested type
	return kvct
}

// estimateKvCacheSize determines the total memory required for KV cache based on the quantization type
func estimateKvCacheSize(cacheType string, numCtx uint64, ggml *GGML) uint64 {
	var bytesPerElement float64

	flashAttnEnabled, _ := checkFlashAttentionSupport(ggml)
	if !flashAttnEnabled && cacheType != "f32" {
		cacheType = "f16"
	}

	switch cacheType {
	case "f32", "fp32":
		bytesPerElement = 4 // fp32
	case "", "f16", "fp16":
		bytesPerElement = 2 // fp16 is the default
	case "q8_0":
		bytesPerElement = 1 // 1/2 of fp16
	case "q4_0":
		bytesPerElement = 0.5 // 1/4 of fp16
	default:
		// Only log a warning if an explicit type was provided but not recognized
		if cacheType != "" {
			slog.Warn("unknown cache type, defaulting to fp16", "type", cacheType)
		}
		bytesPerElement = 2
	}

	kv := ggml.KV()
	estimate := uint64(float64(numCtx*kv.BlockCount()*(kv.EmbeddingHeadCountK()+kv.EmbeddingHeadCountV())*kv.HeadCountKV()) * bytesPerElement)
	// round up to the nearest multiple of 64 bytes
	return ((estimate + 63) / 64) * 64
}
