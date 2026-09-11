package create

import (
	"encoding/json"
	"strings"
)

type qwen4ExpImportTransform struct{}

func newQwen4ExpImportTransform(_ json.RawMessage) (quantizePolicy, error) {
	return qwen4ExpImportTransform{}, nil
}

func (qwen4ExpImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	// PLE is a hashed n-gram feature table rather than the model's token
	// embedding. It dominates the checkpoint size, and its lookup path supports
	// row-wise dequantization, so quantize its shards at the requested type.
	if qwen4ExpIsPLEEmbeddingShard(name) {
		quantize = normalizeQuantType(quantize)
		if len(shape) == 2 && isAligned(shape, quantize) {
			return quantize
		}
		return ""
	}

	// Match the conservative Qwen linear-attention policy: these low-rank
	// projections are sensitive and negligible in size.
	if qwen35IsLowRankProjection(name) {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func qwen4ExpIsPLEEmbeddingShard(name string) bool {
	return strings.Contains(name, ".ple.ple_embedding.ngram_embedding.shard_") &&
		strings.HasSuffix(name, ".weight")
}
