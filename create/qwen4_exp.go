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
	quantize = normalizeQuantType(quantize)

	// PLE is a hashed n-gram feature table rather than the model's token
	// embedding. Its lookup path dequantizes selected rows, so quantize its
	// shards at the requested type.
	if qwen4ExpIsPLEEmbeddingShard(name) {
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

	if quantize == "nvfp4" {
		// Keep QSA Q/K/V and the indexer projection on the validated BF16 path.
		// Keep the whole draft-side control path at source precision as well.
		if qwen4ExpIsSensitiveQSA(name) ||
			(strings.HasPrefix(name, "mtp.") && !qwen4ExpIsRoutedExpert(name)) {
			return ""
		}
		if isEmbedTokensWeight(name) {
			return promoteEmbedding(shape, quantize)
		}

		// Routed experts dominate the remaining model size and have validated NVFP4
		// kernels. Use MXFP8 for the other eligible target weights to avoid the
		// long-context quality loss seen with blanket NVFP4.
		if !qwen4ExpIsRoutedExpert(name) {
			return GetTensorQuantization(name, shape, "mxfp8")
		}
	}
	return GetTensorQuantization(name, shape, quantize)
}

func qwen4ExpIsPLEEmbeddingShard(name string) bool {
	return strings.Contains(name, ".ple.ple_embedding.ngram_embedding.shard_") &&
		strings.HasSuffix(name, ".weight")
}

func qwen4ExpIsRoutedExpert(name string) bool {
	return (strings.HasPrefix(name, "model.language_model.layers.") ||
		strings.HasPrefix(name, "mtp.layers.")) &&
		strings.Contains(name, ".mlp.experts.")
}

func qwen4ExpIsSensitiveQSA(name string) bool {
	return strings.HasPrefix(name, "model.language_model.layers.") &&
		strings.Contains(name, ".self_attn.") &&
		(strings.HasSuffix(name, ".q_proj.weight") ||
			strings.HasSuffix(name, ".k_proj.weight") ||
			strings.HasSuffix(name, ".v_proj.weight") ||
			strings.HasSuffix(name, ".indexer.index_qk_proj.weight"))
}
