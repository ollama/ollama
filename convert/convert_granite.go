package convert

import (
	"cmp"
	"math"

	"github.com/ollama/ollama/llm"
)

type graniteModel struct {
	llamaModel

	ResidualScale  float32 `json:"residual_multiplier"`
	EmbeddingScale float32 `json:"embedding_multiplier"`
	AttentionScale float32 `json:"attention_multiplier"`
	LogitsScale    float32 `json:"logits_scaling"`
}

var _ ModelConverter = (*graniteModel)(nil)

func (p *graniteModel) KV(t *Tokenizer) llm.KV {
	kv := p.ModelParameters.KV(t)
	kv["general.architecture"] = "granite"
	kv["granite.vocab_size"] = p.VocabSize

	kv["granite.block_count"] = cmp.Or(p.NLayers, p.NumHiddenLayers, p.NLayer)

	if contextLength := cmp.Or(p.MaxPositionEmbeddings, p.NCtx); contextLength > 0 {
		kv["granite.context_length"] = contextLength
	}

	if embeddingLength := cmp.Or(p.HiddenSize, p.NEmbd); embeddingLength > 0 {
		kv["granite.embedding_length"] = cmp.Or(p.HiddenSize, p.NEmbd)
	}

	if feedForwardLength := cmp.Or(p.IntermediateSize, p.NInner); feedForwardLength > 0 {
		kv["granite.feed_forward_length"] = cmp.Or(p.IntermediateSize, p.NInner)
	}

	if headCount := cmp.Or(p.NumAttentionHeads, p.NHead); headCount > 0 {
		kv["granite.attention.head_count"] = cmp.Or(p.NumAttentionHeads, p.NHead)
		kv["granite.rope.dimension_count"] = p.HiddenSize / headCount
	}

	if p.RopeTheta > 0 {
		kv["granite.rope.freq_base"] = p.RopeTheta
	}

	if p.RopeScaling.Type == "linear" {
		kv["granite.rope.scaling.type"] = p.RopeScaling.Type
		kv["granite.rope.scaling.factor"] = p.RopeScaling.Factor
	} else if p.RopeScaling.RopeType == "llama3" {
		dim := p.HiddenSize / p.NumAttentionHeads
		for i := uint32(0); i < dim; i += 2 {
			factor := cmp.Or(p.RopeScaling.Factor, 8.0)
			factorLow := cmp.Or(p.RopeScaling.LowFrequencyFactor, 1.0)
			factorHigh := cmp.Or(p.RopeScaling.HighFrequencyFactor, 4.0)

			original := cmp.Or(p.RopeScaling.OriginalMaxPositionalEmbeddings, 8192)
			lambdaLow := float32(original) / factorLow
			lambdaHigh := float32(original) / factorHigh

			lambda := 2 * math.Pi * math.Pow(float64(p.RopeTheta), float64(i)/float64(dim))
			if lambda < float64(lambdaHigh) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0)
			} else if lambda > float64(lambdaLow) {
				p.RopeScaling.factors = append(p.RopeScaling.factors, factor)
			} else {
				smooth := (float32(original)/float32(lambda) - factorLow) / (factorHigh - factorLow)
				p.RopeScaling.factors = append(p.RopeScaling.factors, 1.0/((1-smooth)/factor+smooth))
			}
		}
	}

	if p.NumKeyValueHeads > 0 {
		kv["granite.attention.head_count_kv"] = p.NumKeyValueHeads
	}

	if p.RMSNormEPS > 0 {
		kv["granite.attention.layer_norm_rms_epsilon"] = p.RMSNormEPS
	}

	if layerNormEpsilon := cmp.Or(p.LayerNormEPS, p.LayerNormEpsilon, p.NormEpsilon); layerNormEpsilon > 0 {
		kv["granite.attention.layer_norm_epsilon"] = layerNormEpsilon
	}

	// IBM granite specific scalers
	if p.EmbeddingScale > 0 {
		kv["granite.embedding_scale"] = p.EmbeddingScale
	}
	if p.LogitsScale > 0 {
		kv["granite.logit_scale"] = p.LogitsScale
	}
	if p.ResidualScale > 0 {
		kv["granite.residual_scale"] = p.ResidualScale
	}
	if p.AttentionScale > 0 {
		kv["granite.attention.scale"] = p.AttentionScale
	}
	return kv
}
