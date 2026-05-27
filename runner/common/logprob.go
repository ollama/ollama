package common

import (
	"math"
	"sort"

	"github.com/ollama/ollama/llm"
)

// TokenDecoderFunc is a function that converts token IDs to text.
type TokenDecoderFunc func(tokenID int) string

// CalculateLogprobs converts raw logits to log probabilities and finds top K tokens.
// It uses numerically stable softmax to compute log probabilities.
// If temperature > 0, logits are scaled by temperature before computing probabilities,
// matching the distribution used during sampling.
func CalculateLogprobs(logits []float32, selectedToken int, topK int, temperature float32, decoder TokenDecoderFunc) []llm.Logprob {
	if len(logits) == 0 {
		return nil
	}

	// Apply temperature scaling to match the sampling distribution
	scaledLogits := make([]float32, len(logits))
	copy(scaledLogits, logits)
	if temperature > 0 {
		temp := max(temperature, 1e-7)
		for i := range scaledLogits {
			scaledLogits[i] = scaledLogits[i] / temp
		}
	}

	// Step 1: Convert logits to log probabilities using numerically stable softmax
	maxLogit := scaledLogits[0]
	for _, logit := range scaledLogits[1:] {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	var sumExp float64
	for _, logit := range scaledLogits {
		sumExp += math.Exp(float64(logit - maxLogit))
	}
	logSumExp := float32(math.Log(sumExp))

	logProbs := make([]float32, len(scaledLogits))
	for i, logit := range scaledLogits {
		logProbs[i] = (logit - maxLogit) - logSumExp
	}

	// Step 2: Get selected token's information
	selectedLogprob := logProbs[selectedToken]
	selectedText := decoder(selectedToken)

	result := llm.Logprob{
		TokenLogprob: llm.TokenLogprob{
			Token:   selectedText,
			Logprob: float64(selectedLogprob),
		},
	}

	// Step 3: If topK requested, find the top K tokens
	if topK > 0 {
		type tokenLogprobPair struct {
			tokenID int
			logprob float32
		}

		pairs := make([]tokenLogprobPair, len(logProbs))
		for i, lp := range logProbs {
			pairs[i] = tokenLogprobPair{tokenID: i, logprob: lp}
		}

		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].logprob > pairs[j].logprob
		})

		k := min(topK, len(pairs))
		topLogprobs := make([]llm.TokenLogprob, k)
		for i := range k {
			tokenText := decoder(pairs[i].tokenID)
			topLogprobs[i] = llm.TokenLogprob{
				Token:   tokenText,
				Logprob: float64(pairs[i].logprob),
			}
		}
		result.TopLogprobs = topLogprobs
	}

	return []llm.Logprob{result}
}
