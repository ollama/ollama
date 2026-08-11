package mlxrunner

import (
	"math"
	"slices"

	"github.com/ollama/ollama/llm"
)

func removeMaskedTopLogprobs(logprobs []llm.Logprob) {
	for i := range logprobs {
		logprobs[i].TopLogprobs = slices.DeleteFunc(logprobs[i].TopLogprobs, func(token llm.TokenLogprob) bool {
			return math.IsInf(token.Logprob, -1)
		})
	}
}
