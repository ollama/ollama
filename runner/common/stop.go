package common

import (
	"strings"

	"github.com/ollama/ollama/llm"
)

func FindStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
			return true, stop
		}
	}

	return false, ""
}

func ContainsStopSuffix(sequence string, stops []string) bool {
	for _, stop := range stops {
		for i := 1; i <= len(stop); i++ {
			if strings.HasSuffix(sequence, stop[:i]) {
				return true
			}
		}
	}

	return false
}

// TruncateStop removes the provided stop string from pieces,
// returning the partial pieces with stop removed, including truncating
// the last piece if required (and signalling if this was the case)
func TruncateStop(resps []llm.CompletionResponse, stop string) ([]llm.CompletionResponse, bool) {
	var sequence string
	for _, resp := range resps {
		sequence += resp.Content
	}

	idx := strings.Index(sequence, stop)
	if idx < 0 {
		return resps, false
	}

	truncated := sequence[:idx]
	if len(truncated) == 0 {
		return nil, true
	}

	result := make([]llm.CompletionResponse, 0, len(resps))

	// Track position in truncated sequence
	pos := 0
	truncationHappened := false
	for _, resp := range resps {
		if pos >= len(truncated) {
			break
		}

		chunk := truncated[pos:min(pos+len(resp.Content), len(truncated))]
		if len(chunk) < len(resp.Content) {
			truncationHappened = true
		}
		if len(chunk) > 0 {
			result = append(result, llm.CompletionResponse{Content: chunk})
		}
		pos += len(resp.Content)
	}

	return result, truncationHappened
}
