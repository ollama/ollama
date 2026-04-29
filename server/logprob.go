package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// toAPILogprobs converts llm.Logprobs to api.Logprobs
func toAPILogprobs(logprobs []llm.Logprob) []api.Logprob {
	result := make([]api.Logprob, len(logprobs))
	for i, lp := range logprobs {
		result[i] = api.Logprob{
			TokenLogprob: api.TokenLogprob{
				Token:   lp.Token,
				Bytes:   bytesToInts(lp.Bytes),
				Logprob: lp.Logprob,
			},
		}
		if len(lp.TopLogprobs) > 0 {
			result[i].TopLogprobs = make([]api.TokenLogprob, len(lp.TopLogprobs))
			for j, tlp := range lp.TopLogprobs {
				result[i].TopLogprobs[j] = api.TokenLogprob{
					Token:   tlp.Token,
					Bytes:   bytesToInts(tlp.Bytes),
					Logprob: tlp.Logprob,
				}
			}
		}
	}
	return result
}

// bytesToInts converts a byte slice to an int slice.
// This function uses the raw bytes stored in llm.TokenLogprob.Bytes,
// which preserves partial UTF-8 sequences that would otherwise be
// corrupted during JSON marshaling/unmarshaling.
func bytesToInts(b []byte) []int {
	if len(b) == 0 {
		return nil
	}

	ints := make([]int, len(b))
	for i, v := range b {
		ints[i] = int(v)
	}
	return ints
}

// stringToByteInts converts a string to an int slice of bytes.
// This is kept for backward compatibility with tests.
func stringToByteInts(s string) []int {
	if s == "" {
		return nil
	}

	raw := []byte(s)
	ints := make([]int, len(raw))
	for i, b := range raw {
		ints[i] = int(b)
	}
	return ints
}
