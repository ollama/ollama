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
				Bytes:   stringToByteInts(lp.Token),
				Logprob: lp.Logprob,
			},
		}
		if len(lp.TopLogprobs) > 0 {
			result[i].TopLogprobs = make([]api.TokenLogprob, len(lp.TopLogprobs))
			for j, tlp := range lp.TopLogprobs {
				result[i].TopLogprobs[j] = api.TokenLogprob{
					Token:   tlp.Token,
					Bytes:   stringToByteInts(tlp.Token),
					Logprob: tlp.Logprob,
				}
			}
		}
		if len(lp.RequestedLogprobs) > 0 {
			result[i].RequestedLogprobs = make([]api.TokenLogprob, len(lp.RequestedLogprobs))
			for j, rlp := range lp.RequestedLogprobs {
				result[i].RequestedLogprobs[j] = api.TokenLogprob{
					Token:   rlp.Token,
					Bytes:   stringToByteInts(rlp.Token),
					Logprob: rlp.Logprob,
				}
			}
		}
	}
	return result
}

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
