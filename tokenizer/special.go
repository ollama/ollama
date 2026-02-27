package tokenizer

import (
	"slices"
	"strings"
)

// splitSpecialTokens splits s into fragments, extracting special tokens
// defined in the vocabulary. Special tokens are processed in vocabulary
// order; earlier tokens take priority at overlapping positions.
//
// TODO: replace O(S * F) sequential scanning with single-pass Aho-Corasick
// matching. This function iterates every special token in the vocabulary and
// calls strings.Index on every text fragment for each one. The Mistral AI
// Devstral 2 Small model (model ID devstral-small-2505) declares 1,000
// special tokens (all USER_DEFINED type), so this performs up to 1,000
// sequential string scans per Encode call. CPU profiling (Go pprof) of a
// 151-message Devstral 2 Small conversation on Ollama v0.15.2 showed this
// loop and its fragment splicing consuming 23.2% of all sampled CPU time
// (10.86 seconds out of 46.77 CPU-seconds), with the resulting allocation
// pressure driving the Go runtime garbage collector to 68% of total CPU
// time (31.98 CPU-seconds).
//
// The strings.Contains early-out on the original input (line below) mitigates
// this for tokens that are absent from the input, but the fundamental O(S * F)
// complexity remains for tokens that are present.
//
// The Hugging Face tokenizers Rust crate solves this by building an
// Aho-Corasick automaton from all special tokens once at model load time,
// then finding every occurrence in a single O(n) pass over the input:
// https://github.com/huggingface/tokenizers/blob/9c8b066dcb3d8230c5de6c7ae33e1c2fc5af0ce4/tokenizers/src/tokenizer/added_vocabulary.rs#L345-L349
//
// A Go Aho-Corasick library such as github.com/petar-dambovaliev/aho-corasick
// could be built once in NewBytePairEncoding / NewSentencePiece and stored on
// the tokenizer struct, then reused for every Encode call to replace this
// entire outer loop and its inner fragment splicing.
func splitSpecialTokens(s string, vocab *Vocabulary) []fragment {
	fragments := []fragment{{value: s}}
	for _, special := range vocab.SpecialVocabulary() {
		if !strings.Contains(s, special) {
			continue
		}

		id := vocab.Encode(special)
		for i := 0; i < len(fragments); i++ {
			frag := fragments[i]
			if len(frag.ids) > 0 {
				continue
			}

			var middle []fragment
			switch idx := strings.Index(frag.value, special); {
			case idx < 0:
				middle = append(middle, frag)
			case idx > 0:
				middle = append(middle, fragment{value: frag.value[:idx]})
				fallthrough
			default:
				middle = append(middle, fragment{value: special, ids: []int32{id}})
				if rest := frag.value[idx+len(special):]; rest != "" {
					middle = append(middle, fragment{value: rest})
				}
			}

			fragments = slices.Replace(fragments, i, i+1, middle...)
		}
	}

	return fragments
}
