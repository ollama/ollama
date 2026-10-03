package tokenizer

import (
	"runtime"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const (
	encodeParallelMinInputBytes      = 4 * 1024
	encodeParallelMinChunksPerWorker = 8
)

type encodeChunk struct {
	text      string
	isSpecial bool
}

// splitBySpecialTokens splits text into parts, keeping special tokens as separate elements
func (t *Tokenizer) splitBySpecialTokens(s string) []encodeChunk {
	if s == "" {
		return nil
	}
	if len(t.specialTokens) == 0 {
		return []encodeChunk{{text: s}}
	}

	tokens := t.sortedSpecialTokens
	if len(tokens) == 0 {
		// Fallback for tokenizers constructed outside the loaders.
		tokens = make([]string, 0, len(t.specialTokens))
		for tok := range t.specialTokens {
			tokens = append(tokens, tok)
		}
		sort.Slice(tokens, func(i, j int) bool {
			return len(tokens[i]) > len(tokens[j])
		})
	}

	var result []encodeChunk
	remaining := s

	for len(remaining) > 0 {
		found := false
		for _, tok := range tokens {
			if strings.HasPrefix(remaining, tok) {
				result = append(result, encodeChunk{text: tok, isSpecial: true})
				remaining = remaining[len(tok):]
				found = true
				break
			}
		}
		if !found {
			// Find next special token position
			nextPos := len(remaining)
			for _, tok := range tokens {
				if idx := strings.Index(remaining, tok); idx != -1 && idx < nextPos {
					nextPos = idx
				}
			}
			if nextPos > 0 {
				result = append(result, encodeChunk{text: remaining[:nextPos]})
			}
			remaining = remaining[nextPos:]
		}
	}

	return result
}

func (t *Tokenizer) forEachPartChunk(part string, fn func(encodeChunk)) {
	if t.pretokenizer == nil {
		if t.metaspace != nil && t.metaspace.Split {
			start := 0
			for i, r := range part {
				if i > 0 && r == '▁' {
					fn(encodeChunk{text: part[start:i]})
					start = i
				}
			}
			fn(encodeChunk{text: part[start:]})
			return
		}
		fn(encodeChunk{text: part, isSpecial: false})
		return
	}

	parts := []string{part}
	for _, stage := range t.pretokenizer {
		var next []string
		for _, text := range parts {
			next = append(next, stage.split(text)...)
		}
		parts = next
	}
	for _, text := range parts {
		fn(encodeChunk{text: text})
	}
}

func (t *Tokenizer) appendEncodedChunk(ids []int32, c encodeChunk) []int32 {
	if c.isSpecial {
		if id, ok := t.specialTokens[c.text]; ok {
			return append(ids, id)
		}
		return ids
	}

	return t.encodeChunkInto(c.text, ids)
}

// Encode tokenizes text to token IDs.
// Parallel encoding is used only for very large inputs with enough chunks per worker.
func (t *Tokenizer) Encode(s string, addBOS bool) []int32 {
	// Preserve original added-token identity. Only normalized=true tokens may
	// also be recognized after normalization, before pretokenizer prefixes.
	parts := t.splitBySpecialTokens(s)
	for i, part := range parts {
		if part.isSpecial {
			continue
		}
		if t.normalizeNFC {
			part.text = norm.NFC.String(part.text)
		}
		if t.normalizeSpaces {
			part.text = strings.ReplaceAll(part.text, " ", "▁")
		}
		if t.normalizedTokens[part.text] {
			part.isSpecial = true
			parts[i] = part
			continue
		}
		if t.metaspace != nil {
			part.text = strings.ReplaceAll(part.text, " ", "▁")
			if (t.metaspace.PrependScheme == "always" || (t.metaspace.PrependScheme == "first" && i == 0)) && !strings.HasPrefix(part.text, "▁") {
				part.text = "▁" + part.text
			}
		}
		parts[i] = part
	}

	// Fast path: encode sequentially without materializing chunk slices.
	if len(s) < encodeParallelMinInputBytes {
		var ids []int32
		for _, part := range parts {
			if part.isSpecial {
				ids = t.appendEncodedChunk(ids, part)
				continue
			}
			t.forEachPartChunk(part.text, func(c encodeChunk) {
				ids = t.appendEncodedChunk(ids, c)
			})
		}

		if addBOS && t.vocab.BOS >= 0 {
			ids = append([]int32{t.vocab.BOS}, ids...)
		}
		return ids
	}

	// For large inputs collect chunks to enable parallel processing.
	var allChunks []encodeChunk
	for _, part := range parts {
		if part.isSpecial {
			allChunks = append(allChunks, part)
			continue
		}
		t.forEachPartChunk(part.text, func(c encodeChunk) {
			allChunks = append(allChunks, c)
		})
	}

	// Encode chunks. Use the parallel path only when the chunk count is
	// large enough to amortize goroutine/synchronization overhead.
	useParallel := true
	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers > len(allChunks) {
		numWorkers = len(allChunks)
	}
	if numWorkers < 2 || len(allChunks) < numWorkers*encodeParallelMinChunksPerWorker {
		useParallel = false
	}

	var ids []int32
	if !useParallel {
		for _, c := range allChunks {
			ids = t.appendEncodedChunk(ids, c)
		}
	} else {
		chunksPer := (len(allChunks) + numWorkers - 1) / numWorkers
		results := make([][]int32, numWorkers)
		var wg sync.WaitGroup

		for i := range numWorkers {
			start := i * chunksPer
			end := start + chunksPer
			if end > len(allChunks) {
				end = len(allChunks)
			}
			if start >= end {
				continue
			}

			wg.Add(1)
			go func(i int, chunks []encodeChunk) {
				defer wg.Done()
				var r []int32
				for _, c := range chunks {
					r = t.appendEncodedChunk(r, c)
				}
				results[i] = r
			}(i, allChunks[start:end])
		}
		wg.Wait()

		for _, r := range results {
			ids = append(ids, r...)
		}
	}

	if addBOS && t.vocab.BOS >= 0 {
		ids = append([]int32{t.vocab.BOS}, ids...)
	}
	return ids
}

// encodeChunkInto appends encoded tokens to ids and returns the extended slice.
// Uses BPE merge algorithm for both BPE and SentencePiece tokenization.
func (t *Tokenizer) encodeChunkInto(s string, ids []int32) []int32 {
	if s == "" {
		return ids
	}

	// Apply encoding transformation
	// SentencePiece: replace space with ▁
	// BPE: convert bytes using precomputed table (GPT-2 byte-level encoding)
	var encoded string
	if t.typ == TokenizerSentencePiece {
		encoded = strings.ReplaceAll(s, " ", "▁")
	} else {
		var sb strings.Builder
		sb.Grow(len(s) * 2)
		for i := range len(s) {
			sb.WriteRune(byteToRune[s[i]])
		}
		encoded = sb.String()
	}

	// A whole-vocabulary match can bypass ranked merges only when the model
	// requests it. Otherwise a higher-priority pair may split that same word.
	if t.ignoreMerges || utf8.RuneCountInString(encoded) == 1 {
		if id, ok := t.vocab.Reverse[encoded]; ok {
			return append(ids, id)
		}
	}

	return t.encodeBPEMerge(encoded, ids)
}
