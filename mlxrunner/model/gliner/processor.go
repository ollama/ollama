package gliner

import (
	"cmp"
	"slices"
	"unicode"

	"github.com/ollama/ollama/api"
)

type word struct {
	text                           string
	start, end, byteStart, byteEnd int
}
type prepared struct {
	ids, wordIndices, labelIndices []int32
	words                          []word
	spans                          [][2]int
}

// splitWords matches Python's Unicode \w+(?:[-_]\w+)*|\S and retains
// both byte offsets for slicing and code-point offsets for the public API.
func splitWords(text string) []word {
	runes := []rune(text)
	byteOffsets := make([]int, 0, len(runes)+1)
	for i := range text {
		byteOffsets = append(byteOffsets, i)
	}
	byteOffsets = append(byteOffsets, len(text))
	isWord := func(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_' }
	var words []word
	for i := 0; i < len(runes); {
		// Python's Unicode whitespace also includes the four ASCII record
		// separators, which Go's unicode.IsSpace deliberately excludes.
		if unicode.IsSpace(runes[i]) || (runes[i] >= '\u001c' && runes[i] <= '\u001f') {
			i++
			continue
		}
		start := i
		if isWord(runes[i]) {
			i++
			for i < len(runes) {
				if isWord(runes[i]) {
					i++
					continue
				}
				if runes[i] == '-' && i+1 < len(runes) && isWord(runes[i+1]) {
					i += 2
					continue
				}
				break
			}
		} else {
			i++
		}
		words = append(words, word{text[byteOffsets[start]:byteOffsets[i]], start, i, byteOffsets[start], byteOffsets[i]})
	}
	return words
}

func (m *Model) prepare(text string, labels []string) (*prepared, error) {
	p := &prepared{words: splitWords(text), ids: []int32{m.cls}}
	if len(p.words) > m.cfg.MaxLength {
		return nil, badInput("input has %d words; model limit is %d (split the input into smaller chunks)", len(p.words), m.cfg.MaxLength)
	}
	for _, label := range labels {
		if m.tok.HasAddedToken(label) {
			return nil, badInput("label %q contains a reserved token", label)
		}
		p.labelIndices = append(p.labelIndices, int32(len(p.ids)))
		p.ids = append(p.ids, m.ent)
		ids := m.tok.EncodeWord(label)
		if len(ids) == 0 {
			return nil, badInput("label %q is empty after normalization", label)
		}
		p.ids = append(p.ids, ids...)
	}
	p.ids = append(p.ids, m.textSep)
	for i, w := range p.words {
		ids := m.tok.EncodeWord(w.text)
		if len(ids) == 0 {
			return nil, badInput("word at offset %d is empty after normalization", w.start)
		}
		p.wordIndices = append(p.wordIndices, int32(len(p.ids)))
		p.ids = append(p.ids, ids...)
		for width := 0; width < m.cfg.MaxWidth && i+width < len(p.words); width++ {
			p.spans = append(p.spans, [2]int{i, i + width})
		}
	}
	p.ids = append(p.ids, m.sep)
	if len(p.ids) > m.MaxContextLength() {
		return nil, badInput("input and labels require %d tokens; model limit is %d (split the input or use fewer labels)", len(p.ids), m.MaxContextLength())
	}
	return p, nil
}

// decode performs GLiNER's default flat, single-label greedy decoding.
func decode(text string, labels []string, words []word, spans [][2]int, scores []float32, threshold float32) []api.Entity {
	type candidate struct {
		start, end, label int
		score             float32
	}
	var candidates []candidate
	for i, span := range spans {
		for j := range labels {
			score := scores[i*len(labels)+j]
			if score > threshold {
				candidates = append(candidates, candidate{span[0], span[1], j, score})
			}
		}
	}
	slices.SortStableFunc(candidates, func(a, b candidate) int { return cmp.Compare(b.score, a.score) })
	occupied := make([]bool, len(words))
	entities := make([]api.Entity, 0)
	for _, c := range candidates {
		if slices.Contains(occupied[c.start:c.end+1], true) {
			continue
		}
		for i := c.start; i <= c.end; i++ {
			occupied[i] = true
		}
		a, b := words[c.start], words[c.end]
		entities = append(entities, api.Entity{Text: text[a.byteStart:b.byteEnd], Label: labels[c.label], Start: a.start, End: b.end, Score: c.score})
	}
	slices.SortFunc(entities, func(a, b api.Entity) int { return cmp.Compare(a.Start, b.Start) })
	return entities
}
