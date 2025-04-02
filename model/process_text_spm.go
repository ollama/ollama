package model

import (
	"container/heap"
	"fmt"
	"log/slog"
	"strings"
)

const WhitespaceSeparator = "▁"

type SentencePieceModel struct {
	maxTokenLen int
	vocab       *Vocabulary
}

var _ TextProcessor = (*SentencePieceModel)(nil)

func NewSentencePieceModel(vocab *Vocabulary) SentencePieceModel {
	slog.Debug("Tokens", "num tokens", len(vocab.Values), "vals", vocab.Values[:5], "scores", vocab.Scores[:5], "types", vocab.Types[:5])

	counter := map[int]int{}
	var maxTokenLen int
	for cnt := range vocab.Types {
		switch vocab.Types[cnt] {
		case TOKEN_TYPE_NORMAL, TOKEN_TYPE_USER_DEFINED, TOKEN_TYPE_UNUSED:
			maxTokenLen = max(maxTokenLen, len(vocab.Values[cnt]))
			fallthrough
		default:
			counter[int(vocab.Types[cnt])] += 1
		}
	}

	slog.Debug("Token counts", "normal", counter[TOKEN_TYPE_NORMAL], "unknown", counter[TOKEN_TYPE_UNKNOWN], "control", counter[TOKEN_TYPE_CONTROL],
		"user defined", counter[TOKEN_TYPE_USER_DEFINED], "unused", counter[TOKEN_TYPE_UNUSED], "byte", counter[TOKEN_TYPE_BYTE],
		"max token len", maxTokenLen)

	return SentencePieceModel{
		maxTokenLen: maxTokenLen,
		vocab:       vocab,
	}
}

func (spm SentencePieceModel) Is(id int32, special Special) bool {
	return spm.vocab.Is(id, special)
}

func (spm SentencePieceModel) Encode(s string, addSpecial bool) ([]int32, error) {
	fragments := []fragment{{value: s}}
	for _, special := range spm.vocab.SpecialVocabulary() {
		id := spm.vocab.Encode(special)
		for i := 0; i < len(fragments); i++ {
			frag := fragments[i]
			if len(frag.ids) > 0 {
				continue
			}

			var middle []fragment
			switch i := strings.Index(frag.value, special); {
			case i < 0:
				middle = append(middle, frag)
			case i > 0:
				middle = append(middle, fragment{value: frag.value[:i]})
				fallthrough
			default:
				middle = append(middle, fragment{value: special, ids: []int32{id}})
				if rest := frag.value[i+len(special):]; rest != "" {
					middle = append(middle, fragment{value: rest})
				}
			}

			fragments = append(fragments[:i], append(middle, fragments[i+1:]...)...)
		}
	}

	var ids []int32
	for _, frag := range fragments {
		if len(frag.ids) > 0 {
			ids = append(ids, frag.ids...)
			continue
		}

		text := strings.ReplaceAll(frag.value, " ", WhitespaceSeparator)

		if id := spm.vocab.Encode(text); id >= 0 {
			ids = append(ids, id)
			continue
		}

		q := &queue{}
		heap.Init(q)

		runes := []rune(text)
		merges := make([]merge, len(runes))
		for i, r := range runes {
			merges[i] = merge{
				p:     i - 1,
				n:     i + 1,
				runes: []rune{r},
			}
		}

		merges[len(merges)-1].n = -1

		pairwise := func(a, b int) *candidate {
			if a < 0 || b >= len(runes) {
				return nil
			}

			left, right := string(merges[a].runes), string(merges[b].runes)
			if id := spm.vocab.Encode(left + right); id >= 0 {
				return &candidate{
					a:     a,
					b:     b,
					score: spm.vocab.Scores[id],
					size:  len(left) + len(right),
				}
			}

			return nil
		}

		history := make(map[string][2]int)

		for i := range len(runes) - 1 {
			if pair := pairwise(i, i+1); pair != nil {
				heap.Push(q, pair)
			}
		}

		for q.Len() > 0 {
			c := heap.Pop(q).(*candidate)
			left, right := &merges[c.a], &merges[c.b]

			if string(left.runes) == "" || string(right.runes) == "" || len(string(left.runes))+len(string(right.runes)) != c.size {
				continue
			}

			left.runes = append(left.runes, right.runes...)
			right.runes = nil

			left.n = right.n
			if right.n != -1 {
				merges[right.n].p = c.a
			}

			// Add new bigrams with updated left node
			if left.p != -1 {
				prevSym := &merges[left.p]
				if string(prevSym.runes) != "" {
					combined := string(prevSym.runes) + string(left.runes)
					id := spm.vocab.Encode(combined)

					if id >= 0 && id < int32(len(spm.vocab.Scores)) {
						heap.Push(q, &candidate{
							a:     left.p,
							b:     c.a,
							score: spm.vocab.Scores[id],
							size:  len(combined),
						})
						history[combined] = [2]int{left.p, c.a}
					}
				}
			}

			if left.n != -1 {
				nextSym := &merges[left.n]
				if string(nextSym.runes) != "" {
					combined := string(left.runes) + string(nextSym.runes)
					id := spm.vocab.Encode(combined)

					if id >= 0 && id < int32(len(spm.vocab.Scores)) {
						heap.Push(q, &candidate{
							a:     c.a,
							b:     left.n,
							score: spm.vocab.Scores[id],
							size:  len(combined),
						})
						history[combined] = [2]int{c.a, left.n}
					}
				}
			}
		}

		// Helper function to recursively segment tokens
		var resegment func(string, []merge) []int32
		resegment = func(text string, merges []merge) []int32 {
			id := spm.vocab.Encode(text)

			if id >= 0 {
				return []int32{id}
			}

			if pair, exists := history[text]; exists {
				left := resegment(string(merges[pair[0]].runes), merges)
				right := resegment(string(merges[pair[1]].runes), merges)
				return append(left, right...)
			}

			// Fallback to byte tokenization
			var result []int32
			for _, b := range []byte(text) {
				byteToken := fmt.Sprintf("<0x%02X>", b)
				unknownID := spm.vocab.Encode(byteToken)
				if unknownID >= 0 {
					result = append(result, unknownID)
				} else {
					slog.Debug("unknown byte token", "byte", b, "token", byteToken)
				}
			}

			return result
		}

		// Collect tokens from the merged symbols
		for i := 0; i != -1; i = merges[i].n {
			if string(merges[i].runes) != "" {
				tokens := resegment(string(merges[i].runes), merges)
				ids = append(ids, tokens...)
			}
		}
	}

	if addSpecial && len(ids) > 0 {
		if spm.vocab.AddBOS {
			if ids[0] == spm.vocab.BOS {
				slog.Warn("adding bos token to prompt which already has it", "id", spm.vocab.BOS)
			}

			slog.Debug("adding bos token to prompt", "id", spm.vocab.BOS)
			ids = append([]int32{spm.vocab.BOS}, ids...)
		}

		if spm.vocab.AddEOS {
			if ids[len(ids)-1] == spm.vocab.EOS {
				slog.Warn("adding eos token to prompt which already has it", "id", spm.vocab.EOS)
			}

			slog.Debug("adding eos token to prompt", "id", spm.vocab.EOS)
			ids = append(ids, spm.vocab.EOS)
		}
	}

	return ids, nil
}

type candidate struct {
	a, b  int
	score float32
	size  int
}

type queue []*candidate

func (q queue) Len() int { return len(q) }

func (q queue) Less(i, j int) bool {
	return (q[i].score > q[j].score) || (q[i].score == q[j].score && q[i].a < q[j].a)
}

func (q queue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

func (q *queue) Push(x interface{}) {
	item := x.(*candidate)
	*q = append(*q, item)
}

func (q *queue) Pop() interface{} {
	old := *q
	n := len(old)
	item := old[n-1]
	*q = old[0 : n-1]
	return item
}

func (spm SentencePieceModel) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		data := spm.vocab.Decode(id)
		data = strings.ReplaceAll(data, WhitespaceSeparator, " ")
		if _, err := sb.WriteString(data); err != nil {
			return "", err
		}
	}

	return sb.String(), nil
}
