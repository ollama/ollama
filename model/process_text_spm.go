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
		symbols := make([]symbol, len(runes))
		for i, r := range runes {
			symbols[i] = symbol{
				text: string(r),
				prev: i - 1,
				next: i + 1,
			}
		}

		if len(symbols) == 0 {
			continue
		}

		symbols[len(symbols)-1].next = -1

		history := make(map[string][2]int)

		for i := 0; i < len(symbols)-1; i++ {
			left := &symbols[i]
			right := &symbols[i+1]

			combined := left.text + right.text
			id := spm.vocab.Encode(combined)

			if id >= 0 && id < int32(len(spm.vocab.Scores)) {
				heap.Push(q, &candidate{
					left:  i,
					right: i + 1,
					score: spm.vocab.Scores[id],
					size:  len(combined),
				})
				history[combined] = [2]int{i, i + 1}
			}
		}

		// Process bigrams in order of score
		for q.Len() > 0 {
			bg := heap.Pop(q).(*candidate)
			left := &symbols[bg.left]
			right := &symbols[bg.right]

			if left.text == "" || right.text == "" || len(left.text)+len(right.text) != bg.size {
				continue
			}

			left.text += right.text
			right.text = ""

			left.next = right.next
			if right.next != -1 {
				symbols[right.next].prev = bg.left
			}

			// Add new bigrams with updated left node
			if left.prev != -1 {
				prevSym := &symbols[left.prev]
				if prevSym.text != "" {
					combined := prevSym.text + left.text
					id := spm.vocab.Encode(combined)

					if id >= 0 && id < int32(len(spm.vocab.Scores)) {
						heap.Push(q, &candidate{
							left:  left.prev,
							right: bg.left,
							score: spm.vocab.Scores[id],
							size:  len(combined),
						})
						history[combined] = [2]int{left.prev, bg.left}
					}
				}
			}

			if left.next != -1 {
				nextSym := &symbols[left.next]
				if nextSym.text != "" {
					combined := left.text + nextSym.text
					id := spm.vocab.Encode(combined)

					if id >= 0 && id < int32(len(spm.vocab.Scores)) {
						heap.Push(q, &candidate{
							left:  bg.left,
							right: left.next,
							score: spm.vocab.Scores[id],
							size:  len(combined),
						})
						history[combined] = [2]int{bg.left, left.next}
					}
				}
			}
		}

		// Helper function to recursively segment tokens
		var resegment func(string, []symbol) []int32
		resegment = func(text string, symbols []symbol) []int32 {
			id := spm.vocab.Encode(text)

			if id >= 0 {
				return []int32{id}
			}

			if pair, exists := history[text]; exists {
				left := resegment(symbols[pair[0]].text, symbols)
				right := resegment(symbols[pair[1]].text, symbols)
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
		for i := 0; i != -1; i = symbols[i].next {
			if symbols[i].text != "" {
				tokens := resegment(symbols[i].text, symbols)
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

type symbol struct {
	text string
	prev int
	next int
}

type candidate struct {
	left, right int
	score       float32
	size        int
}

type queue []*candidate

func (q queue) Len() int { return len(q) }

func (q queue) Less(i, j int) bool {
	return (q[i].score > q[j].score) || (q[i].score == q[j].score && q[i].left < q[j].left)
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
